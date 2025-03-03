# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch

from torch.distributed.elastic.multiprocessing.errors import record
import torch.distributed.checkpoint as dcp
import torch.distributed as dist
from torch.distributed.tensor import distribute_module, distribute_tensor, DTensor, Replicate, Shard

from torchtitan import utils
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer, build_hf_processor
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
from torchtitan.models import model_name_to_tokenizer
from torchtitan.parallelisms import ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.train_spec import get_train_spec
from torchtitan.utils import device_module, device_type, import_module_from_path
from huggingface_hub import snapshot_download


def set_nested_attr(obj, name, value):
    """since model.register_buffer() doesn't work on module names with '.',
       manually set a neste attribute for buffers"""
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    world_size = 8
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    #device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device = torch.device("cuda")
    #device_module.set_device(device)
    #utils.init_distributed(job_config)

    train_spec = get_train_spec(job_config.model.name)
    model_name = job_config.model.name

    if job_config.training.dataset == "alfred":
        processor = build_hf_processor(model_name)
        tokenizer = processor.tokenizer
        img_data_dir = snapshot_download(repo_id="bosungkim/alfred-small-img", repo_type="dataset", allow_patterns="*.tar", local_dir='data/alfred-small-img')
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})

        # TODO incorporate with build_hf_data_loader
        from torchtitan.datasets.alfred_dataset import ALFREDDataset, AlfredDataLoader
        dataset = ALFREDDataset(processor=processor, img_data_dir=img_data_dir, max_seq_len=job_config.training.seq_len, world_size=world_size)
        data_loader = AlfredDataLoader(0, dataset, 
                                        batch_size=job_config.training.batch_size,
                                        world_size=world_size)
    else:
        # build tokenizer
        tokenizer_type = model_name_to_tokenizer[model_name]
        tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)
        # build dataloader
        data_loader = build_hf_data_loader(
            job_config.training.dataset,
            job_config.training.dataset_path,
            tokenizer,
            job_config.training.batch_size,
            job_config.training.seq_len,
            dp_degree,
            dp_rank,
        )

    # build model (using meta init)
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words if hasattr(tokenizer, "n_words") else tokenizer.vocab_size
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(
        f"Building {train_spec.name} {job_config.model.flavor} with {model_config}"
    )

    # model_dtype = torch.bfloat16 -> woah!@ 
    # lot of stufff to figure out: activation checkpointing enabled.
    # mainly bc of huggingfaces class related to past_key_values and DynamicCache()
    # TODO: change model code with native Llava classes
    #model_dtype = torch.float16 # need to do torch.autocast with lm_head -> this leads nan loss in DDP eventually
    model_dtype = torch.float32

    if 'llava' in model_name: # need to save buffers (position embeddings, layer norm statistics, etc.)
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype)
        buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        del model

    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    #with torch.device("meta"):
    with torch.device("cuda"):
        if 'llava' in model_name:
            # using different attn_implementation really matters for TP, PP, CP, etc.
            #model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, attn_implementation="eager")
            model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, device_map="auto")
            model.resize_token_embeddings(len(processor.tokenizer))
        else:
            model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = utils.get_num_params(model)
    # num_flop_per_token = utils.get_num_flop_per_token(
    #     utils.get_num_params(model),
    #     model_config,
    #     job_config.training.seq_len,
    # )
    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1).float(), labels.flatten(0, 1)
        )

    # TODO: compiling loss function causes CUDA errors, turning off for now
    # if job_config.training.compile:
    #     loss_fn = torch.compile(loss_fn)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    checkpoint_path = 'distributed_checkpoint/'

    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    # train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
    # model.to_empty(device=init_device)    
    # with torch.no_grad():
    #     dcp.load(model.state_dict(), checkpoint_id=checkpoint_path)
    #     # Restore buffers
    #     for name, buffer in buffers_dict.items():
    #         set_nested_attr(model, name, buffer.to(device_type))
    model.to(model_dtype)
    model.train()
    model_parts = [model]

    data_iterator = iter(data_loader)

    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()

    step = 0
    if True:
        while step < job_config.training.steps:
            step += 1

            # get batch
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
            breakpoint()
            input_ids, labels = batch['input_ids'], batch['labels']
            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)

            pixel_values=batch['pixel_values'].to(device_type, model_dtype)
            image_sizes=batch['image_sizes'].to(device_type, model_dtype)

            inputs_embeds = model.embed(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes)
            
            del input_ids, pixel_values, image_sizes

            # since nn.Embedding is not sharded bc of mask_scatter for pixel_values, manually shard here.
            # if not (parallel_dims.cp_enabled and (parallel_dims.pp_enabled and has_first_stage)):
            if parallel_dims.tp_enabled:
                if parallel_dims.pp_enabled and not has_first_stage:
                    pass
                else:
                    if not isinstance(inputs_embeds, torch.Tensor):
                        inputs_embeds = inputs_embeds.wait()
                    inputs_embeds = distribute_tensor(inputs_embeds, world_mesh['tp'], placements=[Shard(1)])


            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(inputs_embeds, target=targets, losses=losses)
                    else:
                        pp_schedule.step(target=targets, losses=losses)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    output = model.language_model(inputs_embeds=inputs_embeds, use_cache=False)
                    loss = loss_fn(output.logits, labels)
                    logger.info(f"step: {step:2} {color.yellow}{loss}{color.reset}")
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del output
                    loss.backward()

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters() if isinstance(p, DTensor)],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters() if not isinstance(p, DTensor)],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            # log metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(loss, world_mesh["dp_cp"]),
                        utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second per device, abbreviated as tps
                tps = ntokens_since_last_log / (
                    time_delta * parallel_dims.non_data_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                # mfu = 100 * num_flop_per_token * tps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                device_mem_stats = device_memory_monitor.get_peak_stats()

                metrics = {
                    "loss_metrics/global_avg_loss": global_avg_loss,
                    "loss_metrics/global_max_loss": global_max_loss,
                    "throughput(tps)": tps,
                    # "mfu(%)": mfu,
                    "time_metrics/end_to_end(s)": time_end_to_end,
                    "time_metrics/data_loading(s)": time_data_loading,
                    "time_metrics/data_loading(%)": time_data_loading_pct,
                    "memory/max_active(GiB)": device_mem_stats.max_active_gib,
                    "memory/max_active(%)": device_mem_stats.max_active_pct,
                    "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
                    "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
                    "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
                    "memory/num_ooms": device_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({device_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}tps: {round(tps):,} {color.reset}"
                    # f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
                )

                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                device_memory_monitor.reset_peak_stats()

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
