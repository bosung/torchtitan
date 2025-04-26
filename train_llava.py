# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta
from pathlib import Path
import subprocess

import torch
import torch.nn as nn

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
from huggingface_hub import snapshot_download, upload_folder, create_repo

AWS_S3_PATH = os.environ['AWS_S3_PATH']


def get_local_rank():
  return int(os.environ["LOCAL_RANK"])


def set_nested_attr(obj, name, value):
    """since model.register_buffer() doesn't work on module names with '.',
       manually set a neste attribute for buffers"""
    if not hasattr(obj, name):
        return
    
    parts = name.split('.')
    for part in parts[:-1]:
        if isinstance(obj, (nn.ModuleList, list)):
            obj = obj[int(part)]  # Convert string index to int for list access
        else:
            obj = getattr(obj, part)

    #setattr(obj, parts[-1], value)
    obj.register_buffer(parts[-1], value)
    # if obj is not None:
    #     setattr(obj, parts[-1], value)


def combine_model_parts_state(model_parts):
    combined_state = {}
    
    for model in model_parts:
        part_state = model.state_dict()
        
        # Store each parameter using its original name
        for key, value in part_state.items():
            if value is not None:
                combined_state[key] = value
            
    return combined_state


def save_checkpoint_s3(states, step, output_dir): # output_dir: outputs/checkpoints/step-xxxx
    
    # Push checkpoints from local_rank 0
    if get_local_rank() == 0:
        try:
            # Run aws s3 sync in background using nohup
            sync_command = f"nohup aws s3 sync {output_dir} {AWS_S3_PATH}/step-{step} > /tmp/s3_sync_{step}.log 2>&1 &"
            subprocess.Popen(
                sync_command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            logger.info(f"Started background S3 sync for checkpoint at step {step}")
        except Exception as e:
            logger.error(f"Error starting S3 sync: {e}")

    # Ensure upload is complete before proceeding
    dist.barrier()


def warmup_dynamic_rope_scaling(model, device, seq_len, rope_kwargs):
    try:
        layers = model.language_model.model.layers
        # device = model.device if hasattr(model, "device") else torch.device("cuda")
        # device = torch.device("cuda")
        config = model.language_model.config if hasattr(model.language_model, "config") else model.config

        if rope_kwargs['rope_type'] == "yarn":
            config.rope_scaling = rope_kwargs
            for i, layer in enumerate(layers):
                layer.self_attn.rotary_emb.freq_update(seq_len, rope_kwargs, device=device, config=config)
            model.language_model.model.rotary_emb.freq_update(seq_len, rope_kwargs, device=device, config=config)
        else:
            for i, layer in enumerate(layers):
                layer.self_attn.rotary_emb.freq_update(seq_len, rope_kwargs)
            model.language_model.model.rotary_emb.freq_update(seq_len, rope_kwargs)

        logger.info(f"Warmed up RoPE scaling on {len(layers)} layers with seq_len = {seq_len} rope_kwargs = {rope_kwargs}")
    except Exception as e:
        logger.info(f"RoPE warm-up skipped or partial: {e}")


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
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)
    model_name = job_config.model.name

    if job_config.training.dataset == "alfred":
        processor = build_hf_processor(model_name)
        tokenizer = processor.tokenizer
        traj_data_dir = os.environ['TRAJ_DATA_DIR'] 
        img_data_dir = os.environ['IMG_DATA_DIR']
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})
        
        # TODO incorporate with build_hf_data_loader
        if job_config.training.seq_len > 131072:
            from torchtitan.datasets.alfred_dataset_long_ctx import ALFREDDataset, AlfredDataLoader
        else:
            from torchtitan.datasets.alfred_dataset import ALFREDDataset, AlfredDataLoader
        dataset = ALFREDDataset(processor=processor,
            traj_data_dir=traj_data_dir,
            img_data_dir=img_data_dir,
            max_seq_len=job_config.training.seq_len, world_size=world_size,
            cp_degree=job_config.experimental.context_parallel_degree)
        data_loader = AlfredDataLoader(dp_rank, dataset, 
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
    text_config = model_config.text_config

    if job_config.training.rope_theta:
        model_config.text_config.rope_theta = job_config.training.rope_theta

    # sliding window attention
    if job_config.training.attn_impl == "flash_attention_2":
        model_config.attn_impl = "flash_attention_2"
        model_config.text_config._attn_implementation = "flash_attention_2"
        model_config.text_config.use_sliding_window = True
        model_config.text_config.max_window_layers = 0

    if 'llava' in model_name: # need to save buffers (position embeddings, layer norm statistics, etc.)
        model = model_cls.from_pretrained(model_name)
        #logger.info(f"{job_config.training.rope_type}")
        #buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        #logger.info(f"{buffers_dict['language_model.model.layers.0.self_attn.rotary_emb.inv_freq']}")
        if job_config.training.rope_type:
            if job_config.training.rope_type == "nope":
                rope_type = "default"
                model_config.text_config.nope = True

            # refer to: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py
            partial_rotary_factor = text_config.partial_rotary_factor if hasattr(text_config, "partial_rotary_factor") else 1.0
            head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
            dim = int(head_dim * partial_rotary_factor)
            rope_kwargs = {
                "rope_type": job_config.training.rope_type,  # or 'linear', 'longrope', etc.
                "factor": job_config.training.rope_factor,  # 4x the original context length
                #"original_max_position_embeddings": 32768,  # typical default; adjust based on model
                #"original_max_position_embeddings": model_config.text_config.max_position_embeddings
                "dim": dim,
                "base": text_config.rope_theta,
                "max_position_embeddings": model_config.text_config.max_position_embeddings # original max_len
            }
            if job_config.training.rope_type == "longrope":
                rope_kwargs['long_factor'] = job_config.training.rope_factor
                rope_kwargs['short_factor'] = 1
                rope_kwargs['factor'] = 1 # https://github.com/huggingface/transformers/blob/b54c2f46891149210dbbe118fca55b1357a47003/src/transformers/modeling_rope_utils.py#L322

            if job_config.training.rope_type != "nope":
                warmup_dynamic_rope_scaling(model, device, job_config.training.seq_len, rope_kwargs)
            assert model.language_model.model.layers[0].self_attn.rotary_emb.rope_type != 'dynamic'
        buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        #logger.info(f"{buffers_dict['language_model.model.layers.0.self_attn.rotary_emb.inv_freq']}")
        del model
    
    with torch.device("meta"):
        if 'llava' in model_name:
            # using different attn_implementation might matter depending on TP, PP, and CP, etc.
            #model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, attn_implementation="eager")
            model = model_cls.from_pretrained(model_name, config=model_config, attn_implementation=job_config.training.attn_impl)
            assert len(processor.tokenizer) < model.language_model.lm_head.weight.shape[0]
            assert model.language_model.lm_head.weight.shape[0] % 8 == 0
        else:
            model = model_cls.from_model_args(model_config)

    # log model size
    model_param_count = utils.get_num_params(model)
    
    logger.info(
        f"Building {train_spec.name} {job_config.model.flavor} with {model_config}"
    )
    
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
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # Since TP shards input_embeds, position_ids is not algined based on its position
        # set postion_ids as buffer so that not to pass in forward (only works for training)
        position_ids = torch.arange(0, job_config.training.seq_len).unsqueeze(0)

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            # with torch.no_grad():
            #     m.init_weights(buffer_device=buffer_device)
            state_dict = {"model": m.state_dict()}
            dcp.load(state_dict, checkpoint_id=checkpoint_path, planner=dcp.DefaultLoadPlanner(allow_partial_load=True))

            if hasattr(m, "language_model"):
                setattr(m.language_model.model, 'position_ids', position_ids.to(device_type))

            m.train()
    else:
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            if job_config.checkpoint.create_seed_checkpoint:
                checkpoint_path = 'distributed_checkpoint/'
                state_dict = {"model": model.state_dict()}
                dcp.load(state_dict, checkpoint_id=checkpoint_path)
            # Restore buffers
            # for name, buffer in buffers_dict.items():
            #     set_nested_attr(model, name, buffer.to(device_type))
        model.train()
        model_parts = [model]

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    
    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)

    # TODO: where is the best place to put buffer loading
    # Restore buffers after loading checkpoint
    for m in model_parts:
        for name, buffer in buffers_dict.items():
            set_nested_attr(m, name, buffer.to(device_type))
    
    if job_config.training.rope_type and job_config.training.rope_type != "nope":
        logger.info(f"RoPE rescaling is in use: rope_kwargs: {rope_kwargs}")
        logger.info(f"Check RoPE: {model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq}")

    metric_logger = build_metric_logger(job_config, parallel_dims)

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0:
        for idx, step in enumerate(train_state.log_steps):
            metrics = {
                "loss_metrics/global_avg_loss": train_state.global_avg_losses[idx],
                "loss_metrics/global_max_loss": train_state.global_max_losses[idx],
            }
            metric_logger.log(metrics, step=step)

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    device_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)
            
            # get batch
            data_load_start = time.perf_counter()
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(data_loader)
                batch = next(data_iterator)
            
            input_ids, labels = batch['input_ids'], batch['labels']
            ntokens_since_last_log += labels.numel()
            data_loading_times.append(time.perf_counter() - data_load_start)

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            optimizers.zero_grad()

            pixel_values=batch['pixel_values'].to(device_type)
            n_image=batch['n_image'].to(device_type)

            # TODO add to config
            enable_embed_batch = True if (job_config.training.seq_len >= 16384 and job_config.training.batch_size > 1) else False
            with torch.no_grad():
                if parallel_dims.pp_enabled:
                    if has_first_stage:
                        inputs_embeds = model_parts[0].embed(
                            input_ids=input_ids,
                            pixel_values=pixel_values,
                            n_image=n_image,
                            enable_embed_batch=enable_embed_batch)
                else:
                    inputs_embeds = model.embed(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        n_image=n_image,
                        enable_embed_batch=enable_embed_batch)

            position_ids = torch.arange(
                0, input_ids.shape[1], device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], position_ids.shape[1])
            del pixel_values, n_image
            
            # since nn.Embedding is not sharded bc of mask_scatter for pixel_values, manually shard here.
            # if not (parallel_dims.cp_enabled and (parallel_dims.pp_enabled and has_first_stage)):
            # TODO let's do not use pass here
            if parallel_dims.tp_enabled and (not parallel_dims.cp_enabled):
                if parallel_dims.pp_enabled and not has_first_stage:
                    pass
                else:
                    # redistributed for TP
                    inputs_embeds = distribute_tensor(inputs_embeds, world_mesh['tp'], placements=[Shard(1)])
                    inputs_embeds = inputs_embeds.to_local()

            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, inputs_embeds, labels, position_ids],
                    cp_seq_dims=[1, 1, 1, 1],
                    cp_no_restore_buffers={input_ids, inputs_embeds, labels, position_ids},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        logger.info(f"step: {train_state.step:2} inputs_embeds: {inputs_embeds.shape} {type(inputs_embeds)}")
                        #pp_schedule.step(target=targets, losses=losses, **{"inputs_embeds":inputs_embeds})
                        #pp_schedule.step(inputs_embeds, target=targets, losses=losses, **{"position_ids":position_ids})
                        pp_schedule.step(inputs_embeds, target=targets, losses=losses)
                    else:
                        #pp_schedule.step(target=targets, losses=losses, **{"position_ids":position_ids})
                        pp_schedule.step(target=targets, losses=losses)
                        logger.info(f"step: {train_state.step:2} -- stage 2 DONE ! ")
                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
                if has_last_stage:
                    logger.info(f"step: {train_state.step:2} {color.yellow}{loss}{color.reset}")
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    logits = model.language_model(inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    use_cache=False)

                    # hack for CP
                    if (labels + torch.tensor([100], device=labels.device)).sum() == 0:
                        labels[:, -2] = input_ids[:, -1]

                    loss = loss_fn(logits, labels)
                    #logger.info(f"step: {train_state.step:2} {color.yellow}{loss}{color.reset}")
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del logits
                    loss.backward()

            # clip gradients
            utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
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
                train_state.step, force=(train_state.step == job_config.checkpoint.interval)
            )

            if (train_state.step % job_config.checkpoint.interval) == 0:
                output_dir = checkpoint._create_checkpoint_id(step=train_state.step)
                # dcp.save(model.state_dict(), checkpoint_id=output_dir)
                save_checkpoint_s3(
                    states=checkpoint.states,
                    step=train_state.step,
                    output_dir=output_dir
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
