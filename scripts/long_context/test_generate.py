# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
import time
from pathlib import Path
from concurrent import futures
from functools import partial

from typing import Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
import torch.multiprocessing as mp

from torchtitan import utils

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_tokenizer
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_device_memory_monitor
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.parallelisms import ParallelDims
from torchtitan.parallelisms.parallelize_llama import parallelize_llama, apply_tp
from torchtitan.parallelisms.pipeline_llama import pipeline_llama_manual_split
from torchtitan.utils import device_module, device_type

from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.my_datasets import MyDataset

#from torchchat.generate import run_generator
#from torchchat.cli.builder import BuilderArgs

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import sample


def get_example_ins_outs(model, model_config, device, 
    pp_rank, first_pp_rank, last_pp_rank,
    batch_size: int , seqlen: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    This function generates example inputs and outputs for the prefill and decode stages.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the example inputs and outputs.
    """
    model_dtype = torch.bfloat16
    mb_ids = torch.randint(
        0, model_config.vocab_size, (batch_size, seqlen), device=device
    )
    activation = torch.rand(
        batch_size, seqlen, model_config.dim, device=device, dtype=model_dtype
    )
    logits = torch.rand(
        batch_size, seqlen, model_config.vocab_size, device=device, dtype=model_dtype
    )
    example_inputs = (mb_ids if pp_rank == first_pp_rank else activation,)
    example_outputs = (logits if pp_rank == last_pp_rank else activation,)
    return example_inputs, example_outputs
    

def run_in_dist_env(world_size: int, rank: int, target: callable):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RDZV_BACKEND"] = "c10d"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCALRANK"] = str(rank)

    return target()


def apply_tp_minus_sp(model: nn.Module, tp_mesh: DeviceMesh):
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(input_layouts=Replicate()),
            "output": ColwiseParallel(output_layouts=Replicate()),
        },
    )

    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(),
            "feed_forward.w3": ColwiseParallel(),
        }

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

@record
def test_generate(
    config_path: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    ctx_len: int = 8192
):
    init_logger()
    color = utils.Color

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()

    # if local_rank == 0:
        ######################################################################################
        # Construct the Needle-in-a-HayStack Prompt
        # needle = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
        # depth = 0.5
        # context = load_context(fpath="scripts/long_context/eval/needle/PaulGrahamEssays/*.txt", ctx_len=ctx_len)
        # context = insert_needle(context, needle, depth=depth)
        # needle_idx = context.find("The best thing to do in San Francisco is")
        # logger.info("Context has %d chars, needle inserted at %d char location:\n" % (len(context), needle_idx))
        # logger.info(context[needle_idx - 150: needle_idx + 150]) # look at how the needle is inserted 
        # prompt ="\n<|im_start|> This is a very long story book: <book> %s </book>.\n" % context
        # question = "What is the best thing to do in San Francisco?"
        # prompt += "Based on the content of the book, Question: %s\nAnswer:" % question
        ######################################################################################

    # if len(args.prompt) == 0:
    #     logger.warning(
    #         "The input prompt is empty, model will respond from a empty sequence."
    #     )

    model_name = config.model.name

    # Tokenizer setup
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[model_name], config.model.tokenizer_path
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    world_mesh = None
    # Init distributed env
    if world_size > 1:
        utils.init_distributed(config)
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=1,
            tp=2,
            pp=4,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type=device_type)

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    utils.set_determinism(world_mesh, device, seed, deterministic)

    # build dataloader
    '''
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )
    '''

    my_ds = MyDataset(
        #data=[prompt],
        tokenizer=tokenizer,
        seq_len=ctx_len,
        world_size=world_size,
        rank=dp_rank,
    )
    data_loader = DPAwareDataLoader(dp_rank, my_ds, batch_size=batch_size)

    # model
    model_config = models_config[model_name][config.model.flavor]
    model_config.norm_type = config.model.norm_type
    model_config.max_seq_len = config.training.seq_len
    model_config.vocab_size = tokenizer.n_words

    model_cls = model_name_to_cls[model_name]
    init_device = "meta" if world_size > 1 else device
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_config)

    # apply_tp (with Sequence Parallel) on unevenly sharded
    # sequences would require https://github.com/pytorch/torchtitan/pull/686
    # apply_tp_minus_sp(model, world_mesh["tp"])
    # parallelize_llama(model, world_mesh, parallel_dims, config)

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]
        # apply PT-D Pipeline Parallel
        _, model_parts = pipeline_llama_manual_split(model, pp_mesh, parallel_dims, config, device, model_config)
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            parallelize_llama(m, world_mesh, parallel_dims, config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                buffer_device = None
                m.init_weights(buffer_device=buffer_device)
            m.eval()
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        parallelize_llama(model, world_mesh, parallel_dims, config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            buffer_device = None
            model.init_weights(buffer_device=buffer_device)
        model.eval()

        model_parts = [model]

    ################### Loading checkpoints ##########################

    #state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    # begin = time.monotonic()
    # logger.info(f"Loading chkpt at: {checkpoint_path}")
    # dcp.load(state_dict, checkpoint_id=checkpoint_path)
    # logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")
    
    logger.info(f"Resizing model.freqs_cis to fit ctx_len ... ")
    model = model_parts[0]
    prev_freqs_cis_dim = model.freqs_cis.shape
    model.recompute_freqs_cis(ctx_len)
    logger.info(f"Resizing DONE: {prev_freqs_cis_dim} -> {model.freqs_cis.shape} ")

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # setting context 
    train_context = utils.get_train_context(False, False)

    # Tokenize prompt and repeat batch_size times
    # input_ids = (
    #     (
    #         torch.tensor(
    #             tokenizer.encode(prompt, bos=True, eos=False), dtype=torch.long
    #         )
    #         .view(1, -1)
    #         .repeat(batch_size, 1)
    #     )
    # ).to(device_type)

    device_memory_monitor.reset_peak_stats()
    
    rank = dist.get_rank()

    if parallel_dims.pp_enabled:
        pp_rank = pp_mesh.get_local_rank()
        pp_group = pp_mesh.get_group()

        pp_degree = pp_mesh.size()

        # Convenience variables
        first_pp_rank = 0
        last_pp_rank = pp_degree - 1

        first_pp_rank_global_id = dist.get_global_rank(pp_group, first_pp_rank)
        last_pp_rank_global_id = dist.get_global_rank(pp_group, last_pp_rank)

        from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
        example_inputs, example_outputs = get_example_ins_outs(
            model, model_config, device, pp_rank, first_pp_rank, last_pp_rank, 1, 1
        )
        decode_stage = PipelineStage(
            model,
            pp_rank,
            pp_degree,
            device,
            input_args=example_inputs,
            output_args=example_outputs,
            group=pp_group,
        )
        # create schedule
        decoder = ScheduleGPipe(decode_stage, 1)

    ###################################################
    # Run generation
    
    t0 = time.monotonic()

    with torch.no_grad():
        for batch in data_loader:
            # ==== Prepare input ====
            with train_context():
                max_new_tokens = 3
                for i in range(max_new_tokens):

                    new_token = input_ids.view(1, -1)
                    

                    if parallel_dims.pp_enabled:
                        input_pos = torch.tensor([1, 1], device=device)
                        lane = 0
                        kwargs = {"input_pos": input_pos, "cache_lane": lane}

                        # Pipeline Parallel forward / backward inside step() call
                        is_last_stage = pp_mesh.get_local_rank() == pp_mesh.size() - 1
                
                        # Run data through pipeline
                        if pp_rank == first_pp_rank:
                            logits = decoder.step(new_token, **kwargs)
                        elif pp_rank == last_pp_rank:
                            logits = decoder.step(**kwargs)
                        else:  # middle pp ranks
                            decoder.step(**kwargs)

                        # Decode the output
                        if pp_rank == last_pp_rank:
                            new_token, _ = sample(logits, need_probs=need_probs, **sampling_kwargs)
                            if pp_rank != first_pp_rank:
                                dist.send(
                                    new_token,
                                    dst=first_pp_rank_global_id,
                                    group=pp_group,
                                )
                            logger.info(f"Step {i}: generated_tokens shape = {new_token.shape}")
                        else:
                            new_token = torch.zeros(1, 1, device=device, dtype=torch.int64)
                            if pp_rank == first_pp_rank:
                                dist.recv(
                                    new_token,
                                    src=last_pp_rank_global_id,
                                    group=pp_group,
                                )
                                #TODO: Why do we get 2d tensor here?
                                new_token=new_token[0]
                        
                        logger.info(f"Step {i}: pp_rank: {pp_rank}")
                        
                    else:
                        # Step 1: Handle the input_ids on rank 0 and prepare size info
                        if rank == 0:
                            input_ids = batch.to(device)
                            generated_tokens = input_ids.clone()
                            size_tensor = torch.tensor(input_ids.shape, device=device, dtype=torch.long)
                        else:
                            input_ids = None
                            generated_tokens = None
                            size_tensor = torch.empty(2, device=device, dtype=torch.long)  # Assuming max rank 2D tensors

                        # Step 2: Broadcast the size of input_ids from rank 0
                        dist.broadcast(size_tensor, src=0)

                        # Step 3: Dynamically resize input_ids and generated_tokens tensors on other ranks
                        if rank != 0:
                            input_ids = torch.empty(tuple(size_tensor.tolist()), device=device, dtype=torch.int64)
                            generated_tokens = torch.empty_like(input_ids, device=device)
                        
                        # Step 4: Broadcast the data from rank 0
                        dist.broadcast(generated_tokens, src=0)
                        # Proceed with training or further computation
                        logger.info(f"Step {i}: generated_tokens shape = {generated_tokens.shape}")
                        logits = model(input_ids)  # Generate logits
                        logger.info(f"Step {i}: logits shape = {logits.shape}")

                #gathered_tensors = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                #dist.all_gather(gathered_tensors, logits)
                #logger.info(f"Rank {dist.get_rank()} gathered: {len(gathered_tensors)}, gathered[0]: {gathered_tensors[0].shape}")
                #loss = loss_fn(pred, labels)
                # pred.shape=(bs, seq_len, vocab_size)
                # need to free to before bwd to avoid peaking memory
                #del pred
                    
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    #B, T = responses.size()  # B: batch_size, T: total seq length
    #input_n_tokens = input_ids.size(1)
    #generated_n_tokens = T - input_n_tokens  # == max_new_tokens

    if rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")
    '''
        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(responses):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            output_text = tokenizer.decode(out_tok)

            _data = {
                "response_idx": i,
                "input_text": input_text,
                "output_text": output_text,
            }
            output_data["responses"].append(_data)

            #logger.info(f"{r}\n{input_text}{b}{output_text}\n{color.reset}")
            logger.info(f"{b}{output_text}\n{color.reset}")

        device_mem_stats = device_memory_monitor.get_peak_stats()
        output_data["metadata"] = {
            "generated_n_tokens": generated_n_tokens,
            "input_n_tokens": input_n_tokens,
            "generation_time_sec": elapsed_sec,
            "tokens_per_sec": (B * T) / elapsed_sec,
            "batch_size": B,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "memory/max_active(GiB)": device_mem_stats.max_active_gib,
            "memory/max_active(%)": device_mem_stats.max_active_pct,
            "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
            "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
            "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
            "memory/num_ooms": device_mem_stats.num_ooms,
            "world_size": world_size,
            "torch_version": torch.__version__,
        }

        if args.out:
            print(json.dumps(output_data, indent=4))
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint path to load (required)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Default is 1.0",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32,
        help="Max number of tokens to generate. Default is 32",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Number of samples to run in batch"
    )
    parser.add_argument(
        "--top_k", type=int, help="Prune to select from top_k probabilities. Optional"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic algorithms wherever possible, may be slower",
    )

    parser.add_argument("--prompt", type=str, default="", help="Input prompt")

    parser.add_argument(
        "--out",
        action="store_true",
        default=False,
        help="If specified, prints the report to stdout. Defaults to no output.",
    )

    parser.add_argument(
        "--ctx_len",
        type=int,
        default=8192,
    )

    args = parser.parse_args()

    test_generate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
        ctx_len=args.ctx_len
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
