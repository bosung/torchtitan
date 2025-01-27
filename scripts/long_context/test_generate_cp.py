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

@torch.no_grad()
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

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    world_mesh = None
    # Init distributed env
    if world_size > 1:     
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=-1,
            cp=world_size,
            tp=1,
            pp=1,
            world_size=world_size,
            enable_loss_parallel=False,
        )
        utils.init_distributed(config)
        # Build world mesh for parallelism
        world_mesh = parallel_dims.build_mesh(device_type=device_type)

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    utils.set_determinism(world_mesh, device, seed, deterministic)
    model_name = config.model.name

    # Tokenizer setup
    tokenizer = build_tokenizer(
        model_name_to_tokenizer[model_name], config.model.tokenizer_path
    )
    # build dataloader
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
    #init_device = "meta" if world_size > 1 else device
    init_device = device_type
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_model_args(model_config)

    # apply_tp (with Sequence Parallel) on unevenly sharded
    # sequences would require https://github.com/pytorch/torchtitan/pull/686
    # apply_tp_minus_sp(model, world_mesh["tp"])
    # parallelize_llama(model, world_mesh, parallel_dims, config)

    # apply parallelisms and initialization
    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    parallelize_llama(model, world_mesh, parallel_dims, config)
    model.to_empty(device=init_device)
    with torch.no_grad():
        buffer_device = None
        model.init_weights(buffer_device=buffer_device)
    model.eval()

    model_parts = [model]

    ################### Loading checkpoints ##########################

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    # begin = time.monotonic()
    # logger.info(f"Loading chkpt at: {checkpoint_path}")
    # dcp.load(state_dict, checkpoint_id=checkpoint_path)
    # logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")
    
    logger.info(f"Resizing model.freqs_cis to fit ctx_len ... ")
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

    device_memory_monitor.reset_peak_stats()
    
    rank = dist.get_rank()

    ###################################################
    # Run generation
    
    t0 = time.monotonic()

    for batch in data_loader:
        input_ids = batch.to(device)

        generated_tokens = input_ids.clone()
        new_tokens = []

        max_new_tokens = 3
        for i in range(max_new_tokens):
            logger.info(f"{generated_tokens.shape}")
            seq_len = generated_tokens.shape[-1]
            logger.info(f"{seq_len}")
            max_len = (seq_len // (world_size * 2)) * (world_size * 2)
            logger.info(f"{generated_tokens.shape}, {max_len}")
            generated_tokens = generated_tokens[:, :max_len]
            logger.info(f"{generated_tokens.shape}")

            optional_context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[generated_tokens] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={generated_tokens},
                    cp_rotate_method=config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            with train_context(optional_context_parallel_ctx):
                logger.info(f"Step {i}: generated_tokens shape = {generated_tokens.shape}")
                logits = model(generated_tokens)  # Generate logits
                logger.info(f"Step {i}: logits shape = {logits.shape}")

            new_token, _ = sample(logits, need_probs=True, temperature=temperature, top_k=top_k)
            logger.info(f"Step {i}: new_token = {new_token}")
            new_tokens.append(new_token.item())
            generated_tokens = torch.cat([input_ids, torch.tensor(new_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)], dim=1)
                    
    t1 = time.monotonic()
    elapsed_sec = t1 - t0

    # Post process
    B, T = generated_tokens.size()  # B: batch_size, T: total seq length
    input_n_tokens = input_ids.size(1)
    #generated_n_tokens = T - input_n_tokens  # == max_new_tokens
    generated_n_tokens = 3

    if rank == 0:
        logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")
    
        r, b = color.red, color.blue

        output_data = {
            "metadata": {},
            "responses": [],
        }

        for i, tokens in enumerate(generated_tokens):
            inp_tok = tokens[:input_n_tokens].tolist()
            out_tok = tokens[input_n_tokens:].tolist()

            input_text = tokenizer.decode(inp_tok)
            # output_text = tokenizer.decode(out_tok)
            output_text = tokenizer.decode(new_tokens)

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
