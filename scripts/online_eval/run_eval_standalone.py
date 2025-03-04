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
#from torch.distributed._tensor import Replicate
from torch.distributed.tensor import distribute_module, distribute_tensor, DTensor, Replicate, Shard
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
from torchtitan.parallelisms import ParallelDims
from torchtitan.models.llava_onevision.parallelize_llava import parallelize_llava
from torchtitan.utils import device_module, device_type

from torchtitan.datasets.alfred_dataset import ALFREDDataset, AlfredDataLoader

from transformers import AutoConfig, AutoProcessor
from torchtitan.models.llava_onevision import LlavaOnevisionForConditionalGeneration
from huggingface_hub import snapshot_download
import gc

from env_utils.env.thor_env import ThorEnv

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import sample


def set_nested_attr(obj, name, value):
    """since model.register_buffer() doesn't work on module names with '.',
       manually set a neste attribute for buffers"""
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


@torch.no_grad()
@record
def main(
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

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    logger.info(f"World Size: {world_size}, Local Rank: {local_rank} on {device}")

    dp_degree, dp_rank = 1, 0

    model_name = config.model.name

    # Tokenizer setup
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # build dataloader
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})
    dataset = ALFREDDataset(
        processor=processor,
        traj_data_dir='/torchtitan/data/alfred/train_small_traj',
        img_data_dir='/torchtitan/data/alfred/train_small_img',
        cp_degree=1,
        eval=True)
    # data_loader = AlfredDataLoader(dp_rank, dataset, batch_size=1, world_size=world_size)

    model_dtype = torch.bfloat16
    if 'llava' in model_name: # need to save buffers (position embeddings, layer norm statistics, etc.)
        model_cls = LlavaOnevisionForConditionalGeneration
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype)
        buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        del model

    # model
    init_device = "cuda"
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, device_map='auto')
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict, checkpoint_id=checkpoint_path)

    model.eval()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    device_memory_monitor.reset_peak_stats()
    
    ###################################################

    act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
    pad_tok_id = processor.tokenizer.pad_token_id
    cp_degree = 1

    env = ThorEnv()
    
    from eval_subgoals import EvalSubgoals
    eval_subgoals = EvalSubgoals()

    for j, batch in enumerate(dataset):
        logger.info(f"Test id {j}")
        traj = batch

        # batch['task_type'] = 'pick_and_place_with_movable_recep'
        # batch['pddl_params'] = {'mrecep_target': 'Pan', 'object_sliced': True, 'object_target': 'Apple', 'parent_target': 'CounterTop', 'toggle_target': ''}
        #task = batch['task'][0]
        
        # eval_subgoals.setup_scene(env, traj)
        subgoal_idxs = eval_subgoals.get_subgoal_idx(traj)
        
        for eval_idx in subgoal_idxs[1:]:
            # context: text and image list
            context, img_list, sim_success = eval_subgoals.simulate_with_expert(env, traj, processor, eval_idx)
            if not sim_success:
                break # if expert traj fails, even next subgoal cannot make it
            logger.info(f"[Prompt] {context}")
            batch = processor(images=img_list, text=context, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)
            logger.info(f"Input shape: {batch.input_ids.shape}")
            
            done = False

            while not done:
                #if rank == 0:
                # broadcast batch
                generated_tokens = generate(model, batch, device, cp_degree, act_tok_id, pad_tok_id)
                gc.collect()

                #if rank == 0:
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                logger.info(f"{color.blue}{output_text}\n{color.reset}")
                success, done, context, img_list = eval_subgoals.interact_with_env(env, output_text, eval_idx)
                # TODO broadcast success

                if (not success) or done:
                    break
                    
                batch = processor(images=img_list, text=context, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)

            # do smth for metrics
            eval_subgoals.metrics(traj, eval_idx, done)

    env.stop()

@torch.no_grad()
def generate(model, batch, device, cp_degree, act_tok_id, pad_tok_id, max_new_tokens=10):
    input_ids = batch.input_ids.to(device)
    pixel_values = batch.pixel_values.to(device)
    n_image =  torch.tensor([[pixel_values.shape[0]]], device=pixel_values.device)
    logger.info(f"pixel_values: {pixel_values.shape}, n_image: {n_image}")
    
    seq_len = input_ids.shape[1]
    pad_size = max(max_new_tokens, (cp_degree * 2) - (seq_len % (cp_degree * 2)))

    if pad_size > 0:
        pad_token_tensors = torch.tensor([pad_tok_id] * pad_size, device=device)[None, :]
        input_ids = torch.cat([pad_token_tensors, input_ids], dim=1) 
        logger.info(f"with padding ==> input_ids shape {seq_len} => {input_ids.shape}, {type(input_ids)}")

    with torch.no_grad():
        context_embeds = model.embed(
            input_ids=input_ids,
            pixel_values=pixel_values.unsqueeze(0),
            n_image=n_image,
        )

    logger.info(f"context_embeds: {type(context_embeds)}, {context_embeds.shape}, {context_embeds.dtype}")

    #context_embeds = context_embeds.to_local()
    # if isinstance(input_ids, DTensor):
    #     tensor_list = [torch.empty_like(context_embeds) for _ in range(world_size)]
    #     dist.all_gather(tensor_list, context_embeds)
    #     context_embeds = torch.cat(tensor_list, dim=2)
    #     del tensor_list

    inputs_embeds = context_embeds.clone()
    #inputs_embeds = context_embeds

    #del context_embeds, input_ids
    del input_ids

    #input_ids = input_ids.to_local()

    gc.collect()

    train_context = utils.get_train_context(False, False)

    new_tokens = []
    for i in range(max_new_tokens):
        seq_len = inputs_embeds.shape[1]
        max_seq = (seq_len // ((cp_degree * 2))) * (cp_degree * 2)
        inputs_embeds = inputs_embeds[:, -max_seq:, :]

        position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0)

        context_parallel_ctx = (
            utils.create_context_parallel_ctx(
                cp_mesh=world_mesh["cp"],
                cp_buffers=[inputs_embeds, position_ids],
                cp_seq_dims=[1, 1],
                cp_no_restore_buffers={inputs_embeds, position_ids},
                cp_rotate_method=config.experimental.context_parallel_rotate_method,
            )
            if cp_degree > 1
            else None
        )
        
        with train_context(context_parallel_ctx):
            logits = model(inputs_embeds=inputs_embeds,
            #position_ids=position_ids, # you will need later soon
            num_logits_to_keep=1)
            #output = model(input_ids=input_ids, num_logits_to_keep=1)

        new_token, _ = sample(logits, need_probs=True, temperature=1.0)
        #new_token, _ = sample(output.logits, need_probs=True, temperature=1.0, top_k=top_k)
        
        if new_token.item() == act_tok_id:
            break
        
        new_tokens.append(new_token.item())

        del logits

        with torch.no_grad():
            new_tokens_emb = model.embed(input_ids=new_token.unsqueeze(0))

        if isinstance(new_tokens_emb, DTensor):
            new_tokens_emb = new_tokens_emb.to_local()

        # tensor_list = [torch.empty(inputs_embeds.shape, device=device, dtype=inputs_embeds.dtype) for _ in range(world_size)]
        # dist.all_gather(tensor_list, inputs_embeds)
        #inputs_embeds = torch.cat(tensor_list + [new_tokens_emb] , dim=1)
        #del tensor_list
        inputs_embeds = torch.cat([inputs_embeds, new_tokens_emb] , dim=1)

        gc.collect()

    return new_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="distributed_checkpoints/",
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

    main(
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
