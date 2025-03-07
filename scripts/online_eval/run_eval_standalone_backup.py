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
import re
import boto3
from typing import Optional, Tuple

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import DeviceMesh
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
from torchtitan.metrics import build_device_memory_monitor, build_metric_logger
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

AWS_S3_PATH = os.environ['AWS_S3_PATH']

def set_nested_attr(obj, name, value):
    """since model.register_buffer() doesn't work on module names with '.',
       manually set a neste attribute for buffers"""
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

def broadcast_tensor(tensor=None, dtype=None, device=None, src_rank=0):
    """
    Broadcast a tensor from source rank to all ranks.
    If tensor is None on non-source ranks, it will be created with the broadcasted shape and provided dtype.
    
    Args:
        tensor: The tensor to broadcast (required on src_rank)
        dtype: The tensor's dtype (if None, will use float32 or the src tensor's dtype)
        device: Device to place tensor on (defaults to cuda if available, otherwise cpu)
        src_rank: The source rank for broadcasting
    """
    rank = dist.get_rank()
    
    # Set default device if none provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # On source rank, use tensor's dtype if not specified
    if rank == src_rank and tensor is not None and dtype is None:
        dtype = tensor.dtype
    
    # Default dtype if still None
    if dtype is None:
        dtype = torch.float32
    
    # Step 1: Broadcast tensor shape
    if rank == src_rank and tensor is not None:
        # Create a tensor with shape information
        shape_tensor = torch.tensor(list(tensor.shape), dtype=torch.long, device=device)
    else:
        # For non-source ranks, create empty tensor to receive the shape
        shape_tensor = torch.zeros(8, dtype=torch.long, device=device)  # Support up to 8D tensors
    logger.info(f"Rank: {rank} -- 1")
    # First broadcast the number of dimensions
    ndims = torch.tensor([len(tensor.shape) if rank == src_rank else 0], dtype=torch.long, device=device)
    dist.broadcast(ndims, src=src_rank)
    ndims = ndims.item()
    logger.info(f"Rank: {rank} -- 2")
    # Trim shape tensor to actual ndims
    shape_tensor = shape_tensor[:ndims]

    # Broadcast the shape
    dist.broadcast(shape_tensor, src=src_rank)
    logger.info(f"Rank: {rank} -- 3")
    # Process shape
    shape = [dim for dim in shape_tensor.tolist() if dim > 0]
    
    # Create or use tensor
    if rank != src_rank or tensor is None:
        tensor = torch.zeros(shape, dtype=dtype, device=device)
    # else:
    #     # Ensure source tensor is on the correct device and dtype
    #     tensor = tensor.to(device=device, dtype=dtype)
    
    # Broadcast data
    dist.broadcast(tensor, src=src_rank)
    logger.info(f"Rank: {rank} -- 4")
    return tensor

def check_existing_evals(s3_path):
    # Parse bucket and prefix from s3_path
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with 's3://'")
    
    parts = s3_path[5:].split('/', 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ''
    
    # Ensure prefix ends with a slash if it's not empty
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # List objects in the bucket with the given prefix
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return 0
    
    # Check if directory exists and has files
    if 'Contents' not in response:
        return 0
    
    # Pattern to match files like test_id-0.json, test_id-1.json, etc.
    pattern = re.compile(r'test_id-(\d+)\.json$')
    
    # Find the highest test_id
    highest_id = 0
    for item in response['Contents']:
        key = item['Key']
        filename = key.split('/')[-1]
        match = pattern.search(filename)
        if match:
            test_id = int(match.group(1))
            highest_id = max(highest_id, test_id)
    
    return highest_id


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
    parallel_dims = ParallelDims(
        dp_shard=1,
        dp_replicate=1,
        #cp=config.experimental.context_parallel_degree,
        cp=world_size,
        tp=1,
        pp=1,
        world_size=world_size,
        enable_loss_parallel=False,
    )
    cp_degree = world_size
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"{device_type}:{local_rank}")
    device_module.set_device(device)
    utils.init_distributed(config)
    device_memory_monitor = build_device_memory_monitor()
    world_mesh = parallel_dims.build_mesh(device_type=device_type)

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
        traj_data_dir=os.environ['TRAJ_DATA_DIR'],
        cp_degree=1,
        eval=True)

    eval_done_idx = check_existing_evals(s3_path=AWS_S3_PATH)
    
    # data_loader = AlfredDataLoader(dp_rank, dataset, batch_size=1, world_size=world_size)

    model_dtype = torch.bfloat16
    if 'llava' in model_name: # need to save buffers (position embeddings, layer norm statistics, etc.)
        model_cls = LlavaOnevisionForConditionalGeneration
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype)
        buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
        del model

    # model
    #init_device = "cuda"
    with torch.device("meta"):
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype)
        
    init_device = device_type
    model.to_empty(device=init_device)
    state_dict = {"model": model.state_dict()}
    #dcp.load(state_dict, checkpoint_id=checkpoint_path)
    from torch.distributed.checkpoint import DefaultLoadPlanner
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    for name, buffer in buffers_dict.items():
        set_nested_attr(model, name, buffer.to(device_type))

    model.eval()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    device_memory_monitor.reset_peak_stats()

    metric_logger = build_metric_logger(config, parallel_dims)
    
    ###################################################

    act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
    pad_tok_id = processor.tokenizer.pad_token_id

    if dist.get_rank() == 0:
        env = ThorEnv()
    else:
        env = None
    
    from eval_subgoals import EvalSubgoals
    eval_subgoals = EvalSubgoals()

    sim_fail_cnt = 0
    for j, batch in enumerate(dataset):
        if j <= eval_done_idx:
            continue
        if j == 10:
            break
        
        traj = batch
            
        subgoal_idxs = eval_subgoals.get_subgoal_idx(traj)
        
        if dist.get_rank() == 0:
            _, _, sim_success = eval_subgoals.simulate_with_expert(env, traj, processor, subgoal_idxs[-1] + 1)
        else:
            sim_success = False

        sim_success_tensor = torch.tensor([1 if sim_success else 0], dtype=torch.int32).cuda()
        dist.broadcast(sim_success_tensor, src=0)
        sim_success = bool(sim_success_tensor.item())
        logger.info(f"Test id {j} -- sim_success: {sim_success}")
        
        continue
        if not sim_success:
            sim_fail_cnt += 1
            logger.info(f"Skip this example -- expert traj event doens't work (sim_fail_cnt: {sim_fail_cnt})")
            continue

        for eval_idx in subgoal_idxs:
            if dist.get_rank() == 0:
                input_ids, pixel_values, sim_success = eval_subgoals.simulate_with_expert(env, traj, processor, eval_idx)
            else:
                input_ids, pixel_values, sim_success = None, None, False

            sim_success_tensor = torch.tensor([1 if sim_success else 0], dtype=torch.int32, device=device)
            dist.broadcast(sim_success_tensor, src=0)
            sim_success = bool(sim_success_tensor.item())
            logger.info(f"Test id {j} -- expert sim_success 22: {sim_success}")
            dist.barrier()
            if not sim_success:
                break # if expert traj fails, even next subgoal cannot make it

            input_ids = broadcast_tensor(input_ids, torch.int32, device, src_rank=0)
            pixel_values = broadcast_tensor(pixel_values, torch.bfloat16, device, src_rank=0)
            logger.info(f"Rank {dist.get_rank()} - input_ids: {input_ids.shape}")
            done = False

            while not done:
                generated_tokens = generate(model, input_ids, pixel_values, config, device, cp_degree, world_mesh, act_tok_id, pad_tok_id)
                gc.collect()

                # success, done = False, False
                if dist.get_rank() == 0:
                    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.info(f"{color.blue}{output_text}\n{color.reset}")
                    success, done = eval_subgoals.interact_with_env(env, output_text, eval_idx)
                else:
                    success, done = None, None

                success = bool(broadcast_tensor(success, torch.int32, device, src_rank=0).item())
                done = bool(broadcast_tensor(done, torch.int32, device, src_rank=0).item())
                
                if (not success) or done:
                    break
                
                if dist.get_rank() == 0:
                    input_ids, pixel_values = eval_subgoals.process_input(processor)
                    input_ids, pixel_values = processor(images=img_list, text=context, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)
                
                input_ids = broadcast_tensor(input_ids, torch.int32, device, src_rank=0)
                pixel_values = broadcast_tensor(pixel_values, torch.bfloat16, device, src_rank=0)
                break
            break
            metrics = eval_subgoals.update_metrics(traj, eval_idx, done, test_id=j)
        continue
        metric_logger.log(metrics, step=j)
        metric_logger.log(eval_subgoals.results, step=j)
        test_id = j
        eval_subgoals.sync_metrics_s3(AWS_S3_PATH, test_id)

    if dist.get_rank() == 0: 
        env.stop()
    

@torch.no_grad()
def generate(model, input_ids, pixel_values, config, device, cp_degree, world_mesh, act_tok_id, pad_tok_id, max_new_tokens=10):
    #input_ids = batch.input_ids.to(device)
    #pixel_values = batch.pixel_values.to(device)
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device)
    n_image =  torch.tensor([[pixel_values.shape[0]]], device=pixel_values.device)
    #logger.info(f"pixel_values: {pixel_values.shape}, n_image: {n_image}")
    
    seq_len = input_ids.shape[1]
    pad_size = max(max_new_tokens, (cp_degree * 2) - (seq_len % (cp_degree * 2)))

    if pad_size > 0:
        pad_token_tensors = torch.tensor([pad_tok_id] * pad_size, device=device)[None, :]
        input_ids = torch.cat([pad_token_tensors, input_ids], dim=1) 
        #logger.info(f"with padding ==> input_ids shape {seq_len} => {input_ids.shape}, {type(input_ids)}")

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

        #logger.info(f"inputs_embeds: {inputs_embeds.shape}")

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
            logits = model.language_model(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids, # you will need later soon
                use_cache=False,
                num_logits_to_keep=1)

        new_token, _ = sample(logits, need_probs=True, temperature=1.0)
        #new_token, _ = sample(output.logits, need_probs=True, temperature=1.0, top_k=top_k)
        del logits
        #logger.info(f"---------- {i}, new_token: {new_token} (act_tok_id = {act_tok_id})")
        if new_token.item() == act_tok_id:
            break
        
        new_tokens.append(new_token.item())

        with torch.no_grad():
            new_tokens_emb = model.embed(input_ids=new_token.unsqueeze(0))

        if isinstance(new_tokens_emb, DTensor):
            new_tokens_emb = new_tokens_emb.to_local()

        tensor_list = [torch.empty(inputs_embeds.shape, device=device, dtype=inputs_embeds.dtype) for _ in range(cp_degree)]
        dist.all_gather(tensor_list, inputs_embeds)
        inputs_embeds = torch.cat(tensor_list + [new_tokens_emb] , dim=1)
        del tensor_list
        logger.info(f"---------- inputs_embeds: {inputs_embeds.shape}")

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
