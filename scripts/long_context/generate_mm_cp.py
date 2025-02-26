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

from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.datasets.alfred_dataset import ALFREDDataset
from torchtitan.datasets.toy_datasets import MyDataset
#from torchtitan.datasets.long_context_datasets import MyDataset

from transformers import AutoConfig, AutoProcessor
from torchtitan.models.llava_onevision import LlavaOnevisionForConditionalGeneration
from huggingface_hub import snapshot_download
import gc

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import sample


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
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # build dataloader
    #if config.training.dataset == "alfred":
    if False:
        # need to download img files
        img_data_dir = snapshot_download(repo_id="bosungkim/alfred-small-img", repo_type="dataset", allow_patterns="*.tar", local_dir='data/alfred-small-img')
        processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})
        dataset = ALFREDDataset(processor=processor, img_data_dir=img_data_dir)
    else:
        dataset = MyDataset(processor=processor)
        #dataset = MyDataset(tokenizer=processor.tokenizer)

    data_loader = DPAwareDataLoader(dp_rank, dataset, batch_size=batch_size, world_size=world_size)

    # model
    # model_config = models_config[model_name][config.model.flavor]
    # model_config.norm_type = config.model.norm_type
    # model_config.max_seq_len = config.training.seq_len
    # model_config.vocab_size = tokenizer.n_words

    #model_cls = model_name_to_cls[model_name]
    #init_device = "meta" if world_size > 1 else device
    init_device = device_type
    with torch.device(init_device):
        logger.info(f"Init model on init_device: {init_device}")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)

    # if config.training.dataset == "alfred":
    #     model.resize_token_embeddings(len(processor.tokenizer))

    # apply parallelisms and initialization
    parallelize_llava(model, world_mesh, parallel_dims, config)
    #model.to_empty(device=init_device)
    # with torch.no_grad():
    #     buffer_device = None
    #     model.init_weights(buffer_device=buffer_device)

    #model_parts = [model]

    ################### Loading checkpoints ##########################

    state_dict = {"model": model.state_dict()}

    # Checkpoint Loading
    from torch.distributed.checkpoint import DefaultLoadPlanner
    begin = time.monotonic()
    logger.info(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path, planner=DefaultLoadPlanner(allow_partial_load=True))
    logger.info(f"Finished loading chkpt in {time.monotonic() - begin:.2f} seconds.")
    
    # logger.info(f"Resizing model.freqs_cis to fit ctx_len ... ")
    # prev_freqs_cis_dim = model.freqs_cis.shape
    # model.recompute_freqs_cis(1048576)
    # logger.info(f"Resizing DONE: {prev_freqs_cis_dim} -> {model.freqs_cis.shape} ")

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

    max_new_tokens = 30

    ###################################################

    #pad_embed = model.language_model.model.get_input_embeddings()(torch.tensor([processor.tokenizer.pad_token_id], device=device))

    for j, batch in enumerate(data_loader):
        t0 = time.monotonic()

        # TODO: why there is one more dim? [1, 1, seq_len]
        input_ids = batch.input_ids[0].to(device)
        pixel_values = batch.pixel_values.to(device)
        #pixel_values = batch.pixel_values[0].to(torch.float32)
        #image_sizes = batch.image_sizes[0].to(device)

        logger.info(f"Test id {j}: input_ids shape = {input_ids.shape}, {type(input_ids)} / pixel_values: {pixel_values.shape}")

        seq_len = input_ids.shape[1]
        pad_size = max(max_new_tokens, (world_size * 2) - (seq_len % (world_size * 2)))

        if pad_size > 0:
            pad_token_tensors = torch.tensor([processor.tokenizer.pad_token_id] * pad_size, device=device)[None, :]
            input_ids = torch.cat([pad_token_tensors, input_ids], dim=1) 
            logger.info(f"Test id {j}: with padding ==> input_ids shape {seq_len} => {input_ids.shape}, {type(input_ids)}")
        
        # TODO check placements
        if isinstance(model.language_model.model.embed_tokens.weight, DTensor):
            placements = model.language_model.model.get_input_embeddings().weight.placements
            input_ids = distribute_tensor(input_ids, world_mesh['dp_shard_cp'], placements=placements)
            #model.language_model.model.embed_tokens.weight = torch.nn.Parameter(model.language_model.model.embed_tokens.weight).to(device)

        logger.info(f"embed_tokens: {type(model.language_model.model.embed_tokens.weight)}")
        # get context embeds (done in distributed)
        n_image = torch.tensor([[pixel_values.shape[1]]], device=device)
        with torch.no_grad():
            context_embeds = model.embed(
                input_ids=input_ids,
                pixel_values=pixel_values,
                n_image=n_image)

        logger.info(f"context_embeds: {type(context_embeds)}, {context_embeds.shape}, {context_embeds.dtype}")

        #context_embeds = context_embeds.to_local()
        if isinstance(input_ids, DTensor):
            tensor_list = [torch.empty_like(context_embeds) for _ in range(world_size)]
            dist.all_gather(tensor_list, context_embeds)
            context_embeds = torch.cat(tensor_list, dim=2)
            del tensor_list

        inputs_embeds = context_embeds.clone()
        #inputs_embeds = context_embeds

        #del context_embeds, input_ids
        del input_ids

        #input_ids = input_ids.to_local()

        gc.collect()
        torch.cuda.empty_cache()

        # Run generation
        new_tokens = []
        
        for i in range(max_new_tokens):
            seq_len = inputs_embeds.shape[1]
            # seq_len = inputs_embeds.shape[1]
            # pad_size = (world_size * 2) - (seq_len % (world_size * 2))

            max_seq = (seq_len // ((world_size * 2))) * (world_size * 2)
            inputs_embeds = inputs_embeds[:, -max_seq:, :]

            logger.info(f"inputs_embeds: {inputs_embeds.shape}")

            # if pad_size > 0:
            #     # TODO: pad at the head of input_embeds   
            #     pad_embeds = pad_embed.expand(pad_size, -1)  # Expand to match the required padding size
            #     pad_embeds = pad_embeds.unsqueeze(0)  # Add batch dimension
            #     if isinstance(inputs_embeds, DTensor):
            #     inputs_embeds = torch.cat([pad_embeds, inputs_embeds], dim=1)  # Concatenate padding at the head
            #     logger.info(f"inputs_embeds: {inputs_embeds.shape}, original: {seq_len} -> {seq_len + pad_size}")

            context_parallel_ctx = (
                utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[inputs_embeds],
                    cp_seq_dims=[1],
                    cp_no_restore_buffers={inputs_embeds},
                    cp_rotate_method=config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )
            
            with torch.no_grad(), train_context(context_parallel_ctx):
                output = model(inputs_embeds=inputs_embeds, num_logits_to_keep=1)
                #output = model(input_ids=input_ids, num_logits_to_keep=1)

            new_token, _ = sample(output.logits, need_probs=True, temperature=temperature, top_k=top_k)
            #new_token, _ = sample(output.logits, need_probs=True, temperature=1.0, top_k=top_k)
            new_tokens.append(new_token.item())

            del output

            with torch.no_grad():
                new_tokens_emb = model.embed(input_ids=new_token.unsqueeze(0))

            if isinstance(new_tokens_emb, DTensor):
                new_tokens_emb = new_tokens_emb.to_local()

            tensor_list = [torch.empty(inputs_embeds.shape, device=device, dtype=inputs_embeds.dtype) for _ in range(world_size)]
            dist.all_gather(tensor_list, inputs_embeds)
            inputs_embeds = torch.cat(tensor_list + [new_tokens_emb] , dim=1)
            del tensor_list

            # TODO: check if this all_gather works correctly.

            # with torch.no_grad():
            #     new_token_embds = model.embed(input_ids=torch.stack([torch.tensor(x, device=device) for x in new_tokens])[None, :])

            # inputs_embeds = torch.cat([context_embeds, new_token_embds], dim=1)
            
            gc.collect()
            torch.cuda.empty_cache()
            
        if rank == 0:
            r, b = color.red, color.blue
            output_text = tokenizer.decode(new_tokens)
            logger.info(f"{b}{output_text}\n{color.reset}")

            t1 = time.monotonic()
            elapsed_sec = t1 - t0
            logger.info(f"Generation completed in {elapsed_sec:.2f} seconds.")
    '''
    # Post process
    # B, T = generated_tokens.size()  # B: batch_size, T: total seq length
    # input_n_tokens = input_ids.size(1)
    #generated_n_tokens = T - input_n_tokens  # == max_new_tokens
    generated_n_tokens = 3

    if rank == 0:
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
