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
from collections import defaultdict
from io import BytesIO
from PIL import Image
import subprocess
import base64
import io

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
from torchtitan.models.llava_onevision import LlavaOnevisionForConditionalGeneration, llava_onevision_configs
from huggingface_hub import snapshot_download
import gc

#from env_utils.env.thor_env import ThorEnv

from ai2thor_client import ThorEnv
from ai2thor_utils import post_processing_action, get_templated_high_pddl_desc, serialize_action, setup_scene

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import sample

import wandb

class TrajManager:

    def __init__(self, init_event=None):
        self.traj_str = ""
        self.img_list = []
        
        self.last_event = init_event
        
        # state
        self.step = 0
        self.total_reward = 0
        self.agent_only_reward = 0
        self.log = defaultdict(list)

    def append_traj(self, traj_piece):
        self.traj_str += traj_piece
    
    def append_img(self, new_img):
        self.img_list.append(new_img)
        self.traj_str += '<image>'

    def add_log(self, log_type: str, log_data):
        if log_type not in self.log:
            self.log[log_type] = list()
        self.log[log_type].append(log_data)

    def copy_from_expert(self, expert):
        self.traj_str = expert.traj_str
        self.img_list = expert.img_list.copy()
        self.step = expert.step
        self.last_event = expert.last_event
        self.total_reward = expert.total_reward
    
    def load_state(self, last_log):
        self.log = last_log
        self.step = last_log['step'][-1]
        self.total_reward = last_log['total_reward'][-1]
        #self.agent_only_reward = last_log['agent_only_reward'][-1]


def save_json(filename, data, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def save_s3(output_dir, s3_path):
    # Parse the S3 path to get bucket and prefix
    s3_parts = s3_path.replace('s3://', '').split('/', 1)
    bucket_name = s3_parts[0]
    
    # Handle case where s3_path might not have a prefix
    prefix = s3_parts[1] if len(s3_parts) > 1 else ''
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    
    # Check if output_dir is a file or directory
    if os.path.isfile(output_dir):
        # If it's a file, just upload the single file
        file_name = os.path.basename(output_dir)
        object_name = os.path.join(prefix, file_name) if prefix else file_name
        s3_client.upload_file(output_dir, bucket_name, object_name)
    else:
        # If it's a directory, walk through and upload all files
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Create the S3 object name by replacing the local path prefix
                # with the S3 prefix
                relative_path = os.path.relpath(local_path, output_dir)
                s3_object_name = os.path.join(prefix, relative_path) if prefix else relative_path
                
                # Normalize paths for Windows compatibility
                s3_object_name = s3_object_name.replace('\\', '/')
                
                # Upload the file
                s3_client.upload_file(local_path, bucket_name, s3_object_name)


def is_valid_action(last_action: str):
    """
    Check if the last action is valid.
    
    Args:
        last_action (str): The last action taken
    
    Returns:
        bool: True if valid, False otherwise
    """
    # Define valid actions
    valid_actions = ['RotateLeft', 'RotateRight', 'MoveAhead', 'LookUp', 'LookDown',
                    'OpenObject', 'CloseObject', 'PickUpObject', 'PutObject', 'ToggleObjectOn', 
                    'ToggleObjectOff', 'SliceObject']

    tokens = last_action.split()

    if len(tokens) > 0 and len(tokens[0]) > 0:
        last_action = tokens[0]
        if last_action not in valid_actions:
            return False
    else:
        return False

    # for va in valid_actions:
    #     if va in last_action:
    #         return True
    
    return True


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
    
    ndims = torch.tensor([len(tensor.shape) if rank == src_rank else 0], dtype=torch.long, device=device)
    dist.broadcast(ndims, src=src_rank)
    ndims = ndims.item()
    
    shape_tensor = shape_tensor[:ndims]

    # Broadcast the shape
    dist.broadcast(shape_tensor, src=src_rank)
    
    shape = [dim for dim in shape_tensor.tolist() if dim > 0]
    
    # Create or use tensor
    if rank != src_rank or tensor is None:
        tensor = torch.zeros(shape, dtype=dtype, device=device)
    else:
        # Ensure source tensor is on the correct device and dtype
        tensor = tensor.to(device=device, dtype=dtype)
    
    # Broadcast data
    dist.broadcast(tensor, src=src_rank)
    
    return tensor


def distribute_value(value, device, dtype=None, root=0):

    # Determine the value type and appropriate tensor dtype if not specified
    if dtype is None:
        if isinstance(value, bool):
            dtype = torch.int32  # Using int32 for booleans for better compatibility
            tensor_value = 1 if value else 0
        elif isinstance(value, int):
            dtype = torch.int64  # Using int64 for integers
            tensor_value = value
        else:
            raise TypeError(f"Unsupported value type: {type(value)}. Please specify dtype explicitly.")
    else:
        tensor_value = value
    
    # Create tensor with the appropriate type
    value_tensor = torch.tensor([tensor_value], dtype=dtype, device=device)
    
    # Broadcast the tensor from root to all processes
    dist.broadcast(value_tensor, src=root)
    
    # Convert back to the original type and return
    if isinstance(value, bool):
        return bool(value_tensor.item())
    else:
        return value_tensor.item()


def simulate_with_expert(env, expert, expert_actions, update=True):
    success = True

    for t, action in enumerate(expert_actions):
        last_event = env.step(action)
        
        if last_event['lastActionSuccess']:
            if update:
                act_str = serialize_action(action)
                expert.append_traj('<|act|>' + act_str + '<|act|>')

                buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                buffer.seek(0)
                _image = Image.open(buffer)
                expert.append_img(_image)

                t_reward, done, sg_done = env.get_transition_reward(last_event, expert=True)
                expert.total_reward += t_reward
                expert.step += 1
                logger.info(f"expert.step: {expert.step}, action: {action['action']}, expert.total_reward: {expert.total_reward}, t_reward: {t_reward}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
        elif not last_event['lastActionSuccess'] and (last_event['lastAction'] in ["LookDown", "LookUp"]):
            if update:
                expert.total_reward += 0.0
                expert.step += 1
                logger.info(f"expert.step: {expert.step}, action: {action['action']} (but failed), expert.total_reward: {expert.total_reward}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
        else:
            logger.info(f"ERROR - expert initialization failed at {t} (action: {action})")
            logger.info(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
            success = False
            break
    
    expert.last_event = last_event
    
    return success


def interact_with_env(env, agent, action, eval_idx):
    # keep only newely generated action part not to call processor for the entire seq
    new_str = ""
    new_img = []

    subgoal_success = False
    try:
        # convert act to api_action
        if 'Object' in action:
            _action, obj_id = post_processing_action(action, env.last_event['objects'])
            if 'PutObject' in action and obj_id:
                inventory_object_id = env.last_event['inventoryObjects'][0]['objectId']
                put_action = dict(action="PutObject",
                            objectId=inventory_object_id,
                            receptacleObjectId=obj_id,
                            forceAction=True,
                            placeStationary=True)
                last_egent = env.step(put_action)
            elif obj_id:
                last_event = env.step(dict(action=_action, objectId=obj_id, forceAction=True))
            else:
                last_event = env.step(dict(action=_action, forceAction=True))
        else:
            last_event = env.step(dict(action=action, forceAction=True))

        t_success = last_event['lastActionSuccess']
    except:
        t_success = False

    if not t_success:
        logger.info(f"FAIL -- action: {action}")
        invalid_action_reward = 0.0
        return t_success, subgoal_success, invalid_action_reward, new_str, new_img

    agent.append_traj(action + '<|act|>')
    new_str += action + '<|act|>'
    
    buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
    buffer.seek(0)
    _image = Image.open(buffer)
    agent.append_img(_image)
    new_str += '<image>'
    new_img.append(_image)

    t_reward, t_done, sg_done = env.get_transition_reward(last_event, eval_idx, expert=False) # type: (float, bool)

    if sg_done: # done with the subgoal
        return t_success, sg_done, t_reward, new_str, new_img
    else: # for the next action prediction
        agent.append_traj('<|act|>')
        new_str += '<|act|>'
        agent.total_reward += t_reward
        agent.agent_only_reward += t_reward
        agent.step += 1
        return t_success, subgoal_success, t_reward, new_str, new_img


def process_input(traj_str, img_list, processor):
    batch = processor(images=img_list, text=traj_str, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)
    #batch = processor(images=self.img_list, text=prompt, padding=True, return_tensors="pt").to("cuda")
    logger.info(f"batch.input_ids {batch.input_ids.shape} {batch.input_ids.dtype}")
    logger.info(f"batch.pixel_values {batch.pixel_values.shape} {batch.pixel_values.dtype}")
    logger.info(f"[Prompt] {traj_str}")
    return batch.input_ids, batch.pixel_values


@torch.no_grad()
@record
def main(
    config_path: str,
    model_type: str,
    checkpoint_path: str,
    prompt: str,
    *,
    temperature: float = 1.0,
    max_new_tokens: int = 32,
    batch_size: int = 1,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    ctx_extension: str = None,
    ctx_extension_factor: float = None,
):
    init_logger()
    color = utils.Color

    wandb.init(project="long-traj-eval", id=model_type, resume="allow")

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
    processor.tokenizer.model_max_length = 2097152
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})
    
    data_dir = "data/long_alfred"
    log_dir = f"eval_logs/{model_type}"

    model_dtype = torch.bfloat16
    llm_config = llava_onevision_configs['7B'] # AutoConfig.from_pretrained

    if ctx_extension:
        logger.info(f"\n\n\t ******* Using dynamic context length: {ctx_extension} *********** \n")
        if ctx_extension == "longrope":
            llm_config.text_config.rope_scaling = {
                "rope_type": ctx_extension,  # 'longrope'
                "long_factor": ctx_extension_factor, # 4x the original context length
                "short_factor": 1,
                "factor": 1.0, # factor = config.max_position_embeddings / config.original_max_position_embeddings
                "original_max_position_embeddings": 32768,  # typical default; adjust based on model
            }
        else:
            llm_config.text_config.rope_scaling = {
                "rope_type": ctx_extension,  # or 'linear', 'longrope', etc.
                "factor": ctx_extension_factor,  # 4x the original context length
                "original_max_position_embeddings": 32768,  # typical default; adjust based on model
            }

    model_cls = LlavaOnevisionForConditionalGeneration
    model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, config=llm_config)
    buffers_dict = {k: v.clone() for k, v in model.named_buffers()}
    del model

    # model
    with torch.device("meta"):
        model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype)
        
    init_device = device_type
    model.to_empty(device=init_device)
    logger.info(f"rotary_emb.inv_freq: {model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq}")
    state_dict = {"model": model.state_dict()}
    logger.info(f"loading checkpoint: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    for name, buffer in buffers_dict.items():
        set_nested_attr(model, name, buffer.to(device_type))

    logger.info(f"rotary_emb.inv_freq: {model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq}")
    
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

    # Iterate over files in the data directory

    for floorplan_dir in sorted(os.listdir(data_dir)): # floorplan_dir: floorplan301, floorplan227, ... 
        floorplan_path = os.path.join(data_dir, floorplan_dir)
        if not os.path.isdir(floorplan_path):
            continue
        for file in sorted(os.listdir(floorplan_path)):
            if not file.endswith('.json'):
                continue
            
            traj_id = file.split(".")[0]
            logger.info(f"test id: {traj_id}")
            
            file_path = os.path.join(floorplan_path, file)
            with open(file_path, 'r') as f:
                traj_data = json.load(f)

            ############################################################################

            expert = TrajManager()
            agent = TrajManager()

            if dist.get_rank() == 0:
                reward_log_file = f"{log_dir}/{traj_id}.json"
                if os.path.exists(reward_log_file): # resume
                    agent.load_state(json.load(open(reward_log_file)))
                    last_step = agent.log['step'][-1]
                    last_high_idx = agent.log['high_idx'][-1]
                else:
                    last_step = 0
                    last_high_idx = 0
            else:
                last_step = 0
                last_high_idx = 0

            last_step = distribute_value(last_step, device)
            last_high_idx = distribute_value(last_high_idx, device)
            dist.barrier()

            if last_high_idx == 0: # init
                continue # for now we only care for resuming

            resume_sub_traj_idx = 0
            if last_high_idx == 0: # init
                if dist.get_rank() == 0:
                    last_event = setup_scene(env, traj_data, reward_type='dense')
                    agent.add_log(log_type="step", log_data=agent.step)
                    agent.add_log(log_type="total_reward", log_data=agent.total_reward)
                    agent.add_log(log_type="agent_reward", log_data=agent.agent_only_reward)
                    agent.add_log(log_type="token_length", log_data=0)
                    agent.add_log(log_type="action", log_data='INIT')
                    agent.add_log(log_type="subgoal", log_data='INIT')
                    agent.add_log(log_type="t_reward", log_data=0)
                    agent.add_log(log_type="high_idx", log_data=0)
                else:
                    logger.info(f"Rank: {dist.get_rank()} -- setup_scene")
            else:
                for ti, subtraj in enumerate(traj_data['sub_trajs']):
                    start, end = subtraj['high_pddl_idx']
                    if start <= last_high_idx + 1 <= end:
                        resume_sub_traj_idx = ti
                        logger.info(f" ========== [RESUME] resume_sub_traj_idx: {resume_sub_traj_idx} last_high_idx: {last_high_idx }========== ")
                        break
            
            if resume_sub_traj_idx > 0: # proceed experts by resuming point
                if dist.get_rank() == 0:
                    last_event = setup_scene(env, traj_data, reward_type='dense')
                else:
                    last_event = None

                for sub_task, sub_traj in zip(traj_data['sub_tasks'][:resume_sub_traj_idx], traj_data['sub_trajs'][:resume_sub_traj_idx]):
                    goal_str = f"<|goal|>Your main goal: {sub_task['task_desc']}<|goal|>"
                    expert.append_traj(goal_str)

                    if dist.get_rank() == 0:
                        buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                        buffer.seek(0)
                        _image = Image.open(buffer)
                        expert.append_img(_image)
                    else:
                        expert.append_img("<image>")

                    high_start, high_end = sub_traj['high_pddl_idx']
                    
                    if dist.get_rank() == 0:
                        task_info = sub_task['task_info']
                        num_subgoals = high_end - high_start
                        expert_plan = traj_data['plan']['high_pddl'][high_start:high_end]
                        env.set_task(task_info, num_subgoals, last_event, expert_plan)
                    else:
                        logger.info(f"Rank: {dist.get_rank()} -- env.set_task")

                    sim_success = False
                    for eval_idx, high_idx in enumerate(range(high_start, high_end)):
                        expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] == high_idx]

                        if len(expert_actions) == 0:
                            sim_success = False
                            break

                        if dist.get_rank() == 0:
                            #sim_success = simulate_with_expert(env, expert, expert_actions, subgoal_str, high_idx, update=True)
                            sim_success = simulate_with_expert(env, expert, expert_actions, update=True)
                        else:
                            logger.info(f"Rank: {dist.get_rank()} -- simulate_with_expert ")

                        sim_success_tensor = torch.tensor([1 if sim_success else 0], dtype=torch.int32, device=device)
                        dist.broadcast(sim_success_tensor, src=0)
                        sim_success = bool(sim_success_tensor.item())

                        if not sim_success:
                            break
                    # end of one sub task
                    if not sim_success:
                        break
                if not sim_success: # break, go to next json file
                    break

            ########################################
            # Main generation loop
            ########################################

            for sub_task, sub_traj in zip(traj_data['sub_tasks'][resume_sub_traj_idx:], traj_data['sub_trajs'][resume_sub_traj_idx:]):
                goal_str = f"<|goal|>Your main goal: {sub_task['task_desc']}<|goal|>"
                expert.append_traj(goal_str)
                
                if dist.get_rank() == 0:
                    buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                    buffer.seek(0)
                    _image = Image.open(buffer)
                    expert.append_img(_image)
                else:
                    expert.append_img("<image>")
                
                high_start, high_end = sub_traj['high_pddl_idx']
                
                if dist.get_rank() == 0:
                    # to set task-dependent rewards
                    num_subgoals = high_end - high_start
                    task_info = sub_task['task_info']
                    expert_plan = traj_data['plan']['high_pddl'][high_start:high_end]
                    env.set_task(task_info, num_subgoals, last_event, expert_plan)
                else:
                    logger.info(f"[Rank: {dist.get_rank()}] Main generation loop -- env.set_task")

                for eval_idx, high_idx in enumerate(range(high_start, high_end)):
                    subgoal_str = traj_data['plan']['high_pddl'][high_idx]['discrete_action']['action']
                    logger.info(f" ==== evaluating high_idx: {high_idx}, {traj_data['plan']['high_pddl'][high_idx]}")
                    expert.append_traj(f"<|plan|>Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][high_idx])}<|plan|><|act|>")
                    
                    cur_expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] == high_idx]
        
                    if len(cur_expert_actions) == 0:
                        sim_success = False
                        break

                    #########################################################################
                    # Agent actions
                    #########################################################################

                    if dist.get_rank() == 0:
                        input_ids, pixel_values = process_input(expert.traj_str, expert.img_list, processor)
                    else:
                        logger.info(f"Rank: {dist.get_rank()} -- waiting for process_input ... ")
                        input_ids, pixel_values = None, None

                    input_ids = broadcast_tensor(input_ids, torch.int32, device, src_rank=0)
                    pixel_values = broadcast_tensor(pixel_values, torch.bfloat16, device, src_rank=0)

                    done = False
                    n_invalid_actions = 0
                    while not done:
                        generated_tokens = generate(model, input_ids, pixel_values, config, device, cp_degree, world_mesh, act_tok_id, pad_tok_id)
                        gc.collect()

                        new_input_ids = None
                        new_pixel_values = None

                        if dist.get_rank() == 0:
                            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            logger.info(f"{color.blue}{output_text}\n{color.reset}")
                            success, done, t_reward, new_str, new_img = interact_with_env(env, agent, output_text, eval_idx)

                            agent.add_log(log_type="step", log_data=agent.step)
                            agent.add_log(log_type="total_reward", log_data=agent.total_reward)
                            agent.add_log(log_type="agent_reward", log_data=agent.agent_only_reward)
                            agent.add_log(log_type="token_length", log_data=int(input_ids.shape[1]))
                            agent.add_log(log_type="action", log_data=output_text)
                            agent.add_log(log_type="subgoal", log_data=subgoal_str)
                            agent.add_log(log_type="t_reward", log_data=t_reward)
                            agent.add_log(log_type="high_idx", log_data=high_idx)
                            logger.info(f"agent.step: {agent.step}, token: {int(input_ids.shape[1])}, sg_success: {done}, agent.total_reward: {agent.total_reward}, t_reward: {t_reward}, high_idx: {high_idx}, task.finished: {env.task.finished}")
                            
                            wandb.log({
                                f"{model_type}/total_reward": agent.total_reward,
                                f"{model_type}/agent_reward": agent.agent_only_reward
                            }, step=agent.step)

                            # check if the action is valid:
                            if (not is_valid_action(output_text)) or len(new_img) == 0:
                                n_invalid_actions += 1
                                done = True
                                logger.info(f"ERROR - agent failed to generate valid actions. Break")
                            else:
                                n_invalid_actions = 0 # reset the count
                                # get tensors ! 
                                new_input_ids, new_pixel_values = process_input(new_str, new_img, processor)
                        else:
                            success, done = None, None

                        success_tensor = torch.tensor([1 if success else 0], dtype=torch.int32, device=device)
                        dist.broadcast(success_tensor, src=0)
                        success = bool(success_tensor.item())

                        done_tensor = torch.tensor([1 if done else 0], dtype=torch.int32, device=device)
                        dist.broadcast(done_tensor, src=0)
                        done = bool(done_tensor.item())

                        n_inv_act_tensor = torch.tensor([n_invalid_actions], dtype=torch.int32, device=device)
                        dist.broadcast(n_inv_act_tensor, src=0)
                        n_invalid_actions = int(n_inv_act_tensor.item())

                        if (not success) or done or n_invalid_actions > 0:
                            logger.info(f"[Rank: {dist.get_rank()}] Break - success: {success} done: {done} n_invalid_actions: {n_invalid_actions}")
                            break

                        new_input_ids = broadcast_tensor(new_input_ids, torch.int32, device, src_rank=0)
                        new_pixel_values = broadcast_tensor(new_pixel_values, torch.bfloat16, device, src_rank=0)

                        input_ids = torch.concat([input_ids, new_input_ids], dim=1)
                        pixel_values = torch.concat([pixel_values, new_pixel_values], dim=0)
                    
                    #########################################################################
                    # Agent action done. Expert's simulation for the GT context 
                    #########################################################################

                    if n_invalid_actions > 0:
                        break

                    sim_success = False
                    if dist.get_rank() == 0:
                        last_event = setup_scene(env, traj_data, reward_type='dense')
                        # to set task-dependent rewards
                        env.set_task(task_info, num_subgoals, last_event, expert_plan)

                        prev_expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < high_idx]

                        if len(prev_expert_actions) > 0:
                            # replay by the current sub_goal
                            sim_success = simulate_with_expert(env, expert, prev_expert_actions, update=False)
                            if not sim_success:
                                break
                    
                        sim_success = simulate_with_expert(env, expert, cur_expert_actions, update=True)
                    else:
                        logger.info(f"Rank: {dist.get_rank()} -- simulating expert actions ... ")

                    sim_success_tensor = torch.tensor([1 if sim_success else 0], dtype=torch.int32, device=device)
                    dist.broadcast(sim_success_tensor, src=0)
                    sim_success = bool(sim_success_tensor.item())

                # end of one sub task. save logs
                if dist.get_rank() == 0:
                    save_json(reward_log_file, agent.log)
                    s3_path = f"s3://bosung-alfred/eval_logs_long/{model_type}"
                    # reward_log_file = f"{log_dir}/{traj_id}.json"
                    save_s3(reward_log_file, s3_path)
                else:
                    logger.info(f"Rank: {dist.get_rank()} -- uploading to S3 ... ")
                
                if n_invalid_actions > 0:
                    logger.info(f"Rank: {dist.get_rank()} -- n_invalid_actions: {n_invalid_actions}. Break. ")
                    break

                if not sim_success:
                    break # if expert traj fails, even next subgoal cannot make it

@torch.no_grad()
def generate(model, input_ids, pixel_values, config, device, cp_degree, world_mesh, act_tok_id, pad_tok_id, max_new_tokens=10):
    input_ids = input_ids.to(device)
    pixel_values = pixel_values.to(device)
    n_image =  torch.tensor([[pixel_values.shape[0]]], device=pixel_values.device)
    
    seq_len = input_ids.shape[1]
    #pad_size = max(max_new_tokens, (cp_degree * 2) - (seq_len % (cp_degree * 2)))
    pad_size = (cp_degree * 2) - (seq_len % (cp_degree * 2))

    if pad_size > 0:
        pad_token_tensors = torch.tensor([pad_tok_id] * pad_size, device=device)[None, :]
        input_ids = torch.cat([pad_token_tensors, input_ids], dim=1) 

    with torch.no_grad():
        inputs_embeds = model.embed(
            input_ids=input_ids,
            pixel_values=pixel_values.unsqueeze(0),
            n_image=n_image,
        )

    del input_ids

    gc.collect()

    train_context = utils.get_train_context(False, False)

    new_tokens = []
    for i in range(max_new_tokens):
        seq_len = inputs_embeds.shape[1]
        position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = position_ids.unsqueeze(0)

        max_seq = (seq_len // ((cp_degree * 2))) * (cp_degree * 2)
        
        inputs_embeds = inputs_embeds[:, -max_seq:, :]
        position_ids = position_ids[:, -max_seq:] 

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
        del logits
        logger.info(f"Rank: {dist.get_rank()},  new_token: {new_token.item()}")

        if dist.get_rank() == 0 and new_token.item() == act_tok_id:
            end_of_act = True
        else:
            end_of_act = None

        eoa_tensor = torch.tensor([1 if end_of_act else 0], dtype=torch.int32, device=device)
        dist.broadcast(eoa_tensor, src=0)
        end_of_act = bool(eoa_tensor.item())

        if end_of_act:
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

    gc.collect()
    return new_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="expert",
        help="model name",
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
        "--ctx_extension",
        type=str
    )
    parser.add_argument(
        "--ctx_extension_factor",
        type=float,
        default=4.0
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        top_k=args.top_k,
        seed=args.seed,
        deterministic=args.deterministic,
        ctx_extension=args.ctx_extension,
        ctx_extension_factor=args.ctx_extension_factor
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
