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
from torchtitan.models.llava_onevision import LlavaOnevisionForConditionalGeneration
from huggingface_hub import snapshot_download
import gc

#from env_utils.env.thor_env import ThorEnv
from ai2thor_client import ThorEnv
from ai2thor_utils import post_processing_action, get_templated_high_pddl_desc, serialize_action

# support running w/o installing as package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate._generation import sample

# AWS_S3_PATH = os.environ['AWS_S3_PATH']

import wandb


class TrajManager:

    def __init__(self, init_event=None):
        self.traj_str = ""
        self.img_list = []
        self.total_reward = 0
        self.step = 0
        # self.t_reward_arr = []
        # self.token_reward_arr = []
        self.log = defaultdict(list)
        self.last_event = init_event

    def append_traj(self, traj_piece):
        self.traj_str += traj_piece
    
    def append_img(self, new_img):
        self.img_list.append(new_img)
        self.traj_str += '<image>'

    def add_log(self, log_type: str, log_data: list):
        self.log[log_type].append(log_data)

    def copy_from_expert(self, expert):
        self.traj_str = expert.traj_str
        self.img_list = expert.img_list.copy()
        self.step = expert.step
        self.last_event = expert.last_event


def save_json(filename, data, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def save_s3(output_dir, s3_path): # output_dir: outputs/checkpoints/step-xxxx
    sync_command = f"aws s3 sync {output_dir} {s3_path}"
    subprocess.run(
        sync_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


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


def simulate_with_expert(env, expert, expert_actions):
    success = True

    for t, action in enumerate(expert_actions):
        last_event = env.step(action)
        
        if last_event['lastActionSuccess']:
            act_str = serialize_action(action)
            expert.append_traj('<|act|>' + act_str + '<|act|>')

            buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
            buffer.seek(0)
            _image = Image.open(buffer)
            expert.append_img(_image)

            t_reward, done, sg_done = env.get_transition_reward(last_event, expert=True)
            expert.step += 1
            print(f"expert.step: {expert.step}, action: {action['action']}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
            
        elif not last_event['lastActionSuccess'] and (last_event['lastAction'] in ["LookDown", "LookUp"]):
            pass
        else:
            print(f"ERROR - expert initialization failed at {t} (action: {action})")
            print(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
            success = False
            break
    
    expert.last_event = last_event
    
    return success

def interact_with_env(env, agent, action, eval_idx):

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
        invalid_action_reward = -0.5
        return t_success, subgoal_success, invalid_action_reward

    agent.append_traj(action + '<|act|>') 
    
    buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
    buffer.seek(0)
    _image = Image.open(buffer)
    agent.append_img(_image)

    t_reward, t_done, sg_done = env.get_transition_reward(last_event, eval_idx, expert=False) # type: (float, bool)

    if sg_done:
        return t_success, sg_done, t_reward
        
    # if self.t > (self.gt_n_step * 3):
    #     logger.info(f"fail due to the time step limit -- t: {self.t} > {(self.gt_n_step * 3)} (limit)")
    #     return t_success, subgoal_success

    # for the next action prediction
    agent.append_traj('<|act|>')
    agent.step += 1

    return t_success, subgoal_success, t_reward

def process_input(traj_str, img_list, processor):
    batch = processor(images=img_list, text=traj_str, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)
    #batch = processor(images=self.img_list, text=prompt, padding=True, return_tensors="pt").to("cuda")
    logger.info(f"batch.input_ids {batch.input_ids.shape} {batch.input_ids.dtype}")
    logger.info(f"batch.pixel_values {batch.pixel_values.shape} {batch.pixel_values.dtype}")
    logger.info(f"[Prompt] {traj_str}")
    return batch.input_ids, batch.pixel_values

def setup_scene(env, traj, reward_type='dense'):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj['scene']['scene_num']
    object_poses = traj['scene']['object_poses']
    dirty_and_empty = traj['scene']['dirty_and_empty']
    object_toggles = traj['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    last_event = env.reset(scene_name)
    last_event = env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    last_event = env.step(dict(traj['scene']['init_action']))
    
    # setup task for reward
    env.set_task(traj, last_event, reward_type=reward_type)

    logger.info(f"Setup scene: {scene_name}")
    return last_event


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
    ctx_len: int = 8192
):
    init_logger()
    color = utils.Color

    wandb.init(project="long-traj-eval", id=model_type, resume="allow")

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()
    
    cp_degree = 1
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #device = torch.device(f"{device_type}:{local_rank}")
    device = torch.device(f"{device_type}")
    # device_module.set_device(device)
    device_memory_monitor = build_device_memory_monitor()

    model_name = config.model.name

    # Tokenizer setup
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # build dataloader
    #processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})

    #eval_done_idx = check_existing_evals(s3_path=AWS_S3_PATH)   

    data_dir = snapshot_download(repo_id="bosungkim/long_alfred", repo_type="dataset", local_dir="data/long_traj", allow_patterns="*.json")
    log_dir = "online_eval/logs"

    model_dtype = torch.bfloat16
    model_cls = LlavaOnevisionForConditionalGeneration
    model = model_cls.from_pretrained(model_name, torch_dtype=model_dtype, device_map="auto", low_cpu_mem_usage=True)
        
    init_device = device_type
    #print(model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq)
    #model.to_empty(device=init_device)
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    #print(model.language_model.model.layers[0].self_attn.rotary_emb.inv_freq)
    #for name, buffer in buffers_dict.items():
    #    set_nested_attr(model, name, buffer)
    model.eval()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    device_memory_monitor.reset_peak_stats()
    
    ###################################################

    replay_success_file = 'online_eval/replay_success.json'
    replay_success = json.loads(open(replay_success_file).read())

    act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
    pad_tok_id = processor.tokenizer.pad_token_id

    env = ThorEnv()

    # Iterate over files in the data directory
    for floorplan_dir in os.listdir(data_dir):
        floorplan_path = os.path.join(data_dir, floorplan_dir)
        if not os.path.isdir(floorplan_path):
            continue
        for file in os.listdir(floorplan_path):
            if not file.endswith('.json'):
                continue
            
            traj_id = file.split(".")[0]
            if traj_id in replay_success and not replay_success[traj_id]["success"]:
                continue
            if traj_id not in replay_success:
                continue

            logger.info(f"test id: {traj_id}")

            reward_log_file = f"online_eval/logs/{traj_id}.json"
            if os.path.exists(reward_log_file):
                reward_log = json.load(open(reward_log_file))
            else:
                reward_log = {}
            
            if model_type in reward_log: # this is resume thing ...
                log_data = reward_log[model_type]['x_time']
                last_step = log_data[-1][0]
            else:
                reward_log[model_type] = {}
                log_data = []
                last_step = 0

            file_path = os.path.join(floorplan_path, file)
            with open(file_path, 'r') as f:
                traj_data = json.load(f)

            ############################################################################

            last_event = setup_scene(env, traj_data, reward_type='dense')

            # 1. Don't think about the resume right now

            total_expert_reward = 0
            total_agent_reward = 0
            global_t = 0

            expert = TrajManager(init_event=last_event)
            agent = TrajManager()

            agent.add_log(log_type="token_reward", log_data=[0, total_agent_reward])
            agent.add_log(log_type="t_reward", log_data=[agent.step, total_agent_reward])

            for sub_task, sub_traj in zip(traj_data['sub_tasks'], traj_data['sub_trajs']):
                goal_str = f"<|goal|>Your main goal: {sub_task['task_desc']}<|goal|>"
                expert.append_traj(goal_str)
                buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                buffer.seek(0)
                _image = Image.open(buffer)
                expert.append_img(_image)

                num_subgoals = sub_traj['high_pddl_idx'][1] - sub_traj['high_pddl_idx'][0]
                low_start, low_end = sub_traj['low_pddl_idx']

                env.set_task(traj_data, last_event,
                            sub_traj_idx=sub_traj['sub_traj_idx'],
                            task_info=sub_task['task_info'],
                            task_type=sub_task['task_info']['goal'],
                            num_subgoals=num_subgoals,
                            reward_type='dense')

                #subgoal_idxs = eval_subgoals.get_subgoal_idxs(traj)

                for eval_idx, high_idx in enumerate(range(sub_traj['high_pddl_idx'][0], sub_traj['high_pddl_idx'][1])):
                    logger.info(f" ==== evaluating high_idx: {high_idx}, {traj_data['plan']['high_pddl'][high_idx]['discrete_action']['action']}")
                    expert.append_traj(f"<|plan|>Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][high_idx])}<|plan|><|act|>")

                    #########################################################################
                    # Agent actions
                    #########################################################################

                    agent.copy_from_expert(expert)
                    input_ids, pixel_values = process_input(agent.traj_str, agent.img_list, processor)

                    done = False
                    
                    while not done:
                        generated_tokens = generate(model, input_ids, pixel_values, config, device, 1, act_tok_id, pad_tok_id)
                        gc.collect()

                        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        logger.info(f"{color.blue}{output_text}\n{color.reset}")

                        success, done, t_reward = interact_with_env(env, agent, output_text, eval_idx)

                        total_agent_reward += t_reward
                        agent.add_log(log_type="token_reward", log_data=[int(input_ids.shape[1]), total_agent_reward])
                        agent.add_log(log_type="t_reward", log_data=[agent.step, total_agent_reward])
                        logger.info(f"agent.step: {agent.step}, token: {int(input_ids.shape[1])}, sg_success: {done}, total_agent_reward: {total_agent_reward}, t_reward: {t_reward}, task.finished: {env.task.finished}")
                        
                        if (not success) or done:
                            break

                        input_ids, pixel_values = process_input(agent.traj_str, agent.img_list, processor)

                    #########################################################################
                    # Agent action done. Expert's simulation for the GT context 
                    #########################################################################

                    last_event = setup_scene(env, traj_data, reward_type='dense')
                    env.set_task(traj_data, last_event,
                            sub_traj_idx=sub_traj['sub_traj_idx'],
                            task_info=sub_task['task_info'],
                            task_type=sub_task['task_info']['goal'],
                            num_subgoals=num_subgoals,
                            reward_type='dense')

                    expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] <= high_idx]
                    expert = TrajManager()
                    sim_success = simulate_with_expert(env, expert, expert_actions)

                    if not sim_success:
                        break

                # end of one sub task
                # log
                table = wandb.Table(data=agent.log["t_reward"], columns=["t", "Reward"])
                wandb.log({f"{traj_id}/{model_type}": wandb.plot.line(table, "t", "Reward",title="Reward")})

                table = wandb.Table(data=agent.log["token_reward"], columns=["context_length", "Reward"])
                wandb.log({f"{traj_id}/{model_type}": wandb.plot.line(table, "context_length", "Reward",title="Reward")})
                
                if model_type not in reward_log:
                    reward_log[model_type] = {}

                reward_log[model_type]['x_time'] = agent.log["t_reward"]
                reward_log[model_type]['x_token'] = agent.log["token_reward"]

                save_json(reward_log_file, reward_log)

                log_dir = "online_eval/logs"
                s3_path = "s3://bosung-alfred/eval_logs/"
                save_s3(log_dir, s3_path)


@torch.no_grad()
def generate(model, input_ids, pixel_values, config, device, cp_degree, act_tok_id, pad_tok_id, max_new_tokens=10):
    input_ids = input_ids.to("cuda")
    pixel_values = pixel_values.to("cuda")
    n_image =  torch.tensor([[pixel_values.shape[0]]], device=pixel_values.device)
    
    seq_len = input_ids.shape[1]
    pad_size = max(max_new_tokens, (cp_degree * 2) - (seq_len % (cp_degree * 2)))

    if pad_size > 0:
        pad_token_tensors = torch.tensor([pad_tok_id] * pad_size, device="cuda")[None, :]
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
        max_seq = (seq_len // ((cp_degree * 2))) * (cp_degree * 2)
        inputs_embeds = inputs_embeds[:, -max_seq:, :]

        # position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        # position_ids = position_ids.unsqueeze(0)

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
                #position_ids=position_ids, # you will need later soon
                use_cache=False,
                num_logits_to_keep=1)

        new_token, _ = sample(logits, need_probs=True, temperature=1.0)
        del logits

        if new_token.item() == act_tok_id:
            break
        
        new_tokens.append(new_token.item())

        with torch.no_grad():
            new_tokens_emb = model.embed(input_ids=new_token.unsqueeze(0))

        if isinstance(new_tokens_emb, DTensor):
            new_tokens_emb = new_tokens_emb.to_local()

        #tensor_list = [torch.empty(inputs_embeds.shape, device=device, dtype=inputs_embeds.dtype) for _ in range(cp_degree)]
        #dist.all_gather(tensor_list, inputs_embeds)
        inputs_embeds = torch.cat([inputs_embeds, new_tokens_emb] , dim=1)
        #del tensor_list

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
        "--ctx_len",
        type=int,
        default=8192,
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
        ctx_len=args.ctx_len
    )

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
