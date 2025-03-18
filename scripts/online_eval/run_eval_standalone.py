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
        self.agent_only_reward = last_log['agent_only_reward'][-1]


def save_json(filename, data, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


# def save_s3(output_dir, s3_path): # output_dir: outputs/checkpoints/step-xxxx
#     sync_command = f"aws s3 sync {output_dir} {s3_path}"
#     subprocess.run(
#         sync_command,
#         shell=True,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL
#     )
# def save_s3(output_dir, s3_path):
#     try:
#         os.system(f"aws s3 cp {output_dir} {s3_path}")
#     except:
#         logger.info(f"fail to run command: aws s3 cp {output_dir} {s3_path}")
    # sync_command = f"aws s3 cp {output_dir} {s3_path}"
    # subprocess.run(
    #     sync_command,
    #     shell=True,
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL
    # )

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


def set_nested_attr(obj, name, value):
    """since model.register_buffer() doesn't work on module names with '.',
       manually set a neste attribute for buffers"""
    parts = name.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


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
                logger.info(f"expert.step: {expert.step}, action: {action['action']} (but failed), expert.total_reward: {expert.total_reward}, t_reward: {t_reward}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
        else:
            logger.info(f"ERROR - expert initialization failed at {t} (action: {action})")
            logger.info(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
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
        invalid_action_reward = 0.0
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
    agent.total_reward += t_reward
    agent.agent_only_reward += t_reward
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
    
    device = torch.device(f"{device_type}")
    device_memory_monitor = build_device_memory_monitor()

    model_name = config.model.name

    # Tokenizer setup
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # build dataloader
    #processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})

    data_dir = "data/long_alfred"
    log_dir = f"eval_logs/{model_type}"

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

    act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
    pad_tok_id = processor.tokenizer.pad_token_id

    env = ThorEnv()

    # Iterate over files in the data directory
    for floorplan_dir in os.listdir(data_dir): # floorplan_dir: floorplan301, floorplan227, ... 
        floorplan_path = os.path.join(data_dir, floorplan_dir)
        if not os.path.isdir(floorplan_path):
            continue
        for file in os.listdir(floorplan_path):
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

            reward_log_file = f"{log_dir}/{traj_id}.json"
            if os.path.exists(reward_log_file): # resume
                agent.load_state(json.load(open(reward_log_file)))
                last_step = agent.log['step'][-1]
            else:
                last_step = 0

            last_event = setup_scene(env, traj_data, reward_type='dense')

            agent.add_log(log_type="step", log_data=agent.step)
            agent.add_log(log_type="total_reward", log_data=agent.total_reward)
            agent.add_log(log_type="agent_reward", log_data=agent.agent_only_reward)
            agent.add_log(log_type="token_length", log_data=0)
            agent.add_log(log_type="action", log_data='INIT')
            agent.add_log(log_type="subgoal", log_data='INIT')
            agent.add_log(log_type="t_reward", log_data=0)

            for sub_task, sub_traj in zip(traj_data['sub_tasks'], traj_data['sub_trajs']):
                goal_str = f"<|goal|>Your main goal: {sub_task['task_desc']}<|goal|>"
                expert.append_traj(goal_str)
                buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                buffer.seek(0)
                _image = Image.open(buffer)
                expert.append_img(_image)

                num_subgoals = sub_traj['high_pddl_idx'][1] - sub_traj['high_pddl_idx'][0]
                low_start, low_end = sub_traj['low_pddl_idx']

                # to set task-dependent rewards
                env.set_task(traj_data, last_event,
                            sub_traj_idx=sub_traj['sub_traj_idx'],
                            task_info=sub_task['task_info'],
                            task_type=sub_task['task_info']['goal'],
                            num_subgoals=num_subgoals,
                            reward_type='dense')

                for eval_idx, high_idx in enumerate(range(sub_traj['high_pddl_idx'][0], sub_traj['high_pddl_idx'][1])):
                    subgoal_str = traj_data['plan']['high_pddl'][high_idx]['discrete_action']['action']
                    logger.info(f" ==== evaluating high_idx: {high_idx}, {traj_data['plan']['high_pddl'][high_idx]}")
                    expert.append_traj(f"<|plan|>Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][high_idx])}<|plan|><|act|>")
                    
                    cur_expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] == high_idx]
                    
                    if last_step <= expert.step + len(cur_expert_actions):
                        #########################################################################
                        # Agent actions
                        #########################################################################

                        agent.copy_from_expert(expert)
                        input_ids, pixel_values = process_input(agent.traj_str, agent.img_list, processor)

                        done = False
                        
                        while not done:
                            generated_tokens = generate(model, input_ids, pixel_values, device, act_tok_id, pad_tok_id)
                            gc.collect()

                            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            logger.info(f"{color.blue}{output_text}\n{color.reset}")

                            success, done, t_reward = interact_with_env(env, agent, output_text, eval_idx)

                            agent.add_log(log_type="step", log_data=agent.step)
                            agent.add_log(log_type="total_reward", log_data=agent.total_reward)
                            agent.add_log(log_type="agent_reward", log_data=agent.agent_only_reward)
                            agent.add_log(log_type="token_length", log_data=int(input_ids.shape[1]))
                            agent.add_log(log_type="action", log_data=output_text)
                            agent.add_log(log_type="subgoal", log_data=subgoal_str)
                            agent.add_log(log_type="t_reward", log_data=t_reward)
                            logger.info(f"agent.step: {agent.step}, token: {int(input_ids.shape[1])}, sg_success: {done}, agent.total_reward: {agent.total_reward}, t_reward: {t_reward}, task.finished: {env.task.finished}")
                            
                            wandb.log({
                                f"{model_type}/total_reward": agent.total_reward,
                                f"{model_type}/agent_reward": agent.agent_only_reward
                            }, step=agent.step)
                            
                            if (not success) or done:
                                break

                            input_ids, pixel_values = process_input(agent.traj_str, agent.img_list, processor)

                    #########################################################################
                    # Agent action done. Expert's simulation for the GT context 
                    #########################################################################

                    last_event = setup_scene(env, traj_data, reward_type='dense')
                    prev_expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < high_idx]

                    if len(prev_expert_actions) > 0:
                        # replay by the current sub_goal
                        sim_success = simulate_with_expert(env, expert, prev_expert_actions, update=False)
                        if not sim_success:
                            break

                    # to set task-dependent rewards
                    # TODO
                    env.set_task(traj_data, last_event,
                            sub_traj_idx=sub_traj['sub_traj_idx'],
                            task_info=sub_task['task_info'],
                            task_type=sub_task['task_info']['goal'],
                            num_subgoals=num_subgoals,
                            reward_type='dense')

                    sim_success = simulate_with_expert(env, expert, cur_expert_actions, update=True)

                    if not sim_success:
                        break

                # end of one sub task. save logs
                save_json(reward_log_file, agent.log)
                s3_path = f"s3://bosung-alfred/eval_logs/{model_type}"
                # reward_log_file = f"{log_dir}/{traj_id}.json"
                save_s3(reward_log_file, s3_path)

                if not sim_success:
                    break

                if int(input_ids.shape[1]) > 300000: # a limit for the standalone model,
                    break


@torch.no_grad()
def generate(model, input_ids, pixel_values, device, act_tok_id, pad_tok_id, max_new_tokens=10):
    input_ids = input_ids.to("cuda")
    pixel_values = pixel_values.to("cuda")
    n_image =  torch.tensor([[pixel_values.shape[0]]], device=pixel_values.device)

    with torch.no_grad():
        inputs_embeds = model.embed(
            input_ids=input_ids,
            pixel_values=pixel_values.unsqueeze(0),
            n_image=n_image,
        )

    del input_ids

    gc.collect()

    new_tokens = []
    for i in range(max_new_tokens):
        # position_ids = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        # position_ids = position_ids.unsqueeze(0)

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

        inputs_embeds = torch.cat([inputs_embeds, new_tokens_emb] , dim=1)

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
