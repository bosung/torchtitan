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
        self.agent_only_reward = last_log['agent_reward'][-1]


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


def simulate_with_expert(env, expert, expert_actions, subgoal_str, high_idx, update=True):
    success = True

    for t, action in enumerate(expert_actions):
        last_event = env.step(action)
        
        if last_event['lastActionSuccess']:
            if update:
                t_reward, done, sg_done = env.get_transition_reward(last_event, expert=True)
                expert.total_reward += t_reward
                expert.step += 1
                logger.info(f"expert.step: {expert.step}, action: {action['action']}, expert.total_reward: {expert.total_reward}, t_reward: {t_reward}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
                expert.add_log(log_type="step", log_data=expert.step)
                expert.add_log(log_type="total_reward", log_data=expert.total_reward)
                expert.add_log(log_type="agent_reward", log_data=expert.agent_only_reward)
                expert.add_log(log_type="action", log_data=action['action'])
                expert.add_log(log_type="subgoal", log_data=subgoal_str)
                expert.add_log(log_type="t_reward", log_data=t_reward)
                expert.add_log(log_type="high_idx", log_data=high_idx)
            expert.last_event = last_event
        elif not last_event['lastActionSuccess'] and (last_event['lastAction'] in ["LookDown", "LookUp"]):
            if update:
                expert.total_reward += 0.0
                expert.step += 1
                logger.info(f"expert.step: {expert.step}, action: {action['action']} (but failed), expert.total_reward: {expert.total_reward}, task.goal_idx: {env.task.goal_idx}, task.finished: {env.task.finished}")
                expert.add_log(log_type="step", log_data=expert.step)
                expert.add_log(log_type="total_reward", log_data=expert.total_reward)
                expert.add_log(log_type="agent_reward", log_data=expert.agent_only_reward)
                expert.add_log(log_type="action", log_data=action['action'])
                expert.add_log(log_type="subgoal", log_data=subgoal_str)
                expert.add_log(log_type="t_reward", log_data=0.0)
                expert.add_log(log_type="high_idx", log_data=high_idx)
            expert.last_event = last_event
        else:
            logger.info(f"ERROR - expert initialization failed at {t} (action: {action})")
            logger.info(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
            success = False
            break
    
    return success


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
):
    init_logger()
    color = utils.Color

    wandb.init(project="long-traj-eval", id=model_type, resume="allow")

    # Load configuration from toml file
    config = JobConfig()
    config.parse_args([f"--job.config_file={config_path}"])
    config._validate_config()
    
    model_name = config.model.name

    # Tokenizer setup
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # build dataloader
    #processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>']})
    processor.tokenizer.add_special_tokens({"additional_special_tokens": ['<|act|>', '<|plan|>', '<|goal|>']})

    #data_dir = snapshot_download(repo_id="bosungkim/long_alfred", repo_type="dataset", local_dir="data/long_traj", allow_patterns="*.json")
    #data_dir = "data/long_traj"
    data_dir = 'data/long_alfred'
    log_dir = "online_eval/eval_logs_v2/expert"
    
    ###################################################

    #replay_success_file = 'online_eval/replay_success.json'
    #replay_success = json.loads(open(replay_success_file).read())

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

            logger.info(f"test id: {traj_id}")

            reward_log_file = f"eval_logs/expert/{traj_id}.json"
            if os.path.exists(reward_log_file):
                logger.info(f"Already evaluated {reward_log_file}")
                continue
            
            reward_log = {}
            
            file_path = os.path.join(floorplan_path, file)
            with open(file_path, 'r') as f:
                traj_data = json.load(f)

            ############################################################################

            last_event = setup_scene(env, traj_data, reward_type='dense')

            expert = TrajManager(init_event=last_event)

            expert.add_log(log_type="step", log_data=expert.step)
            expert.add_log(log_type="total_reward", log_data=expert.total_reward)
            expert.add_log(log_type="agent_reward", log_data=expert.agent_only_reward)
            expert.add_log(log_type="token_length", log_data=0)
            expert.add_log(log_type="action", log_data='INIT')
            expert.add_log(log_type="subgoal", log_data='INIT')
            expert.add_log(log_type="t_reward", log_data=0)

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
                    subgoal_str = traj_data['plan']['high_pddl'][high_idx]['discrete_action']['action']
                    logger.info(f" ==== evaluating high_idx: {high_idx}, {traj_data['plan']['high_pddl'][high_idx]['discrete_action']['action']}")
                    #expert.append_traj(f"<|plan|>Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][high_idx])}<|plan|><|act|>")

                    #########################################################################
                    # Agent action done. Expert's simulation for the GT context 
                    #########################################################################

                    expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] == high_idx]
                    sim_success = simulate_with_expert(env, expert, expert_actions, subgoal_str, high_idx, update=True)

                    if not sim_success:
                        break

                # end of one sub task

            if sim_success:
                save_json(reward_log_file, expert.log)
                logger.info(f"Replay success: {reward_log_file}")
            else:
                logger.info(f"Fail to replay: {traj_id}")

            #log_dir = "online_eval/logs"
            # s3_path = "s3://bosung-alfred/eval_logs/"
            # save_s3(log_dir, s3_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test generation")
    parser.add_argument(
        "--config", type=str, required=True, help="TOML config file path (required)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="expert"
    )

    args = parser.parse_args()

    main(
        config_path=args.config,
        model_type=args.model_type,
    )
