import os
import re
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader

from torchtitan.logging import logger

from datasets import Dataset, load_dataset

from PIL import Image
import tarfile
from io import BytesIO


def extract_and_convert_tar(tar_path):
    """Extracts a .tar file and converts all .jpg files inside to a list of PIL images."""
    pil_images = []
    
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.lower().endswith(".jpg"):
                file_obj = tar.extractfile(member)
                if file_obj:
                    image = Image.open(BytesIO(file_obj.read()))
                    image = image.convert("RGB")  # Ensure consistent format
                    pil_images.append(image)
    
    return pil_images


def pad_to_multiple(tensor, multiple=4, pad_token=0):
    length = tensor.shape[1]
    pad_length = (multiple - (length % multiple)) % multiple
    if pad_length > 0:
        pad_tensor = torch.full((tensor.shape[0], pad_length), pad_token, dtype=tensor.dtype)
        tensor = torch.cat([tensor, pad_tensor], dim=1)
    return tensor


class ALFREDDataset(IterableDataset, Stateful):

    def __init__(
        self,
        processor,
        img_data_dir,
        split: str = "train",
        max_seq_len: int = 131072,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        ignore_index: int = -100,
        eval: bool = False
    ) -> None:
        self.dataset_name = "alfred"
       
        self.processor = processor
        self.max_seq_len = max_seq_len
        self.infinite = infinite
        self.act_tok_id = processor.tokenizer('<|act|>').input_ids[0]
        self.eos_tok_id = processor.tokenizer.eos_token_id
        self.ignore_index = ignore_index
        self.eval = eval
        self.world_size = world_size

        # if not self.eval:
        #     self.max_seq_len = 131072
        
        self.split = split

        self.act_template = {
            "RotateLeft": "RotateLeft",
            "RotateRight": "RotateRight",
            "MoveAhead": "MoveAhead",
            "LookUp": "LookUp",
            "LookDown": "LookDown",
            "OpenObject": "OpenObject [object]",
            "CloseObject": "CloseObject [object]",
            "PickupObject": "PickupObject [object]",
            "PutObject": "PutObject [object] [receptacle]",
            "ToggleObjectOn": "ToggleObjectOn [object]",
            "ToggleObjectOff": "ToggleObjectOff [object]",
            "SliceObject": "SliceObject [object]",
            "NoOp": "NoOp",
        }

        self.img_data_dir = img_data_dir
        self.traj_data = []

        if len(self.traj_data) == 0:
            self._load_traj_data()

    def __len__(self):
        return len(self.traj_data)

    def __iter__(self):
        for traj in self.traj_data:
            print(f"Loading a new example ... ")
            chunks = self._load_sample(traj, chunk=True)

            if not isinstance(chunks, list):
                chunks = [chunks]

            for ci, chunk in enumerate(chunks):
                # # dict_keys(['input_ids', 'attention_mask', 'pixel_values', 'image_sizes'])
                output = self.processor(images=chunk['img_list'], text=chunk['lang_input'], return_tensors="pt")

                if self.eval:
                    yield output

                labels = output.input_ids.clone()

                act_tok = False
                for i, l in enumerate(labels[0]):
                    if (not act_tok) and l == self.act_tok_id: # 151648
                        act_tok = True
                        continue
                    
                    if (not act_tok) and l != self.act_tok_id:
                        labels[0][i] = self.ignore_index

                    if act_tok and l == self.act_tok_id:
                        act_tok = False
                
                input_ids = output.input_ids[:, :-1]
                labels = labels[:, 1:]

                # TODO: batch for pipelining
                for _ in range(2):
                    yield {
                        'input_ids': pad_to_multiple(input_ids, multiple=(self.world_size*2), pad_token=self.eos_tok_id), # Pad for TP. 8 covers most cases.
                        'pixel_values': output.pixel_values, 
                        'image_sizes': output.image_sizes,
                        'labels': pad_to_multiple(labels, multiple=(self.world_size*2), pad_token=self.ignore_index),
                    }

    def _load_sample(self, traj, chunk=True):
        traj = json.loads(traj['text'])
        chunk_seq_list, chunk_img_idx = self.seq_preprocess(traj)

        img_tar_file = traj['img_tar']
        traj_imgs = set([x['image_name'].split(".")[0] for x in traj['images']])
        tar_file = os.path.join(self.img_data_dir, img_tar_file)

        img_list = extract_and_convert_tar(tar_file)

        chunks = []

        for input_seq, (img_start, img_end) in zip(chunk_seq_list, chunk_img_idx):
            chunks.append({
                'lang_input': input_seq,
                'img_list': img_list[img_start:img_end],
                'task_goal': traj['turk_annotations']['anns'][0]['task_desc'],
                'traj': traj,
            })

        return chunks

    def _load_traj_data(self):
        # self.traj_data = [x for x in load_dataset("bosungkim/alfred-small-traj", split=self.split)]
        self.traj_data = load_dataset("bosungkim/alfred-small-traj", split=self.split)

    def seq_preprocess(self, traj): 
        # Prepare: low_idx_to_image
        low_idx_2_image = defaultdict(list)
        for im_info in traj['images']:
            low_idx_2_image[im_info['low_idx']].append(im_info['image_name'])

        # Prepare
        high_idx_2_low_act_list = defaultdict(list)
        for low_idx, low_act in enumerate(traj['plan']['low_actions']):
            high_idx = low_act['high_idx']
            low_act['low_idx'] = low_idx
            if len(high_idx_2_low_act_list[high_idx]) > 0:
                assert high_idx_2_low_act_list[high_idx][-1]['low_idx'] < low_act['low_idx']
            high_idx_2_low_act_list[high_idx].append(low_act)

        # start: make squences here
        main_goal_str = "Your main goal: "
        if 'turk_annotations' in traj:
            main_goal_str += traj['turk_annotations']['anns'][0]['task_desc']
        # else we need to use templated desc .. later
        n_main_goal_tokens = len(self.processor(text=main_goal_str).input_ids)

        chunk_seq_list = []
        chunk_img_idx = []

        chunk_seq = main_goal_str
        n_chunk_tokens = n_main_goal_tokens # for chunking
        n_chunk_img = 0
        img_start_idx = 0

        for high_idx, low_act_list in high_idx_2_low_act_list.items():
            plan_str = f" Plan: {self.get_templated_high_pddl_desc(traj['plan']['high_pddl'][high_idx])}"
            
            high_plan_seq = ""
            high_plan_seq += plan_str
            n_high_plan_tokens = len(self.processor(text=plan_str).input_ids)
            n_high_plan_img = 0

            for low_idx, low_act in enumerate(low_act_list):
                action_str = self.serialize_action(low_act['api_action'])
                low_act_seq = action_str
                action_str_tok = self.processor(text=action_str).input_ids   
                n_low_act_tokens = len(action_str_tok)
                # count tokens for images
                low_act_seq += (" <image>" * len(low_idx_2_image[low_idx]))
                n_low_act_tokens += (1485 * len(low_idx_2_image[low_idx])) # one frame is 1485 tokens
                    
                if (n_high_plan_tokens + n_low_act_tokens) >= self.max_seq_len:
                    break # do not add this low_act and break
                else:
                    n_high_plan_tokens += n_low_act_tokens
                    high_plan_seq += low_act_seq
                    n_high_plan_img += len(low_idx_2_image[low_idx])

            assert n_high_plan_tokens < self.max_seq_len

            if (n_chunk_tokens + n_high_plan_tokens) >= self.max_seq_len:
                chunk_seq_list.append(chunk_seq)
                chunk_img_idx.append([img_start_idx, img_start_idx + n_chunk_img])

                # reset for next chunk
                chunk_seq = main_goal_str + high_plan_seq
                n_chunk_tokens = n_main_goal_tokens + n_high_plan_tokens
                img_start_idx = img_start_idx + n_chunk_img
                n_chunk_img = n_high_plan_img
            else:
                chunk_seq += high_plan_seq
                n_chunk_tokens += n_high_plan_tokens
                n_chunk_img += n_high_plan_img

        chunk_seq_list.append(chunk_seq)
        chunk_img_idx.append([img_start_idx, img_start_idx + n_chunk_img])

        logger.info(f"# of chunks: {len(chunk_seq_list)}, chunk_img_idx: {chunk_img_idx}")

        return chunk_seq_list, chunk_img_idx

    def serialize_action(self, act):
        template = self.act_template[act['action']]
        if 'objectId' in act:
            template = template.replace("[object]", act['objectId'].split("|")[0])
        if 'receptacleObjectId' in act:
            template = template.replace("[receptacle]", act['receptacleObjectId'].split("|")[0])
        return '<|act|>' + template + '<|act|>'
    
    def get_templated_high_pddl_desc(self, high_pddl):
        a_type = high_pddl['discrete_action']['action']
        args = high_pddl['discrete_action']['args'] if 'args' in high_pddl['discrete_action'] else None

        if 'objectId' in high_pddl['planner_action']:
            objectId = high_pddl['planner_action']['objectId']
            obj_name = objectId.split("|")[0]
        if 'receptacleObjectId' in high_pddl['planner_action']:
            receptacleObjectId = high_pddl['planner_action']['receptacleObjectId']
            recep_name = receptacleObjectId.split("|")[0]

        templated_str = ""

        if 'GotoLocation' in a_type:
            templated_str = f"go to the {args[0]}"
        elif 'OpenObject' in a_type:
            templated_str = f"open the {obj_name}"
        elif 'CloseObject' in a_type:
            templated_str = f"close the {obj_name}"
        elif 'PickupObject' in a_type:
            templated_str = f"pick up the {obj_name}"
        elif 'PutObject' in a_type:
            templated_str = f"put the {obj_name} in the {recep_name}"
        elif 'CleanObject' in a_type:
            templated_str = f"wash the {obj_name}"
        elif 'HeatObject' in a_type:
            templated_str = f"heat the {obj_name}"
        elif 'CoolObject' in a_type:
            templated_str = f"cool the {obj_name}"
        elif 'ToggleObject' in a_type:
            templated_str = f"toggle {obj_name}"
        elif 'SliceObject' in a_type:
            templated_str = f"slice the {obj_name}"
        elif 'End' in a_type:
            templated_str = "<<STOP>>"

        return templated_str
