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


class ALFREDDataset(IterableDataset, Stateful):

    def __init__(
        self,
        processor,
        img_data_dir,
        split: str = "train",
        seq_len: int = 2048,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
    ) -> None:
        self.dataset_name = "alfred"
       
        self.processor = processor
        self.seq_len = seq_len
        self.infinite = infinite
        
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
            print(f"Loading example ... ")
            sample = self._load_sample(traj)
            yield self.processor(images=sample['img_list'], text=sample['lang_input'], return_tensors="pt")

    def _load_sample(self, traj):
        traj = json.loads(traj['text'])
        input_seq = self.seq_preprocess(traj)
        
        img_tar_file = traj['img_tar']
        traj_imgs = set([x['image_name'].split(".")[0] for x in traj['images']])
        tar_file = os.path.join(self.img_data_dir, img_tar_file)

        img_list = extract_and_convert_tar(tar_file)
        
        return {
            'lang_input': input_seq,
            'img_list': img_list,
            'task_goal': traj['turk_annotations']['anns'][0]['task_desc'],
            'traj': traj,
        }

    def _load_traj_data(self):
        # self.traj_data = [x for x in load_dataset("bosungkim/alfred-small-traj", split=self.split)]
        self.traj_data = load_dataset("bosungkim/alfred-small-traj", split=self.split)

    def seq_preprocess(self, traj):
        # with high_pddl
        input_seq = "Your Main Goal: "
        if 'turk_annotations' in traj:
            input_seq += traj['turk_annotations']['anns'][0]['task_desc']
        # else we need to use templated desc .. later

        # low_idx_to_image
        low_idx_2_image = defaultdict(list)
        for im_info in traj['images']:
            low_idx_2_image[im_info['low_idx']].append(im_info['image_name'])

        cur_high_idx = -1

        for low_idx, low_act in enumerate(traj['plan']['low_actions']):
            
            if low_act['high_idx'] > cur_high_idx:
                input_seq += f" Plan: {self.get_templated_high_pddl_desc(traj['plan']['high_pddl'][low_act['high_idx']])}"
                cur_high_idx = low_act['high_idx']

            input_seq += f" {self.serialize_action(low_act['api_action'])}"

            for imgfile in low_idx_2_image[low_idx]:
                input_seq += " <image>"

            # if low_idx == 20:
            #     break
        print(input_seq)

        return input_seq

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
