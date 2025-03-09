import os
from re import A
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime
import time
import random
from collections import defaultdict
import subprocess
import base64
import io
#from .env.thor_env import ThorEnv
#from eval.eval import Eval

#from data import ACT_TEMPLATE, serialize_action
#from ai2thor_utils import visible_and_pickupable, post_processing_action

import torch
from torch.cuda.amp import autocast

import torch.distributed as dist

from torchtitan.logging import logger


def visible(obj_list):
    return [obj for obj in obj_list if obj['visible']]

def find_visible_and_property(objects, objname, property_name):
    for obj in objects:
        if objname.lower() in obj['name'].lower() and obj['visible'] and obj[property_name]:
            print(obj['name'], obj['objectId'])
            return obj['objectId']
    return None

def visible_and_pickupable(objects, objname):
    return find_visible_and_property(objects, objname, 'pickupable')

def visible_and_receptacle(objects, objname):
    return find_visible_and_property(objects, objname, 'receptacle')

def visible_and_openable(objects, objname):
    return find_visible_and_property(objects, objname, 'openable')

def visible_and_sliceable(objects, objname):
    return find_visible_and_property(objects, objname, 'sliceable')

def visible_and_toggleable(objects, objname):
    return find_visible_and_property(objects, objname, 'toggleable')

def visible_and_isOpen(objects, objname): # closeable
    return find_visible_and_property(objects, objname, 'isOpen')


def post_processing_action(action, objects, objname=None):
    actions_map = {
        "OpenObject": visible_and_openable,
        "CloseObject": visible_and_isOpen,
        "PickupObject": visible_and_pickupable,
        "PutObject": visible_and_receptacle,
        "ToggleObjectOn": visible_and_toggleable,
        "ToggleObjectOff": visible_and_toggleable,
        "SliceObject": visible_and_sliceable,
    }
    
    try: 
        if action.startswith('PutObject'):
            receptacle_obj = action.split()[-1].strip()
            return "PutObject", visible_and_receptacle(objects, receptacle_obj)
        elif action.startswith('ToggleObject'):
            objname = action.split()[-1].strip()
            return "ToggleObject", visible_and_toggleable(objects, objname)

        for action_prefix, func in actions_map.items():
            if action.startswith('PutObject'):
                receptacle_obj = action.split()[-1].strip()
                return action_prefix, func(objects, receptacle_obj)
            elif action.startswith(action_prefix):
                objname = action.split(action_prefix)[-1].strip()
                return action_prefix, func(objects, objname)
    except: # action parsing error
        return None, None
    
    return None, None


ACT_TEMPLATE = {
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
    "NoOp": "NoOp",}


def serialize_action(act):
    if act['action'].find("_") >= 0:
        act['action'] = act['action'].split("_")[0]
    
    template = ACT_TEMPLATE[act['action']]
    if 'objectId' in act:
        template = template.replace("[object]", act['objectId'].split("|")[0])
    if 'receptacleObjectId' in act:
        template = template.replace("[receptacle]", act['receptacleObjectId'].split("|")[0])
    return template

def get_templated_high_pddl_desc(high_pddl):
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

class EvalSubgoals():
    '''
    evaluate subgoals by teacher-forching expert demonstrations
    '''

    # subgoal types
    ALL_SUBGOALS = ['GotoLocation', 'PickupObject', 'PutObject', 'CoolObject', 'HeatObject', 'CleanObject', 'SliceObject', 'ToggleObject']

    def __init__(self):
        # success and failure lists
        self.create_stats()

        # make subgoals list
        self.subgoals_to_evaluate = self.ALL_SUBGOALS # if args.subgoals.lower() == "all" else args.subgoals.split(',')
        #subgoals_to_evaluate = [sg for sg in subgoals_to_evaluate if sg in cls.ALL_SUBGOALS]

        # create empty stats per subgoal
        for sg in self.subgoals_to_evaluate:
            self.successes[sg] = list()
            self.failures[sg] = list()

        # set random seed for shuffling
        random.seed(int(time.time()))

    def get_subgoal_idxs(self, traj):
        return [sg['high_idx'] for sg in traj['plan']['high_pddl'] if sg['discrete_action']['action'] in self.subgoals_to_evaluate]

    @classmethod
    def setup_scene(cls, env, traj, reward_type='dense'):
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

        logger.info(f"Setup scene: {scene_name} Task: {traj['task_type']} {traj['pddl_params']}")
        return last_event

    def simulate_with_expert(self, env, traj_data, eval_idx, teacher_forcing=True, max_steps=2000):
        # expert demonstration to reach eval_idx-1
        #expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]
        self.expert_init_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]
        gt_init_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] <= eval_idx]
        self.gt_n_step = len(gt_init_actions)

        # subgoal info
        # if eval_idx >= len(traj_data['plan']['high_pddl']): # no NoOp at the last high_pddl
        #     return False
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']

        # task goal info
        r_idx = 0  # TODO remove turk_annotations's
        task_desc = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # print subgoal info
        print(f" ======== Evaluating: {traj_data['task_id']}\nTask desc: {task_desc}")
        print(f" ======== Subgoal[{eval_idx}] {subgoal_action}")

        # setup scene
        reward_type = 'dense'
        last_event = self.setup_scene(env, traj_data, reward_type=reward_type)
        if not last_event['lastActionSuccess']:
            logger.info(f"ERROR - Initialization fail !")
            return False

        # for model inputs
        self.img_list = []
        self.prefix = ""

        main_goal_str = "Your main goal: "
        if 'turk_annotations' in traj_data:
            main_goal_str += traj_data['turk_annotations']['anns'][0]['task_desc']
        self.prefix = main_goal_str
        
        self.t = 0
        prev_high_idx, curr_high_idx = -1, 0
        # Do expert action here:
        for t, action in enumerate(self.expert_init_actions):
            curr_high_idx = traj_data['plan']['low_actions'][t]['high_idx']
            if  curr_high_idx > prev_high_idx:
                self.prefix += f" Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][curr_high_idx])}"
                prev_high_idx = curr_high_idx

                # add image (new state) for the new plan
                # buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
                # buffer.seek(0)
                # curr_image = Image.open(buffer)

                # self.img_list.append(curr_image)
                # self.prefix += "<image>"

            if action['action'] in ['ToggleObjectOn', 'ToggleObjectOff']:
                action['forceAction'] = True

            last_event = env.step(action)

            if not last_event['lastActionSuccess']:
                if not (last_event['lastAction'] in ["LookDown", "LookUp"]):
                    logger.info(f"ERROR - expert initialization failed at {t} (action: {action})")
                    logger.info(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
                    return False

            act_str = serialize_action(action)
            self.prefix += '<|act|>' + act_str + '<|act|>'
            self.t += 1

            buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
            buffer.seek(0)
            _image = Image.open(buffer)
            _image.save('temp.png')
            self.img_list.append(_image)
            self.prefix += "<image>"
            # else:
            #     for se in smooth_events:
            #         _image = Image.fromarray(np.uint8(se.frame))
            #         self.img_list.append(_image)
            #         self.prefix += ' <image>'
            
            # update transition reward
            _ = env.get_transition_reward(last_event)

        # Done with expert actions
        finished = env.get_subgoal_idx()

        if len(self.expert_init_actions) > 0:
            curr_high_idx += 1

        assert curr_high_idx == eval_idx
        
        self.prefix += f" Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][curr_high_idx])}"

        if curr_high_idx == 0:
            buffer = io.BytesIO( base64.b64decode(last_event['frame_bytes']))
            buffer.seek(0)
            curr_image = Image.open(buffer)
            curr_image.save(f"image_220.png")
            self.img_list.append(curr_image)
            self.prefix += "<image><|act|>"
        else:
            self.prefix += "<|act|>"
        
        return True

    def interact_with_env(self, env, action, eval_idx):
        smooth_events = None
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
                #last_event, smooth_events = env.step(dict(action=action, forceAction=True), smooth_nav=True)
                last_event = env.step(dict(action=action, forceAction=True))

            t_success = last_event['lastActionSuccess']
        except:
            t_success = False

        if not t_success:
            logger.info(f"FAIL -- action: {action}")
            return t_success, subgoal_success

        self.prefix += action + '<|act|>'
        self.prefix += '<image>'

        buffer = io.BytesIO(base64.b64decode(last_event['frame_bytes']))
        buffer.seek(0)
        _image = Image.open(buffer)
        self.img_list.append(_image)

        # next time-step
        # t_done = self.goal_idx >= self.num_subgoals or self.step_num >= self.max_episode_length
        t_reward, t_done, sg_done = env.get_transition_reward(last_event) # type: (float, bool)

        logger.info(f"t: {self.t}, t_done: {t_done}, sg_done: {sg_done} || Pred: {action}, success: {t_success}, Finished: {env.get_subgoal_idx()}")

        # update subgoals
        finished = env.get_subgoal_idx() # get_subgoal_idx returns self.task.finished
        # curr_subgoal_idx = finished + 1
        if finished == eval_idx:
            subgoal_success = True
            return t_success, subgoal_success
            
        if self.t > (self.gt_n_step * 3):
            logger.info(f"fail due to the time step limit -- t: {self.t} > {(self.gt_n_step * 3)} (limit)")
            return t_success, subgoal_success

        # for the next action prediction
        self.t += 1
        self.prefix += '<|act|>'

        return t_success, subgoal_success

    def process_input(self, processor):
        #return t_success, t_done, self.prefix, self.img_list
        batch = processor(images=self.img_list, text=self.prefix, padding=True, return_tensors="pt").to("cuda", torch.bfloat16)
        #batch = processor(images=self.img_list, text=prompt, padding=True, return_tensors="pt").to("cuda")

        logger.info(f"batch.input_ids {batch.input_ids.shape} {batch.input_ids.dtype}")
        logger.info(f"batch.pixel_values {batch.pixel_values.shape} {batch.pixel_values.dtype}")
        logger.info(f"[Prompt] {self.prefix}")
        
        return batch.input_ids, batch.pixel_values

    def update_metrics(self, traj_data, eval_idx, subgoal_success, test_id):
        # metrics
        pl = float(self.t - len(self.expert_init_actions)) + 1 # +1 for last action
        expert_pl = len([ll for ll in traj_data['plan']['low_actions'] if ll['high_idx'] == eval_idx])

        s_spl = (1 if subgoal_success else 0) * min(1., expert_pl / (pl + sys.float_info.epsilon))
        plw_s_spl = s_spl * expert_pl

        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        log_entry = {
                    'test_id': test_id,
                    'subgoal_idx': int(eval_idx),
                    'subgoal_type': subgoal_action,
                    'trial': traj_data['task_id'],
                    'type': traj_data['task_type'],
                    #'subgoal_instr': subgoal_instr,
                    'subgoal_success_spl': float(s_spl),
                    'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
                    'subgoal_path_len_weight': float(expert_pl),}
                    #'reward': float(reward)}

        self.all_log[test_id].append(log_entry)
        if subgoal_success:
            self.successes[subgoal_action].append(log_entry)
        else:
            self.failures[subgoal_action].append(log_entry)

        # save results
        print("-------------")
        subgoals_to_evaluate = list(self.successes.keys())
        subgoals_to_evaluate.sort()
        total_num_eval = 0
        total_num_success = 0
        # if fail_from_len_limit:
        #     results['fail_of_len_limit'] += 1

        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(self.successes[sg]), len(self.failures[sg])
            num_evals = len(self.successes[sg]) + len(self.failures[sg])
            total_num_eval += num_evals
            total_num_success += num_successes
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum([entry['subgoal_path_len_weight'] for entry in self.successes[sg]]) + \
                                        sum([entry['subgoal_path_len_weight'] for entry in self.failures[sg]])
                sr_plw = float(sum([entry['subgoal_path_len_weighted_success_spl'] for entry in self.successes[sg]]) +
                                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in self.failures[sg]])) / total_path_len_weight

                self.results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw,
                }

                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        
        print("%s ==========" % sg)

        self.stat = {
            "test_id": test_id,
            "subgoal_idx": int(eval_idx),
            "total_num_eval": total_num_eval,
            "total_num_success": total_num_success,
        }

        print(f"total # evals: {total_num_eval}")
        if total_num_eval > 0:
            #print(f"Failure due to length limit: {results['fail_of_len_limit']}/{total_num_eval}")
            print(f"total: ({total_num_success/total_num_eval}) {total_num_success}/{total_num_eval}")
            self.stat["total_sr"] = total_num_success/total_num_eval
        print("------------")

        return self.stat

    def load_state_dict(self, state_dict):
        self.successes = state_dict['successes']
        self.failures = state_dict['failures']
        self.all_log = state_dict['all_log']
        self.results = state_dict['results']
        self.stat = state_dict['stat']

    def state_dict(self):
        return {'successes': self.successes,
                'failures': self.failures,
                'all_log': self.all_log,
                'results': self.results,
                "stat": self.stat}

    def sync_metrics_s3(self, aws_s3_path, test_id):
        try:
            # Run aws s3 copy
            with open("stat.json", "w") as f:
                f.write(json.dumps(self.stat, indent=4))

            if aws_s3_path[-1] == "/":
                aws_s3_path = aws_s3_path[:-1]

            sync_command = f"aws s3 cp stat.json {aws_s3_path}/test_id-{test_id}.json"
            subprocess.run(
                sync_command,
                shell=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info(f"Started S3 copy for eval results at test_id {test_id}")
        except Exception as e:
            logger.error(f"Error starting S3 copy: {e}")

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        # self.successes, self.failures = self.manager.dict(), self.manager.dict()
        # self.results = self.manager.dict()
        self.successes, self.failures = defaultdict(list), defaultdict(list)
        self.all_log = defaultdict(list)
        self.results = {}

        # results
        for sg in self.ALL_SUBGOALS:
            self.results[sg] = {
                    'sr': 0.,
                    'successes': 0.,
                    'evals': 0.,
                    'sr_plw': 0.,
                    #"fail_of_len_limit": 0,
            }

    # def save_results(self):
    #     results = {'successes': dict(self.successes),
    #                'failures': dict(self.failures),
    #                'results': dict(self.results)}

    #     save_path = os.path.dirname(self.args.model_path)
    #     save_path = os.path.join(save_path, 'subgoal_results_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
    #     with open(save_path, 'w') as r:
    #         json.dump(results, r, indent=4, sort_keys=True)