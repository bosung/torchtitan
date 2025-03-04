import os
from re import A
import sys
import json
import numpy as np
from PIL import Image
from datetime import datetime
import time
import random
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

def find_visible_and_property(env, objname, property_name):
    for obj in env.last_event.metadata['objects']:
        if objname.lower() in obj['name'].lower() and obj['visible'] and obj[property_name]:
            print(obj['name'], obj['objectId'])
            return obj['objectId']
    return None

def visible_and_pickupable(env, objname):
    return find_visible_and_property(env, objname, 'pickupable')

def visible_and_receptacle(env, objname):
    return find_visible_and_property(env, objname, 'receptacle')

def visible_and_openable(env, objname):
    return find_visible_and_property(env, objname, 'openable')

def visible_and_sliceable(env, objname):
    return find_visible_and_property(env, objname, 'sliceable')

def visible_and_toggleable(env, objname):
    return find_visible_and_property(env, objname, 'toggleable')

def visible_and_isOpen(env, objname): # closeable
    return find_visible_and_property(env, objname, 'isOpen')


def action_with_gt(action, env):
    pass


def post_processing_action(action, env, objname=None):
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
            return "PutObject", visible_and_receptacle(env, receptacle_obj)
        elif action.startswith('ToggleObject'):
            objname = action.split()[-1].strip()
            return "ToggleObject", visible_and_toggleable(env, objname)

        for action_prefix, func in actions_map.items():
            if action.startswith('PutObject'):
                receptacle_obj = action.split()[-1].strip()
                return action_prefix, func(env, receptacle_obj)
            elif action.startswith(action_prefix):
                objname = action.split(action_prefix)[-1].strip()
                return action_prefix, func(env, objname)
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

    def get_subgoal_idx(self, traj):
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
        #breakpoint()
        # update objectName bc obj name are different from ai2thor versions
        objname_dict = {obj['name'].split("_")[0]: obj['name'] for obj in last_event.metadata['objects']}
        valid_object_poses = []
        for obj_pose in object_poses:
            name = obj_pose['objectName'].split("_")[0]
            if name in objname_dict:
                if objname_dict[name] != obj_pose['objectName']:
                    obj_pose.update({"objectName": objname_dict[name]})
                valid_object_poses.append(obj_pose)
        
        last_event = env.restore_scene(valid_object_poses, object_toggles, dirty_and_empty)
        if not last_event.metadata['lastActionSuccess']:
            raise ValueError("Due to the ai2thor version conflicts.")
        
        # initialize to start position
        # updated for ai2thor 5.0.0 (previously 2.1.0)
        init_act = dict(
            action=traj['scene']['init_action']['action'],
            horizon=traj['scene']['init_action']['horizon'],
            position={
                'x': traj['scene']['init_action']['x'],
                'y': traj['scene']['init_action']['y'],
                'z': traj['scene']['init_action']['z'],
            },
            rotation={'x': 0, 'y': traj['scene']['init_action']['rotation'], 'z': 0},
            standing=True
        )
        env.step(init_act)

        # print goal instr
        print(f"Task: {traj['task_type']} {traj['pddl_params']}")
        
        # setup task for reward
        env.set_task(traj, reward_type=reward_type)

    """
    @classmethod
    def run(cls, model, processor, eval_dataloader, args, successes, failures, results):
        '''
        evaluation loop
        '''
        # start THOR
        env = ThorEnv()

        # make subgoals list
        subgoals_to_evaluate = cls.ALL_SUBGOALS if args.subgoals.lower() == "all" else args.subgoals.split(',')
        subgoals_to_evaluate = [sg for sg in subgoals_to_evaluate if sg in cls.ALL_SUBGOALS]

        # create empty stats per subgoal
        for sg in subgoals_to_evaluate:
            successes[sg] = list()
            failures[sg] = list()

        #for i in range(0, len(eval_dataset)):
        for batch in eval_dataloader:
            task = batch['task'][0]
            traj = batch['traj'][0]
            r_idx = task['repeat_idx']
            subgoal_idxs = [sg['high_idx'] for sg in traj['plan']['high_pddl'] if sg['discrete_action']['action'] in subgoals_to_evaluate]

            #print(task)
            for eval_idx in subgoal_idxs:
                cls.evaluate(env, model, processor, eval_idx, r_idx, traj, args, successes, failures, results)

        # stop THOR
        env.stop()
    """

    def simulate_with_expert(self, env, traj_data, processor, eval_idx, teacher_forcing=True, max_steps=2000):
        # expert demonstration to reach eval_idx-1
        #expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]
        expert_init_actions = [a['api_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]

        # subgoal info
        r_idx = 0   # TODO remove turk_annotations's
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        #subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]
        #subgoal_instrs = traj_data['turk_annotations']['anns'][r_idx]['high_descs']

        # task goal info
        task_desc = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # print subgoal info
        print(f"Evaluating: {traj_data['task_id']}\nTask desc: {task_desc}")
        print(f"Subgoal[{eval_idx}] {subgoal_action}")

        # setup scene
        reward_type = 'dense'
        self.setup_scene(env, traj_data, reward_type=reward_type)

        done, subgoal_success = False, False
        t, fails, reward = 0, 0, 0
        prev_subgoal_idx, curr_subgoal_idx = -1, 0
        prev_high_idx = -1
        fail_from_len_limit = False
        
        # for model inputs
        self.sequence, self.token_type_ids = [], []
        self.img_list = []

        self.prefix = ""
        main_goal_str = "Your main goal: "
        if 'turk_annotations' in traj_data:
            main_goal_str += traj_data['turk_annotations']['anns'][0]['task_desc']
        self.prefix = main_goal_str
        self.t = 0

        curr_high_idx = 0
        # Do expert action here:
        for t, action in enumerate(expert_init_actions):
            curr_high_idx = traj_data['plan']['low_actions'][t]['high_idx']
            if  curr_high_idx > prev_high_idx:
                #self.sequence.append(subgoal_instrs[curr_high_idx])
                self.prefix += f" Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][curr_high_idx])}"
                #print(f"\t-----> New subgoal [{curr_high_idx}] is set: {subgoal_instrs[curr_high_idx]}")
                #token_type_ids.append( 'lang' )
                prev_high_idx = curr_high_idx

                # add image for the last state?
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                self.img_list.append(curr_image)
                self.prefix += "<image>"

            # execute expert action
            #_action = action['action'].split("_")[0]
            if 'Object' in action['action']: # expert 
                if 'PutObject' in action['action']:
                    breakpoint()
                last_event = env.step(action)

            else:
                last_event, smooth_events = env.step(dict(action=action['action'], forceAction=True), smooth_nav=True)

            if not last_event.metadata['lastActionSuccess']:
                logger.info("ERROR - expert initialization failed")
                breakpoint()
                return None, None, False

            act_str = serialize_action(action)
            self.prefix += '<|act|>' + act_str + '<|act|>'
            self.t += 1

            if 'Object' in action['action']:
                _image = Image.fromarray(np.uint8(env.last_event.frame))
                self.img_list.append(_image)
                self.prefix += "<image>"
            else:
                for se in smooth_events:
                    _image = Image.fromarray(np.uint8(se.frame))
                    self.img_list.append(_image)
                    self.prefix += ' <image>'

        # Done with expert actions
        finished = env.get_subgoal_idx()

        #assert finished == (eval_idx - 1)
        # TODO here
        #prev_high_idx = finished
        
        #self.sequence.append(subgoal_instrs[eval_idx])
        #print(f"\t-----> New subgoal [{eval_idx}] is set: {subgoal_instrs[eval_idx]}")
        #self.token_type_ids.append( 'lang' )
        # TODO
        if len(expert_init_actions) > 0:
            curr_high_idx += 1

        if curr_high_idx != eval_idx:
            breakpoint()
        assert curr_high_idx == eval_idx
        
        self.prefix += f" Plan: {get_templated_high_pddl_desc(traj_data['plan']['high_pddl'][curr_high_idx])}"

        curr_image = Image.fromarray(np.uint8(env.last_event.frame))
        self.img_list.append(curr_image)
        self.prefix += "<image>"
        
        prompt = self.prefix + "<|act|>"
        
        return prompt, self.img_list, True

    def interact_with_env(self, env, action, eval_idx):
        smooth_events = None
        try:
            # convert act to api_action
            if 'Object' in action:
                _action, obj_id = post_processing_action(action, env)
                if 'PutObject' in action and obj_id:
                    inventory_object_id = env.last_event.metadata['inventoryObjects'][0]['objectId']
                    put_action = dict(action="PutObject",
                                objectId=inventory_object_id,
                                receptacleObjectId=obj_id,
                                forceAction=True,
                                placeStationary=True)
                    last_event = env.step(put_action)
                elif obj_id:
                    last_event = env.step(dict(action=_action, objectId=obj_id))
            else:
                last_event, smooth_events = env.step(dict(action=action, forceAction=true), smooth_nav=True)

            t_success = last_event.metadata['lastActionSuccess']
        except:
            t_success = False

        if not t_success:
            breakpoint()
            logger.info(f"FAIL -- action: {env.last_event.metadata['lastAction']} error: {env.last_event.metadata['errorMessage']}")
            return False, False, None, None

        self.prefix += '<|act|>' + action + '<|act|>'

        if smooth_events:
            for se in smooth_events:
                self.prefix += ' <image>'
                _image = Image.fromarray(np.uint8(se.frame))
                self.img_list.append(_image)
        else:
            self.prefix += ' <image>'
            _image = Image.fromarray(np.uint8(last_event.frame))
            self.img_list.append(_image)

        # next time-step
        # t_done = self.goal_idx >= self.num_subgoals or self.step_num >= self.max_episode_length
        t_reward, t_done = env.get_transition_reward() # type: (float, bool)
        
        # debug
        # print(f"t: {t}, Expert: {traj_data['plan']['low_actions'][t]['api_action']['action']} | {traj_data['plan']['low_actions'][t]['api_action']}")
        #gt_action = traj_data['plan']['low_actions'][t]['api_action']['action'] if t < len(traj_data['plan']['low_actions']) else None
        logger.info(f"t: {self.t}, t_done: {t_done}, Pred: {action}, success: {t_success}, Finished: {env.get_subgoal_idx()}")

        # update subgoals
        finished = env.get_subgoal_idx() # get_subgoal_idx returns self.task.finished
        # curr_subgoal_idx = finished + 1
        if finished == eval_idx:
            subgoal_success = True
            return t_success, subgoal_success, self.prefix, self.img_list
            
        # increment time index
        self.t += 1

        self.prefix += '<|act|>'

        return t_success, t_done, self.prefix, self.img_list
        # ==========================================================

    def metrics():
        # metrics
        pl = float(t - len(expert_init_actions)) + 1 # +1 for last action
        expert_pl = len([ll for ll in traj_data['plan']['low_actions'] if ll['high_idx'] == eval_idx])

        s_spl = (1 if subgoal_success else 0) * min(1., expert_pl / (pl + sys.float_info.epsilon))
        plw_s_spl = s_spl * expert_pl

        # log success/fails
        #lock.acquire()

        # results
        for sg in cls.ALL_SUBGOALS:
            results[sg] = {
                    'sr': 0.,
                    'successes': 0.,
                    'evals': 0.,
                    'sr_plw': 0.
            }

        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(r_idx),
                     'subgoal_idx': int(eval_idx),
                     'subgoal_type': subgoal_action,
                     'subgoal_instr': subgoal_instr,
                     'subgoal_success_spl': float(s_spl),
                     'subgoal_path_len_weighted_success_spl': float(plw_s_spl),
                     'subgoal_path_len_weight': float(expert_pl),
                     'reward': float(reward)}
        if subgoal_success:
            sg_successes = successes[subgoal_action]
            sg_successes.append(log_entry)
            successes[subgoal_action] = sg_successes
        else:
            sg_failures = failures[subgoal_action]
            sg_failures.append(log_entry)
            failures[subgoal_action] = sg_failures

        # save results
        print("-------------")
        subgoals_to_evaluate = list(successes.keys())
        subgoals_to_evaluate.sort()
        total_num_eval = 0
        total_num_success = 0
        if fail_from_len_limit:
            results['fail_of_len_limit'] += 1

        for sg in subgoals_to_evaluate:
            num_successes, num_failures = len(successes[sg]), len(failures[sg])
            num_evals = len(successes[sg]) + len(failures[sg])
            total_num_eval += num_evals
            total_num_success += num_successes
            if num_evals > 0:
                sr = float(num_successes) / num_evals
                total_path_len_weight = sum([entry['subgoal_path_len_weight'] for entry in successes[sg]]) + \
                                        sum([entry['subgoal_path_len_weight'] for entry in failures[sg]])
                sr_plw = float(sum([entry['subgoal_path_len_weighted_success_spl'] for entry in successes[sg]]) +
                                    sum([entry['subgoal_path_len_weighted_success_spl'] for entry in failures[sg]])) / total_path_len_weight

                results[sg] = {
                    'sr': sr,
                    'successes': num_successes,
                    'evals': num_evals,
                    'sr_plw': sr_plw,
                }

                print("%s ==========" % sg)
                print("SR: %d/%d = %.3f" % (num_successes, num_evals, sr))
                print("PLW SR: %.3f" % (sr_plw))
        
        print("%s ==========" % sg)
        print(f"total # evals: {total_num_eval}")
        if total_num_eval > 0:
            print(f"Failure due to length limit: {results['fail_of_len_limit']}/{total_num_eval}")
            print(f"total: ({total_num_success/total_num_eval}) {total_num_success}/{total_num_eval}")
            
        print("------------")
        # lock.release()

    def create_stats(self):
        '''
        storage for success, failure, and results info
        '''
        # self.successes, self.failures = self.manager.dict(), self.manager.dict()
        # self.results = self.manager.dict()
        self.successes, self.failures = {}, {}
        self.results = {"fail_of_len_limit": 0,}

    def save_results(self):
        results = {'successes': dict(self.successes),
                   'failures': dict(self.failures),
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'subgoal_results_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)