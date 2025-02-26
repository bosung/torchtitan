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

        # update objectName bc obj name are different from ai2thor versions
        objname_dict = {obj['name'].split("_")[0]: obj['name'] for obj in last_event.metadata['objects']}
        valid_object_poses = []
        for obj_pose in object_poses:
            name = obj_pose['objectName'].split("_")[0]
            if name in objname_dict and objname_dict[name] != obj_pose['objectName']:
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
        
        # setup scene
        # TODO
        reward_type = 'dense'
        self.setup_scene(env, traj_data, reward_type=reward_type)
        
        # expert demonstration to reach eval_idx-1
        expert_init_actions = [a['discrete_action'] for a in traj_data['plan']['low_actions'] if a['high_idx'] < eval_idx]

        # subgoal info
        r_idx = 0
        subgoal_action = traj_data['plan']['high_pddl'][eval_idx]['discrete_action']['action']
        subgoal_instr = traj_data['turk_annotations']['anns'][r_idx]['high_descs'][eval_idx]
        subgoal_instrs = traj_data['turk_annotations']['anns'][r_idx]['high_descs']

        # task goal info
        task_desc = traj_data['turk_annotations']['anns'][r_idx]['task_desc']

        # print subgoal info
        print(f"Evaluating: {traj_data['task_id']}\nTask desc: {task_desc}")
        print(f"Subgoal[{eval_idx}] {subgoal_action} || Instruction: {subgoal_instr}")

        done, subgoal_success = False, False
        t, fails, reward = 0, 0, 0
        prev_subgoal_idx, curr_subgoal_idx = -1, 0
        prev_high_idx = -1
        fail_from_len_limit = False
        
        # for model inputs
        self.sequence, self.token_type_ids = [], []
        self.img_list = []

        prefix = ""
        main_goal_str = "Your main goal: "
        if 'turk_annotations' in traj_data:
            main_goal_str += traj_data['turk_annotations']['anns'][0]['task_desc']
        prefix = main_goal_str

        # Do expert action here:
        for t, action in enumerate(expert_init_actions):
            curr_high_idx = traj_data['plan']['low_actions'][t]['high_idx']
            if  curr_high_idx > prev_high_idx:
                sequence.append(subgoal_instrs[curr_high_idx])
                print(f"\t-----> New subgoal [{curr_high_idx}] is set: {subgoal_instrs[curr_high_idx]}")
                token_type_ids.append( 'lang' )
                prev_high_idx = curr_high_idx

            # (2) add image
            # extract visual feats
            curr_image = Image.fromarray(np.uint8(env.last_event.frame))
        
            sequence.append(curr_image)
            token_type_ids.append('img')
            self.img_list.append(curr_image)
            prefix += "<image>"

            # compressed_mask = action['args']['mask'] if 'mask' in action['args'] else None
            # mask = env.decompress_mask(compressed_mask) if compressed_mask is not None else None
            
            act_str = serialize_action(action)
            sequence.append(act_str)
            token_type_ids.append('action')
            prefix += act_str

            breakpoint()
            # execute expert action
            last_event = env.step(action['action'], smooth_nav=args.smooth_nav, debug=args.debug)
            if not last_event.metadata['lastActionSuccess']:
                raise ValueError("expert initialization failed")

        # Done with expert actions
        finished = env.get_subgoal_idx()

        #assert finished == (eval_idx - 1)
        prev_high_idx = finished
        curr_high_idx = eval_idx

        self.sequence.append(subgoal_instrs[eval_idx])
        print(f"\t-----> New subgoal [{eval_idx}] is set: {subgoal_instrs[eval_idx]}")
        self.token_type_ids.append( 'lang' )

        # (2) add image
        curr_image = Image.fromarray(np.uint8(env.last_event.frame))

        self.sequence.append(curr_image)
        self.token_type_ids.append('img')
        self.img_list.append(curr_image)
        prefix += "<image>"
        
        prompt = prefix + "<|act|>"

        return prompt, self.img_list

    def interact_with_env(cls, env, action):
        prev_act = action

        sequence.append(action)
        token_type_ids.append('action')

        try:
            # convert act to api_action
            if 'object' in action:
                obj_id = post_processing_action(action, env)
                if obj_id:
                    event, action = env.to_thor_api_exec(action, obj_id)
                    t_success = True if event.metadata['lastActionSuccess'] else False
                    #print(f't: {t}, t_success: {t_success}, action: {action}')
                else:
                    t_success = False
            elif action not in cls.TERMINAL_TOKENS:
                # use predicted action and mask (if provided) to interact with the env
                mask = None
                t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=False, debug=False)
                # output:
                # t_success (bool), <ai2thor.server.Event object at 0x702d0a99d9e8>, "", "", {'action': 'MoveAhead', 'forceAction': True}
        except:
            t_success = False

        if not t_success:
            fails += 1
            if fails >= args.max_fails:
                print("Interact API failed %d times" % (fails) + "; latest error '%s'" % err)
                return False

        # next time-step
        t_reward, t_done = env.get_transition_reward() # type: (float, bool)
        reward += t_reward

        # debug
        # print(f"t: {t}, Expert: {traj_data['plan']['low_actions'][t]['api_action']['action']} | {traj_data['plan']['low_actions'][t]['api_action']}")
        gt_action = traj_data['plan']['low_actions'][t]['api_action']['action'] if t < len(traj_data['plan']['low_actions']) else None
        print(f"t: {t}, Pred: {action} (GT: {gt_action}), success: {t_success}, Finished: {env.get_subgoal_idx()}")

        # update subgoals
        finished = env.get_subgoal_idx() # get_subgoal_idx returns self.task.finished
        # curr_subgoal_idx = finished + 1
        if finished == eval_idx:
            subgoal_success = True
            return False
            
        # terminal tokens predicted
        if action in cls.TERMINAL_TOKENS:
            print("predicted %s" % action)
            return False

        # increment time index
        t += 1

        return True
        # ===============================================

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