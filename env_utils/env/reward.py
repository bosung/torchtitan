from env_utils.gen.utils.game_util import get_object

REWARD_MAP = {
    "Generic":
    {
      "success": 2,
      "failure": -0.05,
      "step_penalty": -0.01,
      "goal_reward": 5
    },
    "BaseAction":
    {
      "positive": 1,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "GotoLocationAction":
    {
      "positive": 4,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.5,
      "min_reach_distance": 3
    },
    "PickupObjectAction":
    {
      "positive": 2,
      "negative": -1,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "PutObjectAction":
    {
      "positive": 2,
      "negative": -1,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "OpenObjectAction":
    {
      "positive": 2,
      "negative": -0.05,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "CloseObjectAction":
    {
      "positive": 1,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "ToggleObjectAction":
    {
      "positive": 1,
      "negative": -1,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "SliceObjectAction":
    {
      "positive": 1,
      "negative": -4,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "CleanObjectAction":
    {
      "positive": 2,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "HeatObjectAction":
    {
      "positive": 2,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.1
    },
    "CoolObjectAction":
    {
      "positive": 2,
      "negative": 0,
      "neutral": 0,
      "invalid_action": -0.1
    }
  }

class BaseAction(object):
    '''
    base class for API actions
    '''

    def __init__(self, gt_graph, env, rewards):
        self.gt_graph = gt_graph # for navigation
        self.env = env
        self.rewards = rewards

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        reward, done = self.rewards['neutral'], True
        return reward, done


class GotoLocationAction(BaseAction):
    '''
    MoveAhead, Rotate, Lookup
    '''

    valid_actions = {'MoveAhead', 'RotateLeft', 'RotateRight', 'LookUp', 'LookDown', 'Teleport', 'TeleportFull'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        curr_pose = state['pose_discrete']
        prev_pose = prev_state['pose_discrete']
        tar_pose = tuple([int(i) for i in subgoal['location'].split('|')[1:]])

        prev_actions, _ = self.gt_graph.get_shortest_path(prev_pose, tar_pose)
        curr_actions, _ = self.gt_graph.get_shortest_path(curr_pose, tar_pose)

        prev_distance = len(prev_actions)
        curr_distance = len(curr_actions)
        reward = (prev_distance - curr_distance) * 0.2 # distance reward factor?

        # [DEPRECATED] Old criteria which requires the next subgoal object to be visible
        # Consider navigation a success if we can see the target object in the next step from here.
        # assert len(expert_plan) > goal_idx + 1
        # next_subgoal = expert_plan[goal_idx + 1]['planner_action']
        # next_goal_object = get_object(next_subgoal['objectId'], state.metadata)
        # done = (next_goal_object['visible'] and curr_distance < self.rewards['min_reach_distance'])

        done = curr_distance < self.rewards['min_reach_distance']

        if done:
            reward += self.rewards['positive']

        return reward, done


class PickupObjectAction(BaseAction):
    '''
    PickupObject
    '''

    valid_actions = {'PickupObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        inventory_objects = state['inventoryObjects']
        if len(inventory_objects):
            inv_object_id = state['inventoryObjects'][0]['objectId']
            goal_object_id = subgoal['objectId']

            # doesn't matter which slice you pick up
            def remove_slice_postfix(object_id):
                return object_id.split("Sliced")[0]

            inv_object_id = remove_slice_postfix(inv_object_id)
            goal_object_id = remove_slice_postfix(goal_object_id)

            reward, done = (self.rewards['positive'], True) if inv_object_id == goal_object_id else (self.rewards['negative'], False)
        return reward, done


class PutObjectAction(BaseAction):
    '''
    PutObject
    '''

    valid_actions = {'PutObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_object_id = subgoal['objectId']
        recep_object = get_object(subgoal['receptacleObjectId'], state)
        if recep_object is not None:
            is_target_in_recep = target_object_id in recep_object['receptacleObjectIds']
            reward, done = (self.rewards['positive'], True) if is_target_in_recep else (self.rewards['negative'], False)
        return reward, done


class OpenObjectAction(BaseAction):
    '''
    OpenObject
    '''

    valid_actions = {'OpenObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_recep = get_object(subgoal['objectId'], state)
        if target_recep is not None:
            is_target_open = target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_open else (self.rewards['negative'], False)
        return reward, done


class CloseObjectAction(BaseAction):
    '''
    CloseObject
    '''

    valid_actions = {'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['negative'], False
        target_recep = get_object(subgoal['objectId'], state)
        if target_recep is not None:
            is_target_closed = not target_recep['isOpen']
            reward, done = (self.rewards['positive'], True) if is_target_closed else (self.rewards['negative'], False)
        return reward, done


class ToggleObjectAction(BaseAction):
    '''
    ToggleObjectOn, ToggleObjectOff
    '''

    valid_actions = {'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_toggle = get_object(subgoal['objectId'], state)
        if target_toggle is not None:
            is_target_toggled = target_toggle['isToggled']
            reward, done = (self.rewards['positive'], True) if is_target_toggled else (self.rewards['negative'], False)
        return reward, done


class SliceObjectAction(BaseAction):
    '''
    SliceObject
    '''

    valid_actions = {'SliceObject', 'OpenObject', 'CloseObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        target_object = get_object(subgoal['objectId'], state)
        if target_object is not None:
            is_target_sliced = target_object['isSliced']
            reward, done = (self.rewards['positive'], True) if is_target_sliced else (self.rewards['negative'], False)
        return reward, done


class CleanObjectAction(BaseAction):
    '''
    CleanObject
    '''

    valid_actions = {'PutObject', 'PickupObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        subgoal = expert_plan[goal_idx]['planner_action']
        reward, done = self.rewards['neutral'], False
        clean_object = get_object(subgoal['cleanObjectId'], state)
        if clean_object is not None:
            is_obj_clean = clean_object['objectId'] in self.env.cleaned_objects
            reward, done = (self.rewards['positive'], True) if is_obj_clean else (self.rewards['negative'], False)
        return reward, done


class HeatObjectAction(BaseAction):
    '''
    HeatObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject', 'ToggleObjectOn', 'ToggleObjectOff'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        reward, done = self.rewards['neutral'], False
        next_put_goal_idx = goal_idx+2 # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        if next_put_goal_idx < len(expert_plan):
            heat_object_id = expert_plan[next_put_goal_idx]['planner_action']['objectId']
            heat_object = get_object(heat_object_id, state)
            is_obj_hot = heat_object['objectId'] in self.env.heated_objects
            reward, done = (self.rewards['positive'], True) if is_obj_hot else (self.rewards['negative'], False)
        return reward, done


class CoolObjectAction(BaseAction):
    '''
    CoolObject
    '''

    valid_actions = {'OpenObject', 'CloseObject', 'PickupObject', 'PutObject'}

    def get_reward(self, state, prev_state, expert_plan, goal_idx):
        if state['lastAction'] not in self.valid_actions:
            reward, done = self.rewards['invalid_action'], False
            return reward, done

        reward, done = self.rewards['neutral'], False
        subgoal = expert_plan[goal_idx]['planner_action']
        next_put_goal_idx = goal_idx+2 # (+1) GotoLocation -> (+2) PutObject (get the objectId from the PutObject action)
        if next_put_goal_idx < len(expert_plan):
            cool_object_id = expert_plan[next_put_goal_idx]['planner_action']['objectId']
            cool_object = get_object(cool_object_id, state)
            is_obj_cool = cool_object['objectId'] in self.env.cooled_objects
            
            # TODO(mohit): support dense rewards for all subgoals
            # intermediate reward if object is cooled
            if is_obj_cool and not self.env.cooled_reward:
                self.env.cooled_reward = True
                reward, done = self.rewards['positive'], False

            # intermediate reward for opening fridge after object is cooled
            elif is_obj_cool and state['lastAction']=='OpenObject':
                target_recep = get_object(subgoal['objectId'], state)
                if target_recep is not None and not self.env.reopen_reward:
                    if target_recep['isOpen']:
                        self.env.reopen_reward = True
                        reward, done = self.rewards['positive'], False

            # intermediate reward for picking up cooled object after reopening fridge
            elif is_obj_cool and state['lastAction']=='PickupObject':
                inventory_objects = state['inventoryObjects']
                if len(inventory_objects):
                    inv_object_id = state['inventoryObjects'][0]['objectId']
                    if inv_object_id == cool_object_id:
                        reward, done = self.rewards['positive'], True # Subgoal completed

        return reward, done


def get_action(action_type, gt_graph, env, reward_config):
    action_type_str = action_type + "Action"

    if action_type_str in globals():
        action = globals()[action_type_str]
        return action(gt_graph, env, reward_config[action_type_str])
    else:
        raise Exception("Invalid action_type %s" % action_type_str)
