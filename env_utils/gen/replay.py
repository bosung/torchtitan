import json
import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

from utils.replay_json import replay_json, replay_agent
from env.thor_env import ThorEnv
from utils.game_util import get_objects_of_type
import time
from scripts.generate_long_trajectories import navigate_to, setup_data_dict
from agents.deterministic_planner_agent import DeterministicPlannerAgent
from game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
import random
import constants

json_file = "/Users/bosungkim/workspace/data/long_traj/floorplan1/floorplan1_13_398_1737151572.json"

with open(json_file) as f:
    traj_data = json.load(f)

env = ThorEnv()

scene_num = traj_data['scene']['scene_num']
object_poses = traj_data['scene']['object_poses']
#dirty_and_empty = traj_data['scene']['dirty_and_empty']
#object_toggles = traj_data['scene']['object_toggles']
dirty_and_empty = False
object_toggles = []

scene_name = 'FloorPlan%d' % scene_num
env.reset(scene_name)
#env.restore_scene(object_poses, object_toggles, dirty_and_empty)
env.step((dict(action='SetObjectPoses', objectPoses=object_poses)))

# initialize
event = env.step(dict(traj_data['scene']['init_action']))

steps_taken = 0
data = []

for t, ll_action in enumerate(traj_data['plan']['low_actions']):
    hl_action_idx, traj_api_cmd = ll_action['high_idx'], ll_action['api_action']

    env.step(traj_api_cmd)
    # if traj_api_cmd['action'] != "MoveAhead":
    #     time.sleep(1)
    # if traj_api_cmd['action'] == "PickupObject":
    #     #breakpoint()
    #     print("Pickup: ", env.last_event.metadata['inventoryObjects'])

    print(t, env.last_event.metadata['lastActionSuccess'], traj_api_cmd, env.last_event.metadata['errorMessage'])
    if not env.last_event.metadata['lastActionSuccess']:
        breakpoint()

    objs = env.last_event.metadata['objects']
    visible = list(set([o['name'].split("_")[0] for o in objs if o['visible']]))
    pickupable = list(set([o['name'].split("_")[0] for o in objs if o['pickupable']]))
    isOpen = list(set([o['name'].split("_")[0] for o in objs if o['isOpen']]))

    data.append(dict(
        t=t,
        action=traj_api_cmd['action'],
        visible=visible,
        pickupable=pickupable,
        isOpen=isOpen
    ))

# import pandas as pd

# df = pd.DataFrame(data)

# breakpoint()

# steps_taken, last_event = replay_json(env, traj_data)