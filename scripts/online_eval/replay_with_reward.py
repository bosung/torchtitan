from datasets import load_dataset
import os
import json
import subprocess

import wandb

def save_json(filename, data, indent=4):
    with open(filename, "w") as f:
        json.dump(data, f, indent=indent)


def save_s3(output_dir, s3_path): # output_dir: outputs/checkpoints/step-xxxx
    
    # Push checkpoints from local_rank 0
    # if get_local_rank() == 0:
    #     try:
            # Run aws s3 sync in background using nohup
    sync_command = f"aws s3 sync {output_dir} {s3_path}"
    subprocess.run(
        sync_command,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    #logger.info(f"Started background S3 sync for checkpoint at step {step}")
    #except Exception as e:
    #logger.error(f"Error starting S3 sync: {e}")

    # Ensure upload is complete before proceeding
    # dist.barrier()


api = wandb.Api()
run_id = "expert" # run_id should be model name or "expert"

try:
    run = api.run(f"lajollafolks/long-traj-eval/{run_id}")
    init_run = False
except: # init run
    init_run = True

wandb.init(project="long-traj-eval", id=run_id, resume="allow")


from ai2thor_client import ThorEnv
from eval_subgoals import EvalSubgoals
eval_subgoals = EvalSubgoals()

env = ThorEnv()

# Download and load the dataset
#alfred_dataset = load_dataset("bosungkim/long_alfred", split="train", features=features)
# from huggingface_hub import snapshot_download
# >>> snapshot_download(repo_id="bosungkim/long_alfred", repo_type="dataset", local_dir="data/long_traj")

# Directory containing downloaded JSON files
data_dir = "/home/bkim/torchtitan/scripts/data/long_traj"

replay_success_file = 'online_eval/replay_success.json'
replay_success = json.loads(open(replay_success_file).read())

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
        if traj_id in replay_success:
            continue

        print("replay_success")
        reward_log_file = f"online_eval/logs/{traj_id}.json"
        if os.path.exists(reward_log_file):
            reward_log = json.load(open(reward_log_file))
        else:
            reward_log = {}
        
        if "expert" in reward_log:
            log_data = reward_log["expert"]['x_time']
            last_step = log_data[-1][0]
        else:
            log_data = []
            last_step = 0

        file_path = os.path.join(floorplan_path, file)
        success = True
        with open(file_path, 'r') as f:
            traj_data = json.load(f)
        
        last_event = eval_subgoals.setup_scene(env, traj_data, reward_type='dense')

        total_reward = 0
        global_t = 0
        for sub_task, sub_traj in zip(traj_data['sub_tasks'], traj_data['sub_trajs']):
            num_subgoals = sub_traj['high_pddl_idx'][1] - sub_traj['high_pddl_idx'][0]
            low_start, low_end = sub_traj['low_pddl_idx']

            env.set_task(traj_data, last_event,
                        sub_traj_idx=sub_traj['sub_traj_idx'],
                        task_info=sub_task['task_info'],
                        task_type=sub_task['task_info']['goal'],
                        num_subgoals=num_subgoals,
                        reward_type='dense')

            expert_actions = [a['api_action'] for a in traj_data['plan']['low_actions'][low_start:low_end]]
            for t, action in enumerate(expert_actions):
                if action['action'] in ['ToggleObjectOn', 'ToggleObjectOff', 'PutObject']:
                    action['forceAction'] = True
                last_event = env.step(action)
                if last_event['lastActionSuccess']:
                    reward, done, sg_done = env.get_transition_reward(last_event)
                    total_reward += reward
                    print(f"global_t: {global_t}, action: {action['action']}, reward: {reward}, total_reward: {total_reward}")

                    if global_t > last_step: # do generation or something
                        log_data.append([int(global_t), total_reward])
                    global_t += 1
                    
                elif not last_event['lastActionSuccess'] and (last_event['lastAction'] in ["LookDown", "LookUp"]):
                    pass
                else:
                    print(f"ERROR - expert initialization failed at {t} (action: {action})")
                    print(f"ERROR - lastAction: {last_event['lastAction']}, err: {last_event['errorMessage']}")
                    success = False
                    break

            if not success:
                break
        # end of expert actions
        # log
        table = wandb.Table(data=log_data, columns=["t", "Reward"])
        wandb.log({f"{traj_id}/expert": wandb.plot.line(table, "t", "Reward",title="Reward")})
        
        if "expert" not in reward_log:
            reward_log["expert"] = {}
        reward_log["expert"]['x_time'] = log_data

        save_json(reward_log_file, reward_log)

        log_dir = "online_eval/logs"
        s3_path = "s3://bosung-alfred/eval_logs/"
        save_s3(log_dir, s3_path)

                    