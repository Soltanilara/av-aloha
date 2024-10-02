import sys
sys.path.append('../')

import os
import h5py
import argparse
from sim_env import make_sim_env
from constants import SIM_DT, SIM_TASK_CONFIGS
import time
import numpy as np
import glob

def main(args):
    dataset_dir = SIM_TASK_CONFIGS[args["task_name"]]["dataset_dir"]
    num_episodes = SIM_TASK_CONFIGS[args["task_name"]]["num_episodes"]

    env = make_sim_env(args["task_name"], cameras=[])


    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            all_qpos = root['/observations/all_qpos'][()]


        ts, info = env.reset()

        reward = []
        
        for qpos in all_qpos:

            step_start = time.time()

            # Apply the action
            env.set_qpos(qpos)
            env.render_viewer()

            reward.append(env.get_reward())

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = 0.001 - (time.time() - step_start)
            time.sleep(max(0, time_until_next_step))

        if max(reward) != env.max_reward:
            print(f"Episode {episode_idx} has max reward {max(reward)}")

        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)

    main(vars(parser.parse_args()))


