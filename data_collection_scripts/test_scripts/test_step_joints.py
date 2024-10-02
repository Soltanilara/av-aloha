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
    episode_idx = args["episode_idx"]

    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
    with h5py.File(dataset_path, 'r') as root:
        all_qpos = root['/observations/all_qpos'][()]
        actions = root['/action'][()]
        qpos = root['/observations/qpos'][()]

    env = make_sim_env(args["task_name"], cameras=[])

    for i in range(10):
        ts, info = env.reset()

        env.set_qpos(all_qpos[0])
        
        for action in actions:

            step_start = time.time()

            # Apply the action
            env.step_joints(action)
            env.render_viewer()

            print(env.get_reward())

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = SIM_DT - (time.time() - step_start)
            time.sleep(max(0, time_until_next_step))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)

    main(vars(parser.parse_args()))


