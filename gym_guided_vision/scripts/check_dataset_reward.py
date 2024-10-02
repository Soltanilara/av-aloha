import sys
sys.path.append('../')

import gym_guided_vision
import gymnasium as gym

import os
import h5py
import argparse
import time
import numpy as np
import glob
from tqdm import tqdm

def main(args):
    dataset_dir = args["dataset_dir"]
    render = args["render"]

    env = gym.make(args["env"]) 

    not_max_reward_episodes = []

    # get files in the dataset directory
    dataset_paths = glob.glob(os.path.join(dataset_dir, f'episode_*.hdf5'))

    # Using tqdm to track progress over episodes
    episode_idx = 0
    for dataset_path in tqdm(dataset_paths, desc="Processing Episodes"):
        with h5py.File(dataset_path, 'r') as root:
            all_qpos = root['/observations/all_qpos'][()]
            actions = root['/action'][()]

        ts, info = env.reset(options={})

        env.unwrapped.set_qpos(all_qpos[0])

        reward = []
        
        for action in actions:
            step_start = time.time()

            # Apply the action
            env.unwrapped.step_action(action)
            if render: env.unwrapped.render_viewer()

            reward.append(env.unwrapped.get_reward())

            # Rudimentary time keeping, will drift relative to wall clock.
            if render:
                time_until_next_step = 0.001 - (time.time() - step_start)
                time.sleep(max(0, time_until_next_step))

        # Check if the episode did not reach the maximum reward
        if max(reward) != env.unwrapped.max_reward:
            not_max_reward_episodes.append(episode_idx)

        episode_idx += 1

    # Print all episodes that did not reach max reward
    if not_max_reward_episodes:
        print(f"Episodes that did not reach max reward: {not_max_reward_episodes}")
    else:
        print("All episodes reached max reward!")

        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Path to the dataset directory.', required=True)
    parser.add_argument('--env', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--render', action='store_true', help='Render the environment.')

    main(vars(parser.parse_args()))


