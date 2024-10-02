import sys
sys.path.append('../')

import gym_guided_vision
from gym_guided_vision.constants import SIM_DT
import gymnasium as gym

import os
import h5py
import argparse
import time
import numpy as np
import glob
from tqdm import tqdm



def main(args):
    episode_path = args["episode_path"]
    env = gym.make(args["env"]) 

    # Using tqdm to track progress over episodes
    with h5py.File(episode_path, 'r') as root:
        all_qpos = root['/observations/all_qpos'][()]


    paused = False

    def key_callback(keycode):
        if chr(keycode) == ' ':
            nonlocal paused
            paused = not paused

    env.unwrapped.create_viewer(key_callback=key_callback)
    
    i = 0
    while True:
        step_start = time.time()

        if not paused:
            # Apply the action
            env.unwrapped.set_qpos(all_qpos[i%len(all_qpos)])
            env.unwrapped.render_viewer()
            i+=1


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_path', action='store', type=str, help='Path to the dataset directory.', required=True)
    parser.add_argument('--env', action='store', type=str, help='Task name.', required=True)

    main(vars(parser.parse_args()))


