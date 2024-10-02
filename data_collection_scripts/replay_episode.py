import os
import h5py
from robot_utils import move_grippers
import argparse
from real_env import make_real_env
from robot_utils import sleep
from constants import REAL_DT
import time

import IPython
e = IPython.embed

def main(args):

    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    dataset_name = f'episode_{episode_idx}'

    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        actions = root['/action'][()]


    env = make_real_env(init_node=True)
    env.reset()
    for action in actions:

        step_start = time.time()


        env.step(action, all_joints=True) # print(action) shows all 21 actions..
        
        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))   


    sleep(env.left_bot, env.right_bot, env.middle_bot)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    main(vars(parser.parse_args()))


