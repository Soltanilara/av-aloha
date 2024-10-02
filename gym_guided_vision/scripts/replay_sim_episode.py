import os
import h5py
import argparse
import time
import glob
from tqdm import tqdm

import gymnasium as gym
import gym_guided_vision

def save_episode(data_dict, save_path):

    camera_names = [key.split('/')[-1] for key in data_dict.keys() if 'images' in key]
    max_timesteps = len(data_dict['/observations/qpos'])

    # HDF5
    try:
        if len(camera_names) > 0:
            image_shapes = {cam_name: data_dict[f'/observations/images/{cam_name}'][0].shape for cam_name in camera_names}
        qpos_len = len(data_dict['/observations/qpos'][0])
        qvel_len = len(data_dict['/observations/qvel'][0])
        action_len = len(data_dict['/action'][0])
    except IndexError:
        print('Empty episode, skipping...')
        return False
    t0 = time.time()
    with h5py.File(save_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        if len(camera_names) > 0:
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, *image_shapes[cam_name]), dtype='uint8',
                                            chunks=(1, *image_shapes[cam_name]), )
        # compression='gzip',compression_opts=2,)
        # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
        data_qpos = obs.create_dataset('qpos', (max_timesteps, qpos_len))
        data_qvel = obs.create_dataset('qvel', (max_timesteps, qvel_len))
        data_action = root.create_dataset('action', (max_timesteps, action_len))

        for name, array in data_dict.items():
            root[name][...] = array
    # print(f'Saving: {time.time() - t0:.1f} secs\n')
    return True


def replay_episode(env, dataset_path, save_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()


    ts, info = env.reset()

    with h5py.File(dataset_path, 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        all_qpos = root['/observations/all_qpos'][()]
        action = root['/action'][()]

    if env.unwrapped.num_arms == 2:
        data_dict = {
            '/observations/qpos': qpos[:,:14],
            '/observations/qvel': qvel[:,:14],
            '/action': action[:,:14],
        }
    else:
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
    for cam_name in ts['pixels']:
        data_dict[f'/observations/images/{cam_name}'] = []
    
    for qpos in all_qpos:
        step_start = time.time()

        # Apply the action
        env.unwrapped.set_qpos(qpos)
        ts = env.unwrapped.get_obs()

        # Save the images
        for cam_name in ts['pixels']:
            data_dict[f'/observations/images/{cam_name}'].append(ts['pixels'][cam_name])

        # print(f'Step time: {time.time() - step_start:.2f} secs')

    save_episode(data_dict, save_path)


def main(args):
    dataset_dir = args["dataset_dir"]
    episode_idx = args["episode_idx"]    
    env_name = args["env"]

    env = gym.make(env_name)

    if episode_idx is None:
        dataset_paths = glob.glob(os.path.join(dataset_dir, f'episode_*.hdf5'))
        dataset_paths.sort()
    else:
        dataset_paths = glob.glob(os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5'))


    for dataset_path in tqdm(dataset_paths):
    
        dataset_name = dataset_path.split('/')[-1]
        save_path = os.path.join(
            dataset_dir, env_name.split('/')[-1], dataset_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        replay_episode(env, dataset_path, save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', action='store', type=str, help='Task name.', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, help='Path to the dataset directory.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)

    main(vars(parser.parse_args()))


