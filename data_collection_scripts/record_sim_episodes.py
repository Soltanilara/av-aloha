import numpy as np
from constants import SIM_DT, SIM_TASK_CONFIGS
import asyncio
from webrtc_headset import WebRTCHeadset
from headset_control import HeadsetFullControl as HeadsetControl
from headset_utils import HeadsetFeedback
import argparse
from sim_env import make_sim_env
import time
import os
from tqdm import tqdm
import h5py

def send_popup_message(headset, message, duration=3.0):
    feedback = HeadsetFeedback()
    feedback.info = message
    headset.send_feedback(feedback)
    time.sleep(duration)
    
def init_step(env):
    ts, info = env.reset()
    action = np.concatenate([
        ts['poses']['left'],
        np.array([0.0]),
        ts['poses']['right'],
        np.array([0.0]),
        ts['poses']['middle'],
    ])
    env.step(action)

def reset_env(env, headset):
    print("Resetting the environment...")
     # reset the environment
    feedback = HeadsetFeedback()
    feedback.info = "Resetting the environment..."
    headset.send_feedback(feedback)
    ts, info = env.reset()
    return ts

def run_episode(env, headset, episode_len, episode_idx):
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()
    action = np.zeros(23)

    # wait for user to start the episode
    print("Waiting for user to start the episode...")
    while True:
        start_time = time.time()
        
        ts = env.get_obs()

        headset_data = headset.receive_data()
        if headset_data is not None:
            # get the action and feedback from the headset control
            action, feedback = headset_control.run(
                headset_data, 
                ts['poses']['left'], 
                ts['poses']['right'], 
                ts['poses']['middle']
            )
            # break if the user holds the right button
            if headset_data.r_button_one == True and feedback.head_out_of_sync == False and \
                feedback.left_out_of_sync == False and feedback.right_out_of_sync == False:
                headset_control.start(
                    headset_data, 
                    ts['poses']['middle']
                )
                break

        feedback.info = f"Align and hold A to start the episode {episode_idx}."
        headset.send_feedback(feedback)
        
        # send initial image to headset
        zed_img = ts["images"]["zed_cam"]
        right_img = zed_img[:, zed_img.shape[1]//2:, :]
        left_img = zed_img[:, :zed_img.shape[1]//2, :]
        headset.send_images(left_img, right_img)

        time_until_next_step = SIM_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))

    # run the episode
    print(f"Starting episode {episode_idx}...")
    episode_replay = [ts]
    action_replay = []
    for step_idx in tqdm(range(episode_len)):
        step_start = time.time()

        # Take a step in the environment using the chosen action
        ts, reward, terminated, truncated, info = env.step(action)
        episode_replay.append(ts)
        action_replay.append(action)  

        # Check if the episode is terminated 
        if terminated:
            send_popup_message(headset, f"Episode {episode_idx} terminated. {info}", 3.0)
            return [], [], False

        # Receive data from the headset
        headset_data = headset.receive_data()
        if headset_data is not None:
            action, feedback = headset_control.run(
                headset_data, 
                ts['poses']['left'], 
                ts['poses']['right'], 
                ts['poses']['middle']
            )
            if headset_data.r_button_one == False:
                send_popup_message(headset, f"Episode {episode_idx} terminated by user. {info}", 3.0)
                return [], [], False  

        feedback.info = f"Episode {episode_idx}, Timestep: {str(step_idx).zfill(len(str(episode_len)))}/{episode_len}\n{info}"
        headset.send_feedback(feedback) 
        
        # send initial image to headset
        zed_img = ts["images"]["zed_cam"]
        right_img = zed_img[:, zed_img.shape[1]//2:, :]
        left_img = zed_img[:, :zed_img.shape[1]//2, :]
        headset.send_images(left_img, right_img)

        print(f"Step time: {time.time() - step_start}s")

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))        
    
    return episode_replay, action_replay, True


def confirm_episode(headset, episode_idx):
    # wait for user to redo or do next episode
    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()

    print("Waiting for user to redo or do next episode...")
    while True:
        start_time = time.time()

        headset_data = headset.receive_data()
        if headset_data is not None:
            if headset_data.l_button_one == True:
                return True
            elif headset_data.l_button_two == True:
                return False      
              
        feedback.info = f"Episode {episode_idx} completed. Press X to start next episode or Y to redo."
        headset.send_feedback(feedback)
                
        time_until_next_step = SIM_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))


def save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx):
    feedback = HeadsetFeedback()
    feedback.info = f"Saving episode {episode_idx}..."
    headset.send_feedback(feedback)

    # convert data to dataset format
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/all_qpos': [],
        '/action': [],
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []

    # len(episode_replay) i.e. time steps: max_timesteps
    max_timesteps = len(episode_replay)
    while episode_replay:
        ts = episode_replay.pop(0)
        data_dict['/observations/qpos'].append(ts['joints']['position'])
        data_dict['/observations/qvel'].append(ts['joints']['velocity'])
        data_dict['/observations/all_qpos'].append(ts['qpos'])
        data_dict['/action'].append(ts['control'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

    # HDF5
    try:
        if len(camera_names) > 0:
            image_shapes = {cam_name: data_dict[f'/observations/images/{cam_name}'][0].shape for cam_name in camera_names}
        qpos_len = len(data_dict['/observations/qpos'][0])
        qvel_len = len(data_dict['/observations/qvel'][0])
        all_qpos_len = len(data_dict['/observations/all_qpos'][0])
        action_len = len(data_dict['/action'][0])
    except IndexError:
        print('Empty episode, skipping...')
        return False
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
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
        data_all_qpos = obs.create_dataset('all_qpos', (max_timesteps, all_qpos_len))
        data_qvel = obs.create_dataset('qvel', (max_timesteps, qvel_len))
        data_action = root.create_dataset('action', (max_timesteps, action_len))

        for name, array in data_dict.items():
            root[name][...] = array
    print(f'Saving: {time.time() - t0:.1f} secs\n')
    return True

def collect_data(args):
    dataset_dir = SIM_TASK_CONFIGS[args["task_name"]]["dataset_dir"]
    num_episodes = SIM_TASK_CONFIGS[args["task_name"]]["num_episodes"]
    episode_len = SIM_TASK_CONFIGS[args["task_name"]]["episode_len"]
    camera_names = SIM_TASK_CONFIGS[args["task_name"]]["camera_names"]
    record_video = args["record_video"]
    start_episode_idx = args["episode_idx"]

    #TODO
    record_video = False

    # create the dataset directory if it does not exist
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # setup the headset control
    headset = WebRTCHeadset()
    headset.run_in_thread()
    headset_control = HeadsetControl()

    # setup the environment
    if record_video:
        print("Recording video is enabled.")
        env = make_sim_env(args["task_name"], cameras=camera_names)
    else:
        print("Recording video is disabled.")
        env = make_sim_env(args["task_name"], cameras=['zed_cam'])
    
    init_step(env)

    episode_idx = start_episode_idx
    while episode_idx < num_episodes:
        reset_env(env, headset)

        # run the episode
        episode_replay, action_replay, ok = run_episode(env, headset, episode_len, episode_idx)

        if not ok:
            continue

        # confirm if the episode is to be saved
        ok = confirm_episode(headset, episode_idx)

        if not ok:
            continue

        # save the episode
        if record_video:
            ok = save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx)
        else:
            ok = save_episode(headset, episode_replay, action_replay, [], dataset_dir, episode_idx)

        if not ok:
            continue

        episode_idx += 1        


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Headset control")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, required=True)
    parser.add_argument("--record_video", type=bool, default=True)
    args = parser.parse_args()
    args = vars(args)

    try:
        collect_data(args)
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
