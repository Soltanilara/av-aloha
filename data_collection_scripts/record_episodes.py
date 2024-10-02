import numpy as np
from constants import REAL_DT, TASK_CONFIGS
import asyncio
from webrtc_headset import WebRTCHeadset
from headset_control import HeadsetControl
from headset_utils import HeadsetFeedback
import argparse
from real_env import RealEnv, get_master_bots_action, wait_for_user, reset_master_arms, reset_env
import time
import os
from tqdm import tqdm
import h5py
from interbotix_xs_modules.arm import InterbotixManipulatorXS


def run_episode(env: RealEnv, headset: WebRTCHeadset, master_bot_left, master_bot_right, episode_len, episode_idx):

    reset_env(env, headset, master_bot_left, master_bot_right)

    reset_master_arms(headset, master_bot_left, master_bot_right)

    #TODO REMOVE THIS
    ts = env.get_obs()
    import matplotlib.pyplot as plt
    zed_image_left = ts["images"]["zed_cam_left"]
    zed_image_right = ts["images"]["zed_cam_right"]
    print(f"zed image left shape: {zed_image_left.shape}")
    print(f"zed image right shape: {zed_image_right.shape}")
    print("Showing THE ZED LEFT IMAGE")
    import cv2

    # Assuming zed_image_left and zed_image_right are numpy arrays in BGR format (as used by OpenCV)
    # If they are in RGB format, convert them to BGR before saving
    cv2.imwrite('zed_image_left.png', zed_image_left)
    cv2.imwrite('zed_image_right.png', zed_image_right)

    print("ZED left image saved as 'zed_image_left.png'")
    print("ZED right image saved as 'zed_image_right.png'")
    # exit()

    headset_control = wait_for_user(env, headset, master_bot_left, master_bot_right, 
                                    message=f"Episode {episode_idx}, align your head and close both grippers to start.")

    # run the episode
    print(f"Starting episode {episode_idx}...")
    ts = env.get_obs()
    action = np.concatenate([
        ts['control'][:14],
        ts['poses']['middle'],
    ])
    feedback = HeadsetFeedback()
    episode_replay = [ts]
    action_replay = []
    for step_idx in tqdm(range(episode_len)):
        step_start = time.time()

        # Take a step in the environment using the chosen action
        ts, reward, terminated, truncated, info = env.step(action)
        episode_replay.append(ts)
        action_replay.append(action)  

        # Receive data from the headset
        headset_data = headset.receive_data()
        if headset_data is not None:
            headset_action, feedback = headset_control.run(
                headset_data, 
                ts['poses']['middle']
            )
            if headset_control.is_running():
                action[14:14+7] = headset_action        

        # update master bot actions
        action[:14] = get_master_bots_action(master_bot_left, master_bot_right)

        feedback.info = f"Episode {episode_idx}, Timestep: {str(step_idx).zfill(len(str(episode_len)))}/{episode_len}\n{info}"
        headset.send_feedback(feedback) 

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))        
    
    return episode_replay, action_replay

def confirm_episode(env, headset, episode_idx):
    # wait for user to redo or do next episode
    feedback = HeadsetFeedback()

    feedback.info = f"Episode {episode_idx} completed. Press Middle Pedal to Save, Right Pedal to Redo"
    headset.send_feedback(feedback)

    pedal_input = input()

    if pedal_input == 'y':
        return True
    else:
        return False
    


def save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx):
    feedback = HeadsetFeedback()
    feedback.info = f"Saving episode {episode_idx}..."
    headset.send_feedback(feedback)

    # convert data to dataset format
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
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
        data_dict['/action'].append(ts['control'])
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts['images'][cam_name])

    # HDF5
    try:
        image_shapes = {cam_name: data_dict[f'/observations/images/{cam_name}'][0].shape for cam_name in camera_names}
        qpos_len = len(data_dict['/observations/qpos'][0])
        qvel_len = len(data_dict['/observations/qvel'][0])
        action_len = len(data_dict['/action'][0])
    except IndexError:
        print('Empty episode, skipping...')
        return False
    t0 = time.time()
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
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
    print(f'Saving: {time.time() - t0:.1f} secs\n')
    return True

    
def collect_data(args: dict):
    dataset_dir = TASK_CONFIGS[args['task_name']]['dataset_dir']
    num_episodes = TASK_CONFIGS[args['task_name']]['num_episodes']
    episode_len = TASK_CONFIGS[args['task_name']]['episode_len']
    camera_names = TASK_CONFIGS[args['task_name']]['camera_names']
    start_episode_idx = args['episode_idx']

    # create the dataset directory if it does not exist
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    # setup the headset control
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)

    # setup the environment
    env = RealEnv(init_node=False, headset=headset) 
    
    episode_idx = start_episode_idx
    while episode_idx < num_episodes:
        # run the episode
        episode_replay, action_replay = run_episode(env, headset, master_bot_left, master_bot_right, episode_len, episode_idx)

        # confirm if the episode is to be saved
        ok = confirm_episode(env, headset, episode_idx)

        if not ok:
            continue

        # save the episode
        ok = save_episode(headset, episode_replay, action_replay, camera_names, dataset_dir, episode_idx)

        if not ok:
            continue

        episode_idx += 1        

if __name__ == "__main__":
    import rospy
    # Parse arguments
    parser = argparse.ArgumentParser(description="Headset control")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--episode_idx", type=int, required=True)
    args = parser.parse_args()
    args = vars(args)  

    rospy.init_node("collect_data")
    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        collect_data(args)
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)