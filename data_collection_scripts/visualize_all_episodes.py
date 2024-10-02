import os
import numpy as np
import cv2
import h5py
import argparse
import glob
import matplotlib.pyplot as plt
from constants import REAL_DT, SIM_DT, LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES
import re

def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def main(args):

    glob_path = args['glob_path']

    episode_files = glob.glob(glob_path)

    if len(episode_files) == 0:
        print(f'No episodes found in {glob_path}')
        exit()

    # make sure files all in same directory
    dataset_dir = os.path.dirname(episode_files[0])
    for episode_file in episode_files:
        if os.path.dirname(episode_file) != dataset_dir:
            print('All episodes must be in the same directory')
            exit()

    # make sure all files are .hdf5
    for episode_file in episode_files:
        if not episode_file.endswith('.hdf5'):
            print('All episodes must be .hdf5 files')
            exit()

    # sort episode files by episode number
    episode_files = sorted(episode_files, key=lambda x: int(re.search(r'\d+', x).group()))

    # make sure all episodes are present (continuously numbered)
    for i, episode_file in enumerate(episode_files):
        # check for string i in episode_file
        if str(i) not in episode_file:
            print(f'Missing episode_{i} in {dataset_dir}')
            exit()
    
    # if dataset_dir starts with sim_, use SIM_DT, else use REAL_DT
    DT = SIM_DT if dataset_dir.startswith('sim_') else REAL_DT    
    
    out = None

    video_path = os.path.join(dataset_dir, 'all_episodes.mp4')
    
    for i in range(len(episode_files)):
        qpos, qvel, action, image_dict = load_hdf5(episode_files[i])
        
        video = image_dict
        cam_names = list(video.keys())
        all_cam_videos = []

        min_h = np.inf
        total_w = 0
        for cam_name in cam_names:
            _, h, w, _ = video[cam_name].shape
            min_h = min(min_h, h)
            total_w += w

        for cam_name in cam_names:
            _, h, w, _ = video[cam_name].shape

            new_w = int(min_h * w / h)

            cam_video = video[cam_name]
            buffer = np.zeros((len(cam_video), min_h, new_w, 3), dtype=np.uint8)
            for ts in range(len(cam_video)):
                buffer[ts, :, :, :] = cv2.resize(cam_video[ts], (new_w, min_h))
            all_cam_videos.append(buffer)
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        
        if out is None:
            fps = int(1 / DT / 10)
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        for t in range(0, n_frames, 20):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            
            # write episode number on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 30)
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            image = cv2.putText(image.copy(), f'Episode {i}', bottomLeftCornerOfText, font,  
                   fontScale, fontColor, 2, cv2.LINE_AA) 
            
            out.write(image)
            
        print(f'Processed episode_{i}')
            
    if out is not None: 
        out.release()   
        print(f'Saved video to: {video_path}')
    else: 
        print('No episodes found to visualize')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glob_path', type=str, help='Dataset dir.', required=True)
    main(vars(parser.parse_args()))