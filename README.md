# AV-ALOHA

![AV-ALOHA](assets/av-aloha.gif)

This repository contains the code for the paper: **"Active Vision Might Be All You Need: Exploring Active Vision in Bimanual Robotic Manipulation"**. You can visit the [Project Page](https://soltanilara.github.io/av-aloha/) and check out the [ArXiv Paper](https://arxiv.org/abs/2409.17435).

## Overview

AV-ALOHA builds upon the ALOHA 2 system and introduces **active vision** for bimanual robotic manipulation. This repository includes:

- Teleoperation and data collection
- Training models with [LeRobot](https://github.com/huggingface/lerobot)
- Evaluation on both simulated and real-world AV-ALOHA setups

For the VR teleoperation and stereo camera passthrough functionality, refer to the [Unity App Repo](https://github.com/Soltanilara/av-aloha-unity).

**Note**: The code is under active development, and a more organized codebase will be available in future updates.

## Hardware Setup

AV-ALOHA extends [ALOHA 2](https://aloha-2.github.io/) by adding another [ViperX 300 S](https://www.trossenrobotics.com/viperx-300) robot arm. To install the additional arm, we used two 840mm 2020 extrusions with 4 L brackets. The [ZED Mini](https://www.stereolabs.com/store/products/zed-mini) serves as the active vision camera, attached using custom 3D-printed parts available in `assets/3D_printed_parts`.

## Software Installation

1. Install ROS Noetic and follow the [ALOHA Setup Instructions](https://github.com/tonyzhaozh/aloha/blob/06369f03cd8e0a47e16d3a90167853fd33af7557/README.md) for software and hardware setup, excluding their repo.
2. Bind the active vision robot arm to `/dev/ttyDXL_puppet_middle`.
3. Clone this repository:

    ```bash
    cd ~/interbotix_ws/src
    git clone https://github.com/Soltanilara/av-aloha
    git submodule init
    git submodule update

    # build ROS packages
    cd ~/interbotix_ws
    catkin_make
    ```

4. Set up the Conda environment:

    ```bash
    conda create -y -n lerobot python=3.10
    conda activate lerobot
    conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
    ```

5. Install the ZED Python API by following [these instructions](https://www.stereolabs.com/docs/app-development/python/install).
6. Install additional dependencies:

    ```bash
    pip install -e gym_guided_vision
    pip install -e lerobot
    pip install -r requirements.txt
    ```

## WebRTC Setup

1. Create a Firebase project and set up a Firestore database at [Firebase Console](https://console.firebase.google.com).

2. In your Firestore database, set the rules as follows:

    ```bash
    rules_version = '2';

    service cloud.firestore {
      match /databases/{database}/documents {
        match /<your_password_for_webrtc>/{document=**} {
          allow read, write: if true;
        }
      }
    }
    ```

3. In **Project Settings -> Service Accounts**, generate a new private key and name it `serviceAccountKey.json`. Place this file in the `data_collection_scripts` directory.

4. Create a file named `signalingSettings.json` in `data_collection_scripts` and paste the following:

    ```json
    {
        "robotID": "<robot id for your robot (e.g. robot_1)>",
        "password": "<your password same as in firestore rules>",
        "turn_server_url": "<turn url>",
        "turn_server_username": "<turn username>",
        "turn_server_password": "<turn password>"
    }
    ```

## Data Collection

### Simulation:

```bash
# in data_collection_scripts/
python record_sim_episodes --task_name sim_insert_peg --episode_idx 0
python replay_sim_episode --task_name sim_insert_peg --num_arms <2 or 3>
```

### Real Robot:

1. In one terminal, launch the robot:

    ```bash
    # in data_collection_scripts/
    source launch_robot.sh
    ```

2. In another terminal, activate the environment:

    ```bash
    # in data_collection_scripts/
    source activate.sh
    python record_episodes --task_name occluded_insertion --episode_idx 0
    ```

### Visualize an Episode:

```bash
# in data_collection_scripts/
python visualize_episodes.py --hdf5_path path/to/your/hdf5
```

### Push Dataset to Hugging Face:

```bash
# in repo root
huggingface-cli login
python lerobot/lerobot/scripts/push_dataset_to_hub.py \
    --raw-dir path/to/your/dataset \
    --repo-id <hf_id>/<dataset_name> \
    --raw-format aloha_hdf5
```

### Visualize Data from Hugging Face:

```bash
# in repo root
python lerobot/lerobot/scripts/visualize_dataset.py \
    --repo-id <hf_id>/<dataset_name>  \
    --episode-index 0
```

## Training

Ensure the config names are set correctly by modifying `lerobot/lerobot/configs`. Start training with:

```bash
# in repo root
python lerobot/lerobot/scripts/train.py \
    hydra.run.dir=outputs/train/sim_sew_needle_3arms_zed_static_wrist_act \
    hydra.job.name=sim_sew_needle_3arms_zed_static_wrist_act \
    device=cuda \
    env=sim_sew_needle_3arms \
    policy=zed_static_wrist_act \
    wandb.enable=true
```

## Evaluation

### Simulation Evaluation (as done in the paper):

```bash
# in repo root
python lerobot/lerobot/scripts/eval.py \
    -p outputs/train/sim_hook_package_2arms_wrist_act/checkpoints \
    --out-dir outputs/eval/sim_hook_package_2arms_wrist_act \
    eval.n_episodes=50 \
    eval.batch_size=10 \
    --save-video
```

### Single Checkpoint Evaluation:

1. Save your model to Hugging Face:

    ```bash
    # in eval_scripts/
    python save_policy.py \
        --repo_id iantc104/sim_slot_insertion_3arms_zed_wrist_act \
        --checkpoint_dir outputs/train/sim_slot_insertion_3arms_zed_wrist_act/checkpoints/014000/pretrained_model
    ```

2. Evaluate using the script in `eval_scripts`:

    #### Simulated Policy:

    ```bash
    # in eval_scripts/
    python eval.py \
        --policy iantc104/sim_slot_insertion_3arms_zed_wrist_act \
        --episode_len 300 \
        --num_episodes 50 \
        --sim_env gym_guided_vision/SlotInsertion-3Arms-v0
    ```

    #### Real Policy:

    ```bash
    # in eval_scripts/
    python eval.py \
        --policy iantc104/real_occluded_key_insertion_3arms_zed_act \
        --episode_len 700 \
        --num_episodes 50
    ```