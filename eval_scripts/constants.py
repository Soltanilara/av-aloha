import pathlib
import os
# task parameters
XML_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets', 'aloha.xml') # note: absolute paths
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets'
DATA_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/data'
TASK_CONFIGS = {
    'cup_transfer': {
        'dataset_dir': DATA_DIR + '/cup_transfer',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': [
            "zed_cam_left",
            "zed_cam_right",
            "wrist_cam_left",
            "wrist_cam_right",
            "overhead_cam",
            "worms_eye_cam",
        ]
    },
    'multi_peg': {
        'dataset_dir': DATA_DIR + '/multi_peg',
        'num_episodes': 50,
        'episode_len': 600,
        'camera_names': [
            "zed_cam_left",
            "zed_cam_right",
            "wrist_cam_left",
            "wrist_cam_right",
            "overhead_cam",
            "worms_eye_cam",
        ]
    },
}
SIM_TASK_CONFIGS = {
    'sim_insert_peg': {
        'dataset_dir': DATA_DIR + '/sim_insert_peg/3arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['zed_cam'],
    },
    '2arms_sim_insert_peg': {
        'dataset_dir': DATA_DIR + '/sim_insert_peg/2arms',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low'],
    },
    'sim_slot_insertion': {
        'dataset_dir': DATA_DIR + '/sim_slot_insertion',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low'],
    },
    'sim_sew_needle': {
        'dataset_dir': DATA_DIR + '/sim_sew_needle',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low'],
    },
    'sim_tube_transfer': {
        'dataset_dir': DATA_DIR + '/sim_tube_transfer',
        'num_episodes': 50,
        'episode_len': 350,
        'camera_names': ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low'],
    },
    'sim_hook_package': {
        'dataset_dir': DATA_DIR + '/sim_hook_package',
        'num_episodes': 50,
        'episode_len': 300,
        'camera_names': ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low'],
    },
}

# control parameters
REAL_DT = 0.02

# physics parameters
SIM_PHYSICS_DT=0.002
SIM_DT = 0.04
SIM_PHYSICS_ENV_STEP_RATIO = int(SIM_DT/SIM_PHYSICS_DT)
SIM_DT = SIM_PHYSICS_DT * SIM_PHYSICS_ENV_STEP_RATIO

# robot parameters
LEFT_ARM_POSE = [0, -0.082, 1.06, 0, -0.953, 0, 0.02239]
RIGHT_ARM_POSE = [0, -0.082, 1.06, 0, -0.953, 0, 0.02239]
MIDDLE_ARM_POSE = [0, -0.8, 0.8, 0, 0.5, 0, 0]
LEFT_JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_left_finger",
]
RIGHT_JOINT_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_right_finger",
]
MIDDLE_JOINT_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_ACTUATOR_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
]
RIGHT_ACTUATOR_NAMES = [
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]
MIDDLE_ACTUATOR_NAMES = [
    "middle_waist",
    "middle_shoulder",
    "middle_elbow",
    "middle_forearm_roll",
    "middle_wrist_1_joint",
    "middle_wrist_2_joint",
    "middle_wrist_3_joint",
]
LEFT_EEF_SITE = "left_gripper_control"
RIGHT_EEF_SITE = "right_gripper_control"
MIDDLE_EEF_SITE = "middle_zed_camera_center"
MIDDLE_BASE_LINK = "middle_base_link"
LEFT_GRIPPER_JOINT_NAMES = ["left_left_finger", "left_right_finger"]
RIGHT_GRIPPER_JOINT_NAMES = ["right_left_finger", "right_right_finger"]

# Gripper joint limits (qpos[6])
LEFT_GRIPPER_JOINT_OPEN = 0.05982525274157524
LEFT_GRIPPER_JOINT_CLOSE = -0.99055535531044006
RIGHT_GRIPPER_JOINT_OPEN =   0.11044661700725555
RIGHT_GRIPPER_JOINT_CLOSE = -1.0139613151550293

# TODO: ANDREW SET THESE VALUES
LEFT_MASTER_GRIPPER_JOINT_OPEN = 0.6596117615699768
LEFT_MASTER_GRIPPER_JOINT_CLOSE = -0.1672039031982422
RIGHT_MASTER_GRIPPER_JOINT_OPEN = 0.7240389585494995
RIGHT_MASTER_GRIPPER_JOINT_CLOSE = -0.07976700365543365

############################ Helper functions ############################
LEFT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEFT_GRIPPER_JOINT_CLOSE) / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
LEFT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE) + LEFT_GRIPPER_JOINT_CLOSE
RIGHT_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - RIGHT_GRIPPER_JOINT_CLOSE) / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE) + RIGHT_GRIPPER_JOINT_CLOSE
LEFT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (LEFT_GRIPPER_JOINT_OPEN - LEFT_GRIPPER_JOINT_CLOSE)
RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (RIGHT_GRIPPER_JOINT_OPEN - RIGHT_GRIPPER_JOINT_CLOSE)

LEFT_MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEFT_MASTER_GRIPPER_JOINT_CLOSE) / (LEFT_MASTER_GRIPPER_JOINT_OPEN - LEFT_MASTER_GRIPPER_JOINT_CLOSE)
LEFT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (LEFT_MASTER_GRIPPER_JOINT_OPEN - LEFT_MASTER_GRIPPER_JOINT_CLOSE) + LEFT_MASTER_GRIPPER_JOINT_CLOSE
RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - RIGHT_MASTER_GRIPPER_JOINT_CLOSE) / (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE)
RIGHT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE) + RIGHT_MASTER_GRIPPER_JOINT_CLOSE