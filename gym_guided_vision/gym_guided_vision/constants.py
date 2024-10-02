import pathlib
import os
# task parameters
XML_PATH = os.path.join(pathlib.Path(__file__).parent.resolve(), 'assets', 'aloha.xml') # note: absolute paths
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets'
DATA_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/data/recordings'

CAMERAS = [
    "zed_cam_left",
    "zed_cam_right",
    "wrist_cam_left",
    "wrist_cam_right",
    "overhead_cam",
    "worms_eye_cam",
]

RENDER_CAMERA = "overhead_cam"

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