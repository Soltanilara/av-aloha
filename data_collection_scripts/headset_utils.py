import numpy as np
from numba import jit, float64
from numba.types import UniTuple
from scipy.spatial.transform import Rotation as R
from transform_utils import (
    pose2mat,
    mat2pose,
)

TRANSFORM_TO_WORLD = np.ascontiguousarray(np.eye(4))
TRANSFORM_TO_WORLD[:3, :3] = R.from_euler('xyz', [-90, 0, -90], degrees=True).as_matrix()
WORLD_TO_TRANSFORM = np.ascontiguousarray(np.linalg.inv(TRANSFORM_TO_WORLD))

class HeadsetData:
    h_pos = np.zeros(3)
    h_quat = np.zeros(4)
    l_pos = np.zeros(3)
    l_quat = np.zeros(4)
    l_thumbstick_x = 0
    l_thumbstick_y = 0
    l_index_trigger = 0
    l_hand_trigger = 0
    l_button_one = False
    l_button_two = False
    l_button_thumbstick = False
    r_pos = np.zeros(3)
    r_quat = np.zeros(4)
    r_thumbstick_x = 0
    r_thumbstick_y = 0
    r_index_trigger = 0
    r_hand_trigger = 0
    r_button_one = False
    r_button_two = False
    r_button_thumbstick = False

class HeadsetFeedback:
    head_out_of_sync = False
    left_out_of_sync = False
    right_out_of_sync = False
    info = ""
    left_arm_position = np.zeros(3)
    left_arm_rotation = np.zeros(4)
    right_arm_position = np.zeros(3)
    right_arm_rotation = np.zeros(4)
    middle_arm_position = np.zeros(3)
    middle_arm_rotation = np.zeros(4)

@jit(UniTuple(float64[:], 2)(float64[:], float64[:]), nopython=True, fastmath=True, cache=True)
def convert_left_to_right_coordinates(left_pos, left_quat):

    x = left_pos[0]
    y = -left_pos[1] # flip y from left to right
    z = left_pos[2]
    qx = -left_quat[0] # flip rotation from left to right
    qy = left_quat[1]
    qz = -left_quat[2] # flip rotation from left to right
    qw = left_quat[3]

    transform = pose2mat(np.array([x, y, z]), np.array([qx, qy, qz, qw]))

    transform = np.ascontiguousarray(transform)

    transform = TRANSFORM_TO_WORLD @ transform

    right_pos, right_quat = mat2pose(transform)

    return right_pos, right_quat

@jit(UniTuple(float64[:], 2)(float64[:], float64[:]), nopython=True, fastmath=True, cache=True)
def convert_right_to_left_coordinates(right_pos, right_quat):

    transform = pose2mat(right_pos, right_quat)

    transform = np.ascontiguousarray(transform)

    transform = WORLD_TO_TRANSFORM @ transform

    pos, quat = mat2pose(transform)

    x = pos[0]
    y = -pos[1] # flip y from right to left
    z = pos[2]
    qx = -quat[0] # flip rotation from right to left
    qy = quat[1]
    qz = -quat[2] # flip rotation from right to left
    qw = quat[3]

    return np.array([x, y, z]), np.array([qx, qy, qz, qw])