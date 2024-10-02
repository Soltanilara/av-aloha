import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from constants import (
    XML_DIR, REAL_DT, 
    LEFT_GRIPPER_JOINT_UNNORMALIZE_FN, RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN,
    LEFT_GRIPPER_JOINT_NORMALIZE_FN, RIGHT_GRIPPER_JOINT_NORMALIZE_FN,
    LEFT_GRIPPER_VELOCITY_NORMALIZE_FN, RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN,
    LEFT_MASTER_GRIPPER_JOINT_NORMALIZE_FN, RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN,
    LEFT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN, RIGHT_MASTER_GRIPPER_JOINT_UNNORMALIZE_FN,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN,
    LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE,
    LEFT_MASTER_GRIPPER_JOINT_OPEN, RIGHT_MASTER_GRIPPER_JOINT_OPEN,
    LEFT_MASTER_GRIPPER_JOINT_CLOSE, RIGHT_MASTER_GRIPPER_JOINT_CLOSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE,
)
from image_recorders import ROSImageRecorder, ZEDImageRecorder
from robot_utils import setup_puppet_bot, move_arms, move_grippers, sleep
from transform_utils import xyzw_to_wxyz, mat2pose
import os

from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

DT = REAL_DT

class RealEnv(gym.Env):

    def __init__(self, init_node=True):
        # setup observation and action space
        self.observation_space = spaces.Dict({
            'joints': spaces.Dict({
                'position': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
                'velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,))  # 21 joint velocities
            }),
            'control': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
            'poses': spaces.Dict({
                'left': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # left arm pose
                'right': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # right arm pose
                'middle': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))  # middle arm pose
            }),
            'images': spaces.Dict({
                'zed_cam_left': spaces.Box(low=0, high=255, shape=(720, 1280, 3)),  # zed camera image
                'zed_cam_right': spaces.Box(low=0, high=255, shape=(720, 1280, 3)),  # zed camera image
                'wrist_cam_left': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # left wrist camera image
                'wrist_cam_right': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # right wrist camera image
                'overhead_cam': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # high camera image
                'worms_eye_cam': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # low camera image
            }),
        })
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        )   

        # setup ROS image recorder
        self.image_recorder = ROSImageRecorder(init_node=init_node, camera_names=['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'])
        # setup ZED image recorder
        self.zed_image_recorder = ZEDImageRecorder(auto_start=True)

        # setup bots
        self.left_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=False)
        self.right_bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)
        self.middle_bot = InterbotixManipulatorXS(robot_model="vx300s_7dof", group_name="arm", gripper_name=None, robot_name=f"puppet_middle", init_node=False)
        sleep(self.left_bot, self.right_bot, self.middle_bot)
        setup_puppet_bot(self.left_bot)
        setup_puppet_bot(self.right_bot)
        setup_puppet_bot(self.middle_bot)
        
        # cmd buffer
        self.left_ctrl = np.array(self.left_bot.arm.core.joint_states.position[:7])
        self.right_ctrl = np.array(self.right_bot.arm.core.joint_states.position[:7])
        self.middle_ctrl = np.array(self.middle_bot.arm.core.joint_states.position[:7])  

        # normalize the gripper joints
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(self.left_ctrl[6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(self.right_ctrl[6])  

    def get_obs(self) -> np.ndarray:
        # get joint positions and velocities
        left_joint_pos = np.array(self.left_bot.arm.core.joint_states.position[:7])
        left_joint_pos[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(left_joint_pos[6])
        right_joint_pos = np.array(self.right_bot.arm.core.joint_states.position[:7])
        right_joint_pos[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(right_joint_pos[6])
        middle_joint_pos = np.array(self.middle_bot.arm.core.joint_states.position[:7])

        try:
            left_joint_vel = np.array(self.left_bot.arm.core.joint_states.velocity[:7])
            left_joint_vel[6] = LEFT_GRIPPER_VELOCITY_NORMALIZE_FN(left_joint_vel[6])
            right_joint_vel = np.array(self.right_bot.arm.core.joint_states.velocity[:7])
            right_joint_vel[6] = RIGHT_GRIPPER_VELOCITY_NORMALIZE_FN(right_joint_vel[6])
            middle_joint_vel = np.array(self.middle_bot.arm.core.joint_states.velocity[:7])
        except IndexError:
            left_joint_vel = np.zeros(7)
            right_joint_vel = np.zeros(7)
            middle_joint_vel = np.zeros(7)

        # get images
        image_dict = self.image_recorder.get_images()
        zed_image = self.zed_image_recorder.get_image()
        
        return {
            'pixels': {
                'zed_cam_left': zed_image[:, :1280, :],
                'zed_cam_right': zed_image[:, 1280:, :],
                'wrist_cam_left': image_dict['cam_left_wrist'],
                'wrist_cam_right': image_dict['cam_right_wrist'],
                'overhead_cam': image_dict['cam_high'],
                'worms_eye_cam': image_dict['cam_low'],
            },
            'agent_pos': np.concatenate([left_joint_pos, right_joint_pos, middle_joint_pos]),
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # Reboot puppet robot gripper motors
        self.left_bot.dxl.robot_reboot_motors("single", "gripper", True)
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
        move_arms([self.left_bot, self.right_bot, self.middle_bot], [LEFT_ARM_POSE[:6], RIGHT_ARM_POSE[:6], MIDDLE_ARM_POSE[:7]], move_time=3.0)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)

        self.left_ctrl[:6] = np.array(LEFT_ARM_POSE[:6])
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(LEFT_GRIPPER_JOINT_CLOSE)
        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(RIGHT_GRIPPER_JOINT_CLOSE)
        self.middle_ctrl = np.array(MIDDLE_ARM_POSE[:7])

        observation = self.get_obs()
        info = {}

        return observation, info
    
    def sleep(self):
        sleep(self.left_bot, self.right_bot, self.middle_bot)

    def step(self, action: np.ndarray, all_joints=False) -> tuple:
        left_ctrl = action[:6]
        left_gripper = action[6] # val from 0 to 1
        right_ctrl = action[7:13]
        right_gripper = action[13] # val from 0 to 1
        middle_target = action[14:21]

        # set vals
        self.left_ctrl[:6] = left_ctrl
        self.right_ctrl[:6] = right_ctrl
        self.middle_ctrl = middle_target
        self.left_ctrl[6] = left_gripper
        self.right_ctrl[6] = right_gripper

        # move the robots
        self.left_bot.arm.set_joint_positions(self.left_ctrl[:6], blocking=False)
        self.left_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", 
                                                                         cmd=LEFT_GRIPPER_JOINT_UNNORMALIZE_FN(self.left_ctrl[6])))
        self.right_bot.arm.set_joint_positions(self.right_ctrl[:6], blocking=False)
        self.right_bot.gripper.core.pub_single.publish(JointSingleCommand(name="gripper", 
                                                                          cmd=RIGHT_GRIPPER_JOINT_UNNORMALIZE_FN(self.right_ctrl[6])))
        self.middle_bot.arm.set_joint_positions(self.middle_ctrl, blocking=False) 
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info
    
    def close(self):
        self.zed_image_recorder.stop()


def get_master_bots_action(master_bot_left, master_bot_right):
    action = np.zeros(14) # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    action[6] = LEFT_MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    action[7+6] = RIGHT_MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])

    return action

def make_real_env(init_node):
    env = RealEnv(init_node, headset=None)
    return env

def main():
    # setup the environment
    env = RealEnv(init_node=False)

    # reset the environment
    obs, info = env.reset()

    # sleep
    env.sleep()

if __name__ == "__main__":
    import rospy
    import os

    def shutdown():
        print("Shutting down...")
        os._exit(42)
    rospy.on_shutdown(shutdown)

    try:
        main()
    except KeyboardInterrupt:
        print("Shutting down...")
        os._exit(42)
    
    