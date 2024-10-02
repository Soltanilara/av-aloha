import time
import numpy as np
from dm_control import mjcf
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
from diff_ik import DiffIK
from grad_ik import GradIK
from webrtc_headset import WebRTCHeadset
from image_recorders import ROSImageRecorder, ZEDImageRecorder
from robot_utils import setup_puppet_bot, move_arms, move_grippers, sleep, setup_master_bot, torque_off, torque_on, get_arm_gripper_positions
from transform_utils import xyzw_to_wxyz, mat2pose, pose2mat, wxyz_to_xyzw
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
import rospy
from headset_control import HeadsetControl
from headset_utils import HeadsetFeedback
from kinematics import create_fk_fn, create_safety_fn
import mujoco
import os
from tqdm import tqdm

DT = REAL_DT

class RealEnv(gym.Env):

    def __init__(self, init_node=True, headset: WebRTCHeadset = None):
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

        # setup mujoco for forward kinematics
        self._mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'aloha_real.xml'))
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 
        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        self._left_eef_site = self._mjcf_root.find('site', LEFT_EEF_SITE)
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        self._left_fk_fn = create_fk_fn(self._physics, self._left_joints, self._left_eef_site)
        self._right_fk_fn = create_fk_fn(self._physics, self._right_joints, self._right_eef_site)
        self._middle_fk_fn = create_fk_fn(self._physics, self._middle_joints, self._middle_eef_site)
        
        self._middle_controller = DiffIK(
            physics=self._physics,
            joints=self._middle_joints,
            actuators=self._middle_actuators,
            eef_site=self._middle_eef_site,
            k_pos=0.3,
            k_ori=0.3,
            damping=1.0e-4,
            k_null=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
            q0=np.array(MIDDLE_ARM_POSE),
            max_angvel=3.14,
            integration_dt=DT,
            iterations=10
        )

        # setup ROS image recorder
        self.image_recorder = ROSImageRecorder(init_node=init_node, camera_names=['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'])
        # setup ZED image recorder
        self.zed_image_recorder = ZEDImageRecorder(headset=headset, auto_start=True)

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

        # do forward kinematics
        # send back ctrl instead of qpos because we want to send back the commanded position
        # real position might be affected by gravity and other forces
        left_pos, left_quat = mat2pose(self._left_fk_fn(self.left_ctrl[:6]))
        right_pos, right_quat = mat2pose(self._right_fk_fn(self.right_ctrl[:6]))
        middle_pos, middle_quat = mat2pose(self._middle_fk_fn(self.middle_ctrl))
        left_quat = xyzw_to_wxyz(left_quat)
        right_quat = xyzw_to_wxyz(right_quat)
        middle_quat = xyzw_to_wxyz(middle_quat)

        # get images
        image_dict = self.image_recorder.get_images()
        zed_image = self.zed_image_recorder.get_image()
        
        return {
            'joints': {
                'position': np.concatenate([left_joint_pos, right_joint_pos, middle_joint_pos]),
                'velocity': np.concatenate([left_joint_vel, right_joint_vel, middle_joint_vel]),
            },
            'control': np.concatenate([self.left_ctrl, self.right_ctrl, self.middle_ctrl]),
            'poses': {
                'left': np.concatenate([left_pos, left_quat]),
                'right': np.concatenate([right_pos, right_quat]),
                'middle': np.concatenate([middle_pos, middle_quat]),
            },
            'images': {
                'zed_cam_left': zed_image[:, :1280, :],
                'zed_cam_right': zed_image[:, 1280:, :],
                'wrist_cam_left': image_dict['cam_left_wrist'],
                'wrist_cam_right': image_dict['cam_right_wrist'],
                'overhead_cam': image_dict['cam_high'],
                'worms_eye_cam': image_dict['cam_low'],
            },
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # Reboot puppet robot gripper motors
        self.left_bot.dxl.robot_reboot_motors("single", "gripper", True)
        self.right_bot.dxl.robot_reboot_motors("single", "gripper", True)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
        time.sleep(1.0)
        move_grippers([self.left_bot, self.right_bot], [LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)
        move_arms([self.left_bot, self.right_bot, self.middle_bot], [LEFT_ARM_POSE[:6], RIGHT_ARM_POSE[:6], MIDDLE_ARM_POSE[:7]], move_time=2.5)

        self.left_ctrl[:6] = np.array(LEFT_ARM_POSE[:6])
        self.left_ctrl[6] = LEFT_GRIPPER_JOINT_NORMALIZE_FN(LEFT_GRIPPER_JOINT_CLOSE)
        self.right_ctrl[:6] = np.array(RIGHT_ARM_POSE[:6])
        self.right_ctrl[6] = RIGHT_GRIPPER_JOINT_NORMALIZE_FN(RIGHT_GRIPPER_JOINT_CLOSE)
        self.middle_ctrl = np.array(MIDDLE_ARM_POSE[:7])

        observation = self.get_obs()
        info = {}

        return observation, info

    def step(self, action: np.ndarray, all_joints=False) -> tuple:
        left_ctrl = action[:6]
        left_gripper = action[6] # val from 0 to 1
        right_ctrl = action[7:13]
        right_gripper = action[13] # val from 0 to 1
        middle_target = action[14:21]

        # set vals
        self.left_ctrl[:6] = left_ctrl
        self.right_ctrl[:6] = right_ctrl
        if all_joints:
            self.middle_ctrl = middle_target
        else:
            self.middle_ctrl = self._middle_controller.run(self.middle_ctrl, middle_target[:3], middle_target[3:])
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



def reset_env(env: RealEnv, headset: WebRTCHeadset, master_bot_left, master_bot_right):
    feedback = HeadsetFeedback()
    feedback.info = "Resetting the puppet arms..."
    headset.send_feedback(feedback)

    ts, info = env.reset()

    action = np.concatenate([
        ts['control'][:14],
        ts['poses']['middle'],
    ])
    env.step(action) # I step once for numba to compile the function (it doesn't actually do anything to robots)

def reset_master_arms(headset: WebRTCHeadset, master_bot_left, master_bot_right):

    feedback = HeadsetFeedback()
    feedback.info = "Resetting the master arms..."
    headset.send_feedback(feedback)

    """ Move all 4 robots to a pose where it is easy to start demonstration """
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")

    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # move arms to starting position
    move_arms([master_bot_left, master_bot_right], 
              [LEFT_ARM_POSE[:6], RIGHT_ARM_POSE[:6]],
              move_time=1.5)
    
    # halfway open master gripper position
    left_master_gripper_middle = (LEFT_MASTER_GRIPPER_JOINT_OPEN + LEFT_MASTER_GRIPPER_JOINT_CLOSE) / 2  
    right_master_gripper_middle = (RIGHT_MASTER_GRIPPER_JOINT_OPEN + RIGHT_MASTER_GRIPPER_JOINT_CLOSE) / 2

    # move grippers to starting position
    move_grippers([master_bot_left, master_bot_right], 
                    [left_master_gripper_middle, right_master_gripper_middle],
                  move_time=0.5)

def wait_for_user(env, headset, master_bot_left, master_bot_right, message="Align your head and close both grippers to start."):

    # almost closed master gripper position (90% closed)
    left_master_gripper_almost_close = LEFT_MASTER_GRIPPER_JOINT_CLOSE + 0.1 * (LEFT_MASTER_GRIPPER_JOINT_OPEN - LEFT_MASTER_GRIPPER_JOINT_CLOSE)
    right_master_gripper_almost_close = RIGHT_MASTER_GRIPPER_JOINT_CLOSE + 0.1 * (RIGHT_MASTER_GRIPPER_JOINT_OPEN - RIGHT_MASTER_GRIPPER_JOINT_CLOSE)

    # press gripper to start data collection
    # disable torque for only gripper joint of master robot to allow user movement
    master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
    master_bot_right.dxl.robot_torque_enable("single", "gripper", False)

    headset_control = HeadsetControl()
    feedback = HeadsetFeedback()
    headset_control.reset()

    while True:

        start_time = time.time()

        headset_data = headset.receive_data()
        if headset_data is not None:

            ts = env.get_obs()

            # get the action and feedback from the headset control
            headset_action, feedback = headset_control.run(
                headset_data, 
                ts['poses']['middle']
            )

            gripper_pos_left = master_bot_left.dxl.joint_states.position[6]
            gripper_pos_right = master_bot_right.dxl.joint_states.position[6]

            if (gripper_pos_left < left_master_gripper_almost_close) and \
                (gripper_pos_right < right_master_gripper_almost_close) and \
                feedback.head_out_of_sync == False:
                headset_control.start(
                    headset_data, 
                    ts['poses']['middle']
                )
                break


        feedback.info = message
        headset.send_feedback(feedback)


        time_until_next_step = REAL_DT - (time.time() - start_time)
        time.sleep(max(0, time_until_next_step))

    torque_off(master_bot_left)
    torque_off(master_bot_right)
    print(f'Started!')

    return headset_control

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
    from real_env import RealEnv
    from webrtc_headset import WebRTCHeadset

    # setup the headset
    headset = WebRTCHeadset()
    headset.run_in_thread()

    # source of data
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)

    # setup the environment
    env = RealEnv(init_node=False, headset=headset)

    reset_env(env, headset, master_bot_left, master_bot_right)

    reset_master_arms(headset, master_bot_left, master_bot_right)

    headset_control = wait_for_user(env, headset, master_bot_left, master_bot_right)

    # run 
    print(f"Starting...")
    ts = env.get_obs()
    action = np.concatenate([
        ts['control'][:14],
        ts['poses']['middle'],
    ])
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        ts, reward, terminated, truncated, info = env.step(action)

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

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = REAL_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))  

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
    
    
