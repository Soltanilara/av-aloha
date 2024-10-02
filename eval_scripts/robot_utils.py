import numpy as np
import time
from constants import REAL_DT
import time

DT = REAL_DT

import IPython
e = IPython.embed

from interbotix_xs_msgs.msg import JointSingleCommand

def get_arm_joint_positions(bot):
    if hasattr(bot, 'gripper') is False:
        return bot.arm.core.joint_states.position[:7]
    return bot.arm.core.joint_states.position[:6]

def get_arm_gripper_positions(bot):
    # check if bot has attr gripper
    if hasattr(bot, 'gripper') is False:
        raise NotImplementedError("This bot does not have a gripper")
    joint_position = bot.gripper.core.joint_states.position[6]
    return joint_position

def move_arms(bot_list, target_pose_list, move_time=1):
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            bot.arm.set_joint_positions(traj_list[bot_id][t], blocking=False)
        time.sleep(DT)

def move_grippers(bot_list, target_pose_list, move_time):
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [np.linspace(curr_pose, target_pose, num_steps) for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(DT)

def setup_puppet_bot(bot):
    if hasattr(bot, 'gripper') is False:
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
    else:
        bot.dxl.robot_reboot_motors("single", "gripper", True)
        bot.dxl.robot_set_operating_modes("group", "arm", "position")
        bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)

def setup_master_bot(bot):
    bot.dxl.robot_set_operating_modes("group", "arm", "pwm")
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_off(bot)

def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 800)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_P_Gain', 100)
    bot.dxl.robot_set_motor_registers("group", "arm", 'Position_I_Gain', 0)

def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    if hasattr(bot, 'gripper') is True:
        bot.dxl.robot_torque_enable("single", "gripper", False)
    

def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    if hasattr(bot, 'gripper') is True:
        bot.dxl.robot_torque_enable("single", "gripper", True)

def sleep(puppet_bot_left, puppet_bot_right, puppet_bot_middle):
    all_bots = [puppet_bot_left, puppet_bot_right, puppet_bot_middle]
    for bot in all_bots:
        torque_on(bot)

    left_puppet_sleep_position = (0, -1.7, 1.55, 0, 0.65, 0)
    right_puppet_sleep_position = (0, -1.7, 1.55, 0, 0.65, 0)
    middle_puppet_sleep_position = (0, -1.7, 1.55, 0, 0.65, 0, 0)
    move_arms(all_bots, [left_puppet_sleep_position, right_puppet_sleep_position, middle_puppet_sleep_position], move_time=3)

    left_puppet_sleep_position = (0, -1.85, 1.6, 0, 0.65, 0)
    right_puppet_sleep_position = (0, -1.85, 1.6, 0, 0.65, 0)
    middle_puppet_sleep_position = (0, -1.85, 1.6, 0, 0.65, 0, 0)
    move_arms(all_bots, [left_puppet_sleep_position, right_puppet_sleep_position, middle_puppet_sleep_position], move_time=3)
