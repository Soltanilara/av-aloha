from interbotix_xs_modules.arm import InterbotixManipulatorXS
from ..robot_utils import move_grippers, torque_on
from ..constants import (
    LEFT_GRIPPER_JOINT_CLOSE,
    LEFT_GRIPPER_JOINT_OPEN,
    RIGHT_GRIPPER_JOINT_CLOSE,
    RIGHT_GRIPPER_JOINT_OPEN,
)
import rospy

def main():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)    
    
    puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_left.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    puppet_bot_right.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    
    torque_on(puppet_bot_left)
    torque_on(puppet_bot_right)

    move_grippers([puppet_bot_left, puppet_bot_right], [LEFT_GRIPPER_JOINT_OPEN, RIGHT_GRIPPER_JOINT_OPEN], move_time=1.0)
    rospy.sleep(1.0)
    move_grippers([puppet_bot_left, puppet_bot_right], [LEFT_GRIPPER_JOINT_CLOSE, RIGHT_GRIPPER_JOINT_CLOSE], move_time=1.0)

if __name__ == '__main__':
    main()
