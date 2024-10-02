from interbotix_xs_modules.arm import InterbotixManipulatorXS
from robot_utils import torque_off
import rospy

def main():
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)

    torque_off(puppet_bot_left)

    while not rospy.is_shutdown():
        start_time = rospy.Time.now()
        print("velocity", puppet_bot_left.arm.core.joint_states.velocity[:6])
        print("position", puppet_bot_left.arm.core.joint_states.position[:6])
        rospy.sleep(0.2)


if __name__ == '__main__':
    main()
