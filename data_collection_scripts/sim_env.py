import time
import numpy as np
import mujoco.viewer
from dm_control import mjcf
import gymnasium as gym
from gymnasium import spaces
from constants import (
    XML_DIR, 
    SIM_DT, SIM_PHYSICS_DT, SIM_PHYSICS_ENV_STEP_RATIO,
    LEFT_ARM_POSE, RIGHT_ARM_POSE, MIDDLE_ARM_POSE,
    LEFT_JOINT_NAMES, RIGHT_JOINT_NAMES, MIDDLE_JOINT_NAMES,
    LEFT_ACTUATOR_NAMES, RIGHT_ACTUATOR_NAMES, MIDDLE_ACTUATOR_NAMES,
    LEFT_EEF_SITE, RIGHT_EEF_SITE, MIDDLE_EEF_SITE, MIDDLE_BASE_LINK,
    LEFT_GRIPPER_JOINT_NAMES, RIGHT_GRIPPER_JOINT_NAMES
)
from diff_ik import DiffIK
from grad_ik import GradIK
import os
from kinematics import create_fk_fn, create_safety_fn
from transform_utils import xyzw_to_wxyz, mat2pose, pose2mat, wxyz_to_xyzw

CAMERAS = ['zed_cam', 'cam_left_wrist', 'cam_right_wrist', 'cam_high', 'cam_low']

def make_sim_env(task_name, cameras=CAMERAS):
    if 'sim_insert_peg' in task_name:
        return InsertPegEnv(cameras=cameras)
    elif 'sim_slot_insertion' in task_name:
        return SlotInsertionEnv(cameras=cameras)
    elif 'sim_sew_needle' in task_name:
        return SewNeedleEnv(cameras=cameras)
    elif 'sim_tube_transfer' in task_name:
        return TubeTransferEnv(cameras=cameras)
    elif 'sim_hook_package' in task_name:
        return HookPackageEnv(cameras=cameras)
    else:
        raise NotImplementedError

class GuidedVisionEnv(gym.Env):

    def __init__(self, xml, cameras=CAMERAS):
        self._mjcf_root = mjcf.from_path(xml)  
        self._mjcf_root.option.timestep = SIM_PHYSICS_DT  
        
        self._physics = mjcf.Physics.from_mjcf_model(self._mjcf_root) 

        self.observation_space = spaces.Dict({
            'joints': spaces.Dict({
                'position': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
                'velocity': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,))  # 21 joint velocities
            }),
            'qpos': spaces.Box(low=-float('inf'), high=float('inf')), 
            'control': spaces.Box(low=-float('inf'), high=float('inf'), shape=(21,)),  # 21 joint positions
            'poses': spaces.Dict({
                'left': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # left arm pose
                'right': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,)),  # right arm pose
                'middle': spaces.Box(low=-float('inf'), high=float('inf'), shape=(7,))  # middle arm pose
            }),
            'images': spaces.Dict({
                'zed_cam': spaces.Box(low=0, high=255, shape=(720, 2*1280, 3)),  # zed camera image
                'cam_left_wrist': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # left wrist camera image
                'cam_right_wrist': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # right wrist camera image
                'cam_high': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # high camera image
                'cam_low': spaces.Box(low=0, high=255, shape=(480, 640, 3)),  # low camera image
            }),
        })
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
        )   

        self._middle_base_link = self._mjcf_root.find('body', MIDDLE_BASE_LINK)
        self._middle_base_link_init_pos = self._middle_base_link.pos.copy()

        self._left_joints = [self._mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
        self._right_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_JOINT_NAMES]
        self._middle_joints = [self._mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
        self._left_actuators = [self._mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
        self._right_actuators = [self._mjcf_root.find('actuator', name) for name in RIGHT_ACTUATOR_NAMES]
        self._middle_actuators = [self._mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
        self._left_eef_site = self._mjcf_root.find('site', LEFT_EEF_SITE)
        self._right_eef_site = self._mjcf_root.find('site', RIGHT_EEF_SITE)
        self._middle_eef_site = self._mjcf_root.find('site', MIDDLE_EEF_SITE)
        self._left_fk_fn = create_fk_fn(self._physics, self._left_joints[:6], self._left_eef_site)
        self._right_fk_fn = create_fk_fn(self._physics, self._right_joints[:6], self._right_eef_site)
        self._middle_fk_fn = create_fk_fn(self._physics, self._middle_joints[:7], self._middle_eef_site)
        self._left_gripper_joints = [self._mjcf_root.find('joint', name) for name in LEFT_GRIPPER_JOINT_NAMES]
        self._right_gripper_joints = [self._mjcf_root.find('joint', name) for name in RIGHT_GRIPPER_JOINT_NAMES]

        # set up controllers
        self._left_controller = GradIK(
            physics=self._physics,
            joints = self._left_joints[:6],
            actuators=self._left_actuators[:6],
            eef_site=self._left_eef_site,
            step_size=0.0001, 
            min_cost_delta=1.0e-12, 
            max_iterations=50, 
            position_weight=500.0,
            rotation_weight=100.0,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
            joint_displacement_weight=np.array(6*[50.0]),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=0.3,
            joint_p = 0.9,
        )
        self._right_controller = GradIK(
            physics=self._physics,
            joints=self._right_joints[:6],
            actuators=self._right_actuators[:6],
            eef_site=self._right_eef_site,
            step_size=0.0001, 
            min_cost_delta=1.0e-12, 
            max_iterations=50, 
            position_weight=500.0,
            rotation_weight=100.0,
            joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0]),
            joint_displacement_weight=np.array(6*[50.0]),
            position_threshold=0.001,
            rotation_threshold=0.001,
            max_pos_diff=0.1,
            max_rot_diff=0.3,
            joint_p = 0.9,
        )
        self._middle_controller = DiffIK(
            physics=self._physics,
            joints=self._middle_joints,
            actuators=self._middle_actuators,
            eef_site=self._middle_eef_site,
            k_pos=0.9,
            k_ori=0.9,
            damping=1.0e-4,
            k_null=np.array([10.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
            q0=np.array(MIDDLE_ARM_POSE),
            max_angvel=3.14,
            integration_dt=SIM_DT,
            iterations=10,
        )

        self.left_gripper_range = self._physics.bind(self._left_actuators[-1]).ctrlrange
        self.right_gripper_range = self._physics.bind(self._right_actuators[-1]).ctrlrange
        self.left_gripper_norm_fn = lambda x: (x - self.left_gripper_range[0]) / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_norm_fn = lambda x: (x - self.right_gripper_range[0]) / (self.right_gripper_range[1] - self.right_gripper_range[0])
        self.left_gripper_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0]) + self.left_gripper_range[0]
        self.right_gripper_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0]) + self.right_gripper_range[0]
        self.left_gripper_vel_norm_fn = lambda x: x / (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_vel_norm_fn = lambda x: x / (self.right_gripper_range[1] - self.right_gripper_range[0])
        self.left_gripper_vel_unnorm_fn = lambda x: x * (self.left_gripper_range[1] - self.left_gripper_range[0])
        self.right_gripper_vel_unnorm_fn = lambda x: x * (self.right_gripper_range[1] - self.right_gripper_range[0])

        # for GUI and time keeping
        self._viewer = None 

        # check all cameras are valid
        for camera in cameras:
            assert camera in CAMERAS, f"Invalid camera name: {camera}"

        self._cameras = cameras

    def get_obs(self) -> np.ndarray:
        left_qpos = self._physics.bind(self._left_joints).qpos.copy()
        left_qpos[6] = self.left_gripper_norm_fn(left_qpos[6])
        right_qpos = self._physics.bind(self._right_joints).qpos.copy()
        right_qpos[6] = self.right_gripper_norm_fn(right_qpos[6])
        middle_qpos = self._physics.bind(self._middle_joints).qpos.copy()
        left_qvel = self._physics.bind(self._left_joints).qvel.copy()
        left_qvel[6] = self.left_gripper_vel_norm_fn(left_qvel[6])
        right_qvel = self._physics.bind(self._right_joints).qvel.copy()
        right_qvel[6] = self.right_gripper_vel_norm_fn(right_qvel[6])
        middle_qvel = self._physics.bind(self._middle_joints).qvel.copy()
        left_ctrl = self._physics.bind(self._left_actuators).ctrl.copy()
        left_ctrl[6] = self.left_gripper_norm_fn(left_ctrl[6])
        right_ctrl = self._physics.bind(self._right_actuators).ctrl.copy()
        right_ctrl[6] = self.right_gripper_norm_fn(right_ctrl[6])
        middle_ctrl = self._physics.bind(self._middle_actuators).ctrl.copy()
        qpos = self._physics.data.qpos.copy()

        # send back ctrl instead of qpos because we want to send back the commanded position
        # real position might be affected by gravity and other forces
        left_pos, left_quat = mat2pose(self._left_fk_fn(self._physics.bind(self._left_actuators[:6]).ctrl))
        right_pos, right_quat = mat2pose(self._right_fk_fn(self._physics.bind(self._right_actuators[:6]).ctrl))
        middle_pos, middle_quat = mat2pose(self._middle_fk_fn(self._physics.bind(self._middle_actuators).ctrl))
        left_quat = xyzw_to_wxyz(left_quat)
        right_quat = xyzw_to_wxyz(right_quat)
        middle_quat = xyzw_to_wxyz(middle_quat)

        images = {}
        for camera in self._cameras:
            if 'zed_cam' in camera:
                images['zed_cam'] = np.concatenate([
                    self._physics.render(height=720, width=720, camera_id='zed_cam_left'),
                    self._physics.render(height=720, width=720, camera_id='zed_cam_right'),
                ], axis=1)
            elif 'cam_left_wrist' in camera:
                images['cam_left_wrist'] = self._physics.render(height=480, width=640, camera_id='wrist_cam_left')
            elif 'cam_right_wrist' in camera:
                images['cam_right_wrist'] = self._physics.render(height=480, width=640, camera_id='wrist_cam_right')
            elif 'cam_high' in camera:
                images['cam_high'] = self._physics.render(height=480, width=640, camera_id='overhead_cam')
            elif 'cam_low' in camera:
                images['cam_low'] = self._physics.render(height=480, width=640, camera_id='worms_eye_cam')
            else:
                raise NotImplementedError(f"Camera {camera} not implemented")

        return {
            'joints': {
                'position': np.concatenate([left_qpos, right_qpos, middle_qpos]),
                'velocity': np.concatenate([left_qvel, right_qvel, middle_qvel])
            },
            'qpos': qpos,
            'control': np.concatenate([left_ctrl, right_ctrl, middle_ctrl]),
            'poses': {
                'left': np.concatenate([left_pos,left_quat]),
                'right': np.concatenate([right_pos, right_quat]),
                'middle': np.concatenate([middle_pos, middle_quat])
            },
            'images': images,
        }

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        self._physics.reset()
        self._physics.bind(self._left_joints).qpos = LEFT_ARM_POSE
        self._physics.bind(self._left_gripper_joints).qpos = self.left_gripper_unnorm_fn(1)
        self._physics.bind(self._right_joints).qpos = RIGHT_ARM_POSE
        self._physics.bind(self._right_gripper_joints).qpos = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_joints).qpos = MIDDLE_ARM_POSE
        self._physics.bind(self._left_actuators).ctrl = LEFT_ARM_POSE
        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1)
        self._physics.bind(self._right_actuators).ctrl = RIGHT_ARM_POSE
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1)
        self._physics.bind(self._middle_actuators).ctrl = MIDDLE_ARM_POSE

        self._physics.forward()

        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info   

    def set_qpos(self, qpos: np.ndarray):
        self._physics.data.qpos[:] = qpos
        # forward kinematics
        self._physics.forward()


    def step_joints(self, action: np.ndarray) -> tuple:
        left_joints = action[:6]
        left_gripper = action[6] # val from 0 to 1
        right_joints = action[7:13]
        right_gripper = action[13] # val from 0 to 1
        middle_joints = action[14:21]

        self._physics.bind(self._left_actuators[:6]).ctrl = left_joints
        self._physics.bind(self._right_actuators[:6]).ctrl = right_joints
        self._physics.bind(self._middle_actuators).ctrl = middle_joints

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(right_gripper)

        # step physics
        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = ""

        return observation, reward, terminated, truncated, info




    def step(self, action: np.ndarray) -> tuple:
        left_target = action[:7]
        left_gripper = action[7] # val from 0 to 1
        right_target = action[8:15]
        right_gripper = action[15] # val from 0 to 1
        middle_target = action[16:23]

        self._physics.bind(self._left_actuators[:6]).ctrl = self._left_controller.run(
            self._physics.bind(self._left_joints).qpos[:6],
            left_target[:3],
            left_target[3:]
        )
        self._physics.bind(self._right_actuators[:6]).ctrl = self._right_controller.run(
            self._physics.bind(self._right_joints).qpos[:6],
            right_target[:3],
            right_target[3:]
        )
        self._physics.bind(self._middle_actuators).ctrl = self._middle_controller.run(
            self._physics.bind(self._middle_joints).qpos,
            middle_target[:3],
            middle_target[3:]
        )

        self._physics.bind(self._left_actuators[6]).ctrl = self.left_gripper_unnorm_fn(1-left_gripper)
        self._physics.bind(self._right_actuators[6]).ctrl = self.right_gripper_unnorm_fn(1-right_gripper)

        # step physics
        self._physics.step(nstep=SIM_PHYSICS_ENV_STEP_RATIO)
        
        observation = self.get_obs()
        reward = 0
        terminated = False
        truncated = False
        info = ""

        return observation, reward, terminated, truncated, info

    def render_viewer(self) -> np.ndarray:
        if self._viewer is None:
            # launch viewer
            self._viewer = mujoco.viewer.launch_passive(
                self._physics.model.ptr,
                self._physics.data.ptr,
                show_left_ui=True,
                show_right_ui=True,
            )
        # render viewer
        self._viewer.sync()


    def hide_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = np.array([0, -2, 0])

    def show_middle_arm(self):
        self._physics.bind(self._middle_base_link).pos = self._middle_base_link_init_pos


    def close(self) -> None:
        """
        Closes the viewer if it's open.
        """
        if self._viewer is not None:
            self._viewer.close()


class InsertPegEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_insert_peg.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4

        self._peg_joint = self._mjcf_root.find('joint', 'peg_joint')
        self._hole_joint = self._mjcf_root.find('joint', 'hole_joint')

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        peg_touch_table = False
        hole_touch_table = False
        peg_touch_hole = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "peg" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1.startswith("hole-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "peg":
                peg_touch_table = True

            if geom1 == "table" and geom2.startswith("hole-"):
                hole_touch_table = True

            if geom1 == "peg" and geom2.startswith("hole-"):
                peg_touch_hole = True

            if geom1 == "peg" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not hole_touch_table): # grasp both
            reward = 2
        if peg_touch_hole and (not peg_touch_table) and (not hole_touch_table): # peg and socket touching
            reward = 3
        if pin_touched: # successful insertion
            reward = 4
        return reward

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [0.1, 0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.01, 0.01]
        ranges = np.vstack([x_range, y_range, z_range])
        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.1, -0.2]
        y_range = [-0.1, 0.1]
        z_range = [0.021, 0.021]
        ranges = np.vstack([x_range, y_range, z_range])
        hole_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hole_quat = np.array([1, 0, 0, 0])

        self._physics.bind(self._peg_joint).qpos = np.concatenate([peg_position, peg_quat])
        self._physics.bind(self._hole_joint).qpos = np.concatenate([hole_position, hole_quat])

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    
class SlotInsertionEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_slot_insertion.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4

        self._slot_joint = self._mjcf_root.find('joint', 'slot_joint')
        self._stick_joint = self._mjcf_root.find('joint', 'stick_joint')

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [-0.05, 0.05]
        y_range = [0.1, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        slot_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        slot_quat = np.array([1, 0, 0, 0])


        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.08, 0.08]
        y_range = [-0.1, 0.0]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        stick_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        stick_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._slot_joint).qpos = np.concatenate([slot_position, slot_quat])
        self._physics.bind(self._stick_joint).qpos = np.concatenate([stick_position, stick_quat])

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        stick_touch_table = False
        stick_touch_slot = False
        pins_touch = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "stick" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1 == "stick" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "stick":
                stick_touch_table = True

            if geom1 == "stick" and geom2.startswith("slot-"):
                stick_touch_slot = True

            if geom1 == "pin-stick" and geom2 == "pin-slot":
                pins_touch = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not stick_touch_table): # grasp stick
            reward = 2
        if stick_touch_slot and (not stick_touch_table): # peg and socket touching
            reward = 3
        if pins_touch: # successful insertion
            reward = 4
        return reward
    

class SewNeedleEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_sew_needle.xml')
        super().__init__(xml, cameras)

        self.max_reward = 5

        self._needle_joint = self._mjcf_root.find('joint', 'needle_joint')
        self._wall_joint = self._mjcf_root.find('joint', 'wall_joint')

        self._threaded_needle = False

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [0.15, 0.2]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        needle_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        needle_quat = np.array([1, 0, 0, 0])


        peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        peg_quat = np.array([1, 0, 0, 0])

        x_range = [-0.025, 0.025]
        y_range = [-.025,0.1]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        wall_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        wall_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._needle_joint).qpos = np.concatenate([needle_position, needle_quat])
        self._physics.bind(self._wall_joint).qpos = np.concatenate([wall_position, wall_quat])

        self._physics.forward()

        self._threaded_needle = False


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        needle_touch_table = False
        needle_touch_wall = False
        pins_touch = False
        needle_touch_pin = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1 == "needle" and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1 == "needle" and geom2.startswith("left"):
                touch_left_gripper = True

            if geom1 == "table" and geom2 == "needle":
                needle_touch_table = True

            if geom1 == "needle" and geom2.startswith("wall-"):
                needle_touch_wall = True

            if geom1 == "pin-needle" and geom2 == "pin-wall":
                self._threaded_needle = True
                pins_touch = True

            if geom1 == "needle" and geom2 == "pin-wall":
                needle_touch_pin = True

        reward = 0
        if touch_right_gripper: # touch needle
            reward = 1
        if touch_right_gripper and (not needle_touch_table): # grasp needle
            reward = 2
        if needle_touch_wall and (not needle_touch_table): # peg and socket touching
            reward = 3
        if self._threaded_needle: # needle threaded
            reward = 4
        if touch_left_gripper and (not touch_right_gripper) and (not needle_touch_table) and (not needle_touch_pin) and self._threaded_needle: # grasped needle on other side
            reward = 5
        return reward
    

class TubeTransferEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_tube_transfer.xml')
        super().__init__(xml, cameras)

        self.max_reward = 3

        self._ball_joint = self._mjcf_root.find('joint', 'ball_joint')
        self._tube1_joint = self._mjcf_root.find('joint', 'tube1_joint')
        self._tube2_joint = self._mjcf_root.find('joint', 'tube2_joint')


    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [0.05, 0.1]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        ball_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        ball_quat = np.array([1, 0, 0, 0])
        tube1_position = ball_position
        tube1_quat = ball_quat

        x_range = [-.1, -0.05]
        y_range = [-0.05, 0.05]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        tube2_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        tube2_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._ball_joint).qpos = np.concatenate([ball_position, ball_quat])
        self._physics.bind(self._tube1_joint).qpos = np.concatenate([tube1_position, tube1_quat])
        self._physics.bind(self._tube2_joint).qpos = np.concatenate([tube2_position, tube2_quat])

        self._physics.forward()


        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    

    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        tube1_touch_table = False
        tube2_touch_table = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("tube1-") and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1.startswith("tube2-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2.startswith("tube1-"):
                tube1_touch_table = True

            if geom1 == "table" and geom2.startswith("tube2-"):
                tube2_touch_table = True

            if geom1 == "ball" and geom2 == "pin":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not tube1_touch_table) and (not tube2_touch_table): # grasp both
            reward = 2
        if pin_touched:
            reward = 3
        return reward
    

class HookPackageEnv(GuidedVisionEnv):
    def __init__(self, cameras):
        xml = os.path.join(XML_DIR, 'task_hook_package.xml')
        super().__init__(xml, cameras)

        self.max_reward = 4

        self._package_joint = self._mjcf_root.find('joint', 'package_joint')
        self._hook_joint = self._mjcf_root.find('joint', 'hook_joint')

    def reset(self, seed=None) -> tuple:
        super().reset(seed=seed)

        # reset physics
        x_range = [-0.1, 0.1]
        y_range = [.3, .3]
        z_range = [0.2, 0.3]
        ranges = np.vstack([x_range, y_range, z_range])
        hook_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        hook_quat = np.array([1, 0, 0, 0])

        x_range = [-.1, 0.1]
        y_range = [0, 0.15]
        z_range = [0.0, 0.0]
        ranges = np.vstack([x_range, y_range, z_range])
        package_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
        package_quat = np.array([1, 0, 0, 0]) 

        self._physics.bind(self._hook_joint).qpos = np.concatenate([hook_position, hook_quat])
        self._physics.bind(self._package_joint).qpos = np.concatenate([package_position, package_quat])

        self._physics.forward()

        observation = self.get_obs()
        info = "Resetting arms..."

        return observation, info
    
    def get_reward(self):

        touch_left_gripper = False
        touch_right_gripper = False
        package_touch_table = False
        package_touch_hook = False
        pin_touched = False

        # return whether peg touches the pin
        contact_pairs = []
        for i_contact in range(self._physics.data.ncon):
            id_geom_1 = self._physics.data.contact[i_contact].geom1
            id_geom_2 = self._physics.data.contact[i_contact].geom2
            geom1 = self._physics.model.id2name(id_geom_1, 'geom')
            geom2 = self._physics.model.id2name(id_geom_2, 'geom')
            contact_pairs.append((geom1, geom2))
            contact_pairs.append((geom2, geom1))

        for geom1, geom2 in contact_pairs:
            if geom1.startswith("package-") and geom2.startswith("right"):
                touch_right_gripper = True
            
            if geom1.startswith("package-") and geom2.startswith("left"): 
                touch_left_gripper = True

            if geom1 == "table" and geom2.startswith("package-"):
                package_touch_table = True

            if geom1 == "hook" and geom2.startswith("package-"):
                package_touch_hook = True

            if geom1 == "pin-package" and geom2 == "pin-hook":
                pin_touched = True

        reward = 0
        if touch_left_gripper and touch_right_gripper: # touch both
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not package_touch_table): # grasp both
            reward = 2
        if package_touch_hook and (not package_touch_table):
            reward = 3
        if pin_touched:
            reward = 4
        return reward


if __name__ == '__main__':
    # setup the environment
    env = make_sim_env('sim_tube_transfer', cameras=[])
    observation, info = env.reset(seed=42)

    init_action = np.concatenate([
        observation['poses']['left'],
        np.array([0.03]),
        observation['poses']['right'],
        np.array([0.03]),
        observation['poses']['middle'],
    ])
    action = init_action
    
    i = 0
    while True:
        step_start = time.time()

        # Take a step in the environment using the chosen action
        observation, reward, terminated, truncated, info = env.step(action)
        env.render_viewer()

        # Check if the episode is over (terminated) or max steps reached (truncated)
        if terminated or truncated:
            # If the episode ends or is truncated, reset the environment
            observation, info = env.reset()

        print("Step time:", time.time() - step_start)

        if i % 100 == 0:
            env.reset()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = SIM_DT - (time.time() - step_start)
        time.sleep(max(0, time_until_next_step))

        i += 1
