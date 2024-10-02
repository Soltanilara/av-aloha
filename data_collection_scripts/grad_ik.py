import numpy as np
import numpy as np
from numba import jit, prange
from kinematics import create_fk_fn
from transform_utils import wxyz_to_xyzw, quat2mat, angular_error, within_pose_threshold, limit_pose


@jit(nopython=True, fastmath=True, cache=False)
def run_grad_ik(
    q_start,
    target_pos,
    target_quat,
    fk_fn,
    cost_fn,
    solution_fn,
    max_iterations,
    step_size,
    min_cost_delta,
    joint_limits,
    joint_p,
    max_pos_diff,
    max_rot_diff,
):    
    
    # fk to get current pose
    current_pose = fk_fn(q_start)
    current_xpos = current_pose[:3, 3]
    current_xmat = current_pose[:3, :3]

    # limit target pose
    target_xpos = target_pos
    target_xmat = quat2mat(wxyz_to_xyzw(target_quat))
    target_pos, target_xmat = limit_pose(
        current_xpos, 
        current_xmat, 
        target_xpos,
        target_xmat,
        max_pos_diff,
        max_rot_diff,
    )

    init_cost = cost_fn(q_start, q_start, target_pos, target_xmat)
    gradient = np.zeros(len(q_start))
    working = q_start.copy()
    local = q_start.copy()
    best = q_start.copy()
    local_cost = init_cost
    best_cost = init_cost
    
    previous_cost = 0.0
    for i in prange(max_iterations):
        count = len(local)

        for i in prange(count):
            working[i] = local[i] - step_size
            p1 = cost_fn(working, q_start, target_pos, target_xmat)

            working[i] = local[i] + step_size
            p3 = cost_fn(working, q_start, target_pos, target_xmat)

            working[i] = local[i]

            gradient[i] = p3 - p1

        sum_gradients = np.sum(np.abs(gradient)) + step_size
        f = step_size / sum_gradients
        gradient *= f

        working = local - gradient
        p1 = cost_fn(working, q_start, target_pos, target_xmat)

        working = local + gradient
        p3 = cost_fn(working, q_start, target_pos, target_xmat)
        p2 = 0.5 * (p1 + p3)

        cost_diff = 0.5 * (p3 - p1)
        joint_diff = p2 / cost_diff if np.isfinite(cost_diff) and cost_diff != 0.0 else 0.0

        working = local - gradient * joint_diff
        working = np.clip(working, joint_limits[:, 0], joint_limits[:, 1])

        local[:] = working
        local_cost = cost_fn(local, q_start, target_pos, target_xmat)

        if local_cost < best_cost:
            best[:] = local
            best_cost = local_cost

        if solution_fn(local, q_start, target_pos, target_xmat):
            break

        if abs(local_cost - previous_cost) <= min_cost_delta:
            break

        previous_cost = local_cost

    new_q = q_start + joint_p * (best - q_start)

    return new_q


class GradIK():
    def __init__(
        self, 
        physics, 
        joints,
        actuators,
        eef_site,
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
        joint_p = 0.1,
    ):
        self.physics = physics
        self.joints = joints
        self.actuators = actuators
        self.eef_site = eef_site
        self.fk_fn = create_fk_fn(physics, joints, eef_site)
        
        self.step_size = step_size
        self.min_cost_delta = min_cost_delta
        self.max_iterations = max_iterations

        self.num_joints = len(self.joints)
        self.joint_limits = self.physics.bind(self.joints).range.copy()
        self.joint_centers = 0.5 * (self.joint_limits[:, 0] + self.joint_limits[:, 1])
        self.half_ranges = 0.5 * (self.joint_limits[:, 1] - self.joint_limits[:, 0])
        
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.joint_center_weight = joint_center_weight / self.half_ranges
        self.joint_displacement_weight = joint_displacement_weight
        self.position_threshold = position_threshold
        self.rotation_threshold = rotation_threshold
        self.max_pos_diff = max_pos_diff
        self.max_rot_diff = max_rot_diff
        self.joint_p = joint_p

        self.cost_fn = self.make_cost_fn()
        self.solution_fn = self.make_solution_fn()

    def run(self, q, target_pos, target_quat):        
        # run grad ik
        return run_grad_ik(
            q,
            target_pos,
            target_quat,
            self.fk_fn,
            self.cost_fn,
            self.solution_fn,
            self.max_iterations,
            self.step_size,
            self.min_cost_delta,
            self.joint_limits,
            self.joint_p,
            self.max_pos_diff,
            self.max_rot_diff,
        )

    def make_cost_fn(self):
        fk_fn = self.fk_fn
        position_weight = self.position_weight
        rotation_weight = self.rotation_weight
        joint_center_weight = self.joint_center_weight
        joint_centers = self.joint_centers
        joint_displacement_weight = self.joint_displacement_weight

        @jit(nopython=True, fastmath=True, cache=False)
        def cost_fn(q, q_start, target_xpos, target_xmat): 
            current_pose = fk_fn(q)
            current_xpos = current_pose[:3, 3]
            current_xmat = current_pose[:3, :3]

            cost = 0.0

            # position cost
            cost += (position_weight * np.linalg.norm(target_xpos - current_xpos))**2

            # rotation cost
            cost += (rotation_weight * np.linalg.norm(angular_error(target_xmat, current_xmat)))**2

            # center joints cost
            cost += np.sum( (joint_center_weight * (q - joint_centers)) ** 2 )

            # minimal displacement cost
            cost += np.sum((joint_displacement_weight * (q - q_start))**2)

            return cost
        
        return cost_fn
    
    def make_solution_fn(self):
        fk_fn = self.fk_fn
        position_threshold = self.position_threshold
        rotation_threshold = self.rotation_threshold

        @jit(nopython=True, fastmath=True, cache=False)
        def solution_fn(q, q_start, target_xpos, target_xmat):
            current_pose = fk_fn(q)
            current_xpos = current_pose[:3, 3]
            current_xmat = current_pose[:3, :3]

            return within_pose_threshold(
                current_xpos, 
                current_xmat, 
                target_xpos, 
                target_xmat, 
                position_threshold, 
                rotation_threshold
            )
        
        return solution_fn


if __name__ == '__main__':
    from dm_control import mjcf
    from constants import XML_DIR, MIDDLE_ACTUATOR_NAMES, MIDDLE_ARM_POSE, MIDDLE_JOINT_NAMES, MIDDLE_EEF_SITE
    import mujoco.viewer
    import time
    import os
    from transform_utils import mat2quat, xyzw_to_wxyz

    MOCAP_NAME = "target"
    PHYSICS_DT=0.002
    DT = 0.04
    PHYSICS_ENV_STEP_RATIO = int(DT/PHYSICS_DT)
    DT = PHYSICS_DT * PHYSICS_ENV_STEP_RATIO

    xml_path = os.path.join(XML_DIR, f'single_arm.xml')
    mjcf_root = mjcf.from_path(xml_path)  
    mjcf_root.option.timestep = PHYSICS_DT  
    
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) 

    left_joints = [mjcf_root.find('joint', name) for name in MIDDLE_JOINT_NAMES]
    left_actuators = [mjcf_root.find('actuator', name) for name in MIDDLE_ACTUATOR_NAMES]
    left_eef_site = mjcf_root.find('site', MIDDLE_EEF_SITE)
    mocap = mjcf_root.find('body', MOCAP_NAME)

    # set up controllers
    left_controller = GradIK(
        physics=physics,
        joints = left_joints[:7],
        actuators=left_actuators[:7],
        eef_site=left_eef_site,
        step_size=0.0001, 
        min_cost_delta=1.0e-3, 
        max_iterations=20, 
        position_weight=500.0,
        rotation_weight=100.0,
        joint_center_weight=np.array([10.0, 10.0, 1.0, 50.0, 1.0, 1.0, 1.0]),
        joint_displacement_weight=np.array(7*[50.0]),
        position_threshold=0.001,
        rotation_threshold=0.001,
        max_pos_diff=0.1,
        max_rot_diff=0.3,
        joint_p = 0.9,
    )

    physics.bind(left_joints).qpos = MIDDLE_ARM_POSE
    physics.bind(left_actuators).ctrl = MIDDLE_ARM_POSE
    physics.bind(mocap).mocap_pos = physics.bind(left_eef_site).xpos
    physics.bind(mocap).mocap_quat = xyzw_to_wxyz(mat2quat(physics.bind(left_eef_site).xmat.reshape(3,3)))

    with mujoco.viewer.launch_passive(physics.model.ptr, physics.data.ptr) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mocap_pos = physics.bind(mocap).mocap_pos
            mocap_quat = physics.bind(mocap).mocap_quat
            start = time.time()
            physics.bind(left_actuators).ctrl = left_controller.run(physics.bind(left_joints).qpos, mocap_pos, mocap_quat)
            print("Time taken: ", time.time() - start)
            physics.step(nstep=PHYSICS_ENV_STEP_RATIO)
            viewer.sync()

            time_until_next_step = DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  