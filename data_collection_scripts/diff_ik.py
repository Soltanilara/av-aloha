import numpy as np
from transform_utils import angular_error, wxyz_to_xyzw, quat2mat
from kinematics import create_fk_fn, create_safety_fn, create_jac_fn
from numba import jit
import mujoco

class DiffIK():
    def __init__(
        self, 
        physics, 
        joints,
        actuators,
        eef_site,
        k_pos,
        k_ori,
        damping,
        k_null,
        q0,
        max_angvel,
        integration_dt,
        iterations,
    ):
        self.physics = physics
        self.joints = joints
        self.actuators = actuators
        self.eef_site = eef_site
        self.k_pos = k_pos
        self.k_ori = k_ori
        self.damping = damping
        self.k_null = k_null
        self.q0 = q0
        self.max_angvel = max_angvel
        self.integration_dt = integration_dt
        self.iterations = iterations

        self.diff_ik_fn = self.make_diff_ik_fn()

    def make_diff_ik_fn(self):
        integration_dt = self.integration_dt
        k_pos = self.k_pos
        k_ori = self.k_ori
        fk_fn = create_fk_fn(self.physics, self.joints, self.eef_site)
        jac_fn = create_jac_fn(self.physics, self.joints)
        diag = np.ascontiguousarray(self.damping * np.eye(6))
        eye = np.ascontiguousarray(np.eye(len(self.joints)))
        k_null = self.k_null
        q0 = self.q0
        max_angvel = self.max_angvel
        joint_limits = self.physics.bind(self.joints).range.copy()

        @jit(nopython=True, fastmath=True, cache=False)
        def diff_ik(q, target_pos, target_quat, iterations=1):
            for _ in range(iterations):
                current_pose = fk_fn(q)
                current_pos = current_pose[:3, 3]
                current_mat = current_pose[:3, :3]

                target_pos = target_pos
                target_mat = quat2mat(wxyz_to_xyzw(target_quat))

                twist = np.zeros(6)
                dx = target_pos - current_pos
                twist[:3] = k_pos * dx / integration_dt
                dr = angular_error(target_mat, current_mat)
                twist[3:] = k_ori *dr / integration_dt

                # Jacobian.
                jac = jac_fn(q)

                # Damped least squares.
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, twist)

                # Null space control.
                dq += (eye - np.linalg.pinv(jac) @ jac) @ (k_null * (q0 - q))

                # Limit joint velocity.
                dq = np.clip(dq, -max_angvel, max_angvel)

                # integrate
                q = q + dq * integration_dt

                # Limit joint position.
                q = np.clip(q, joint_limits[:,0], joint_limits[:,1])
            
            return q
        
        return diff_ik

    def run(self, q, target_pos, target_quat):
        return self.diff_ik_fn(q, target_pos, target_quat, iterations=self.iterations)

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
    left_controller = DiffIK(
        physics=physics,
        joints = left_joints[:7],
        actuators=left_actuators[:7],
        eef_site=left_eef_site,
        k_pos=0.3,
        k_ori=0.3,
        damping=1.0e-4,
        k_null=np.array([20.0, 10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
        q0=np.array(MIDDLE_ARM_POSE[:7]),
        max_angvel=3.14,
        integration_dt=DT,
        iterations=10
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