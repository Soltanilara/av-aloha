
import numpy as np
import mujoco
from transform_utils import exp2mat, adjoint, within_pose_threshold, pose2mat, wxyz_to_xyzw
from numba import jit, prange

def create_fk_fn(physics, joints, eef_site):
    physics.bind(joints).qpos = np.zeros(len(joints))
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    w0 = physics.bind(joints).xaxis.copy()
    p0 = physics.bind(joints).xanchor.copy()
    v0 = -np.cross(w0, p0)
    site0 = np.eye(4)
    site0[:3, :3] = physics.bind(eef_site).xmat.reshape(3,3).copy()
    site0[:3, 3] = physics.bind(eef_site).xpos.copy()

    @jit(nopython=True, fastmath=True, cache=False)
    def forward_kinematics(theta):
        M = np.eye(4)
        M[:,:] = site0
        for i in prange(len(theta)-1, -1, -1):
            T = exp2mat(w0[i], v0[i], theta[i])
            M = np.dot(T, M)
        return M
    
    return forward_kinematics

def create_jac_fn(physics, joints):
    physics.bind(joints).qpos = np.zeros(len(joints))
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    w0 = physics.bind(joints).xaxis.copy()
    p0 = physics.bind(joints).xanchor.copy()
    v0 = -np.cross(w0, p0)

    @jit(nopython=True, fastmath=True, cache=False)
    def jacobian(theta):
        # screw axis at rest place
        S = np.hstack((w0, v0)) 
        J = np.zeros((6, len(theta)))
        Ts = np.eye(4)

        # compute each column of the Jacobian
        for i in prange(len(theta)):
            J[:, i] = adjoint(Ts) @ S[i,:]
            Ts = np.dot(Ts, exp2mat(w0[i], v0[i], theta[i]))

        # swap jacp and jacr
        J = J[np.array((3,4,5,0,1,2)),:]

        return J
    
    return jacobian

@jit(nopython=True, fastmath=True)
def safety(
    qpos, 
    ctrl, 
    Taction,
    fk_fn,
    joint_limits,
    xyz_bounds,
    joint_tracking_safety_margin,
    eef_pos_tracking_safety_margin,
    eef_rot_tracking_safety_margin,
):
    # check if difference of any qpos and ctrl is too large
    if np.any(np.abs(qpos - ctrl) > joint_tracking_safety_margin):
        return False, "Joint tracking safety margin exceeded"
    
    # check that not near joint limits
    if np.any(qpos < joint_limits[:,0]) or np.any(qpos > joint_limits[:,1]):
        return False, "Joint limit safety margin exceeded"
    
    # check that not near boundaries of workspace
    Tqpos = fk_fn(qpos)
    if np.any(Tqpos[:3, 3] < xyz_bounds[:,0]) or np.any(Tqpos[:3, 3] > xyz_bounds[:,1]):
        return False, "End effector position outside bounds"
    
    if Taction is not None:
        if np.any(Taction[:3,3] < xyz_bounds[:,0]) or np.any(Taction[:3,3] > xyz_bounds[:,1]):
            return False, "End effector action position outside bounds"
                    
        if not within_pose_threshold(
            Tqpos[:3, 3],
            Tqpos[:3, :3],
            Taction[:3, 3],
            Taction[:3, :3],
            eef_pos_tracking_safety_margin, 
            eef_rot_tracking_safety_margin):
            return False, "End effector pose tracking safety margin exceeded"

    return True, ""


# check that ctrl and qpos are close to each other
# check that not near joint limits
# check that not near boundaries of workspace
# check that action is reasonably close to current position
def create_safety_fn(
    physics,
    joints,
    eef_site,
    xyz_bounds,
    joint_limit_safety_margin=0.01,
    joint_tracking_safety_margin=1.0,
    eef_pos_tracking_safety_margin=0.2,
    eef_rot_tracking_safety_margin=3.0):
    
    joint_limit_safety_margin = joint_limit_safety_margin
    xyz_bounds = np.array(xyz_bounds)
    joint_tracking_safety_margin = joint_tracking_safety_margin
    eef_pos_tracking_safety_margin = eef_pos_tracking_safety_margin
    eef_rot_tracking_safety_margin = eef_rot_tracking_safety_margin
    fk_fn = create_fk_fn(physics, joints, eef_site)
    joint_limits = physics.bind(joints).range.copy()
    # check safety margin is not larger than joint limit half range
    assert np.all(joint_limit_safety_margin < (joint_limits[:,1] - joint_limits[:,0])/2)
    # add some safety margin to joint limits
    joint_limits[:,0] += joint_limit_safety_margin
    joint_limits[:,1] -= joint_limit_safety_margin

    def safety_fn(qpos, ctrl, Taction=None):
        return safety(
            qpos,
            ctrl,
            Taction,
            fk_fn,
            joint_limits,
            xyz_bounds,
            joint_tracking_safety_margin,
            eef_pos_tracking_safety_margin,
            eef_rot_tracking_safety_margin
        )
    
    return safety_fn



if __name__ == "__main__":
    from constants import XML_DIR, LEFT_JOINT_NAMES, LEFT_ACTUATOR_NAMES, LEFT_EEF_SITE
    from dm_control import mjcf
    from transform_utils import mat2quat
    import os

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    
    # set some random pose
    LEFT_ARM_POSE = np.array([1,0,0,-1,0,1])

    # setup mujoco 
    mjcf_root = mjcf.from_path(os.path.join(XML_DIR, 'aloha_sim.xml'))
    physics = mjcf.Physics.from_mjcf_model(mjcf_root) 
    left_joints = [mjcf_root.find('joint', name) for name in LEFT_JOINT_NAMES]
    left_actuators = [mjcf_root.find('actuator', name) for name in LEFT_ACTUATOR_NAMES]
    left_eef_site = mjcf_root.find('site', LEFT_EEF_SITE)
    left_eef_site_id = physics.bind(left_eef_site).element_id
    jnt_dof_ids = physics.bind(left_joints[:6]).dofadr

    # test forward kinematics in mujoco
    physics.bind(left_joints[:6]).qpos = LEFT_ARM_POSE[:6]
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    left_xpos = physics.data.site_xpos[left_eef_site_id]
    left_xmat = physics.data.site_xmat[left_eef_site_id].reshape(3,3)
    left_quat = mat2quat(left_xmat)
    print("Mujoco FK left arm pose: ", left_xpos, left_quat)

    # test forward kinematics in custom implementation
    fk_fn = create_fk_fn(physics, left_joints[:6], left_eef_site)
    theta = np.array(LEFT_ARM_POSE[:6]).copy()
    M = fk_fn(theta)
    print("PoE FK left arm pose: ", M[:3, 3], mat2quat(M[:3, :3]))

    # test jacobian in mujoco
    physics.bind(left_joints[:6]).qpos = LEFT_ARM_POSE[:6]
    mujoco.mj_kinematics(physics.model.ptr, physics.data.ptr)
    jac = np.zeros((6, physics.model.nv))
    mujoco.mj_jacBody(physics.model.ptr, physics.data.ptr, jac[:3], jac[3:], physics.bind(left_eef_site).bodyid)
    jac = jac[:,jnt_dof_ids]
    print("Mujoco Jacobian: ", jac)

    # test jacobian in custom implementation
    jac_fn = create_jac_fn(physics, left_joints[:6])
    theta = np.array(LEFT_ARM_POSE[:6]).copy()
    J = jac_fn(theta)
    print("PoE Jacobian: ", J)