"""MDP rewards.

Reference: ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/mdp/rewards.py
"""

import torch
import torch.nn.functional as F

from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import matrix_from_quat
from omni.isaac.lab.envs import ManagerBasedRLEnv

def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    r"""Reward the robot for reaching the door handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)

def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the y direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -x direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]
    
    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)
    
    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]
    
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)
    
    

def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Reward the robot's gripper reaching the door handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))

def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]
    
    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])
    
    return is_graspable

def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)
    
def open_door_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the door given by the joint position of the door.

    The bonus is given when the door is open. If the grasp is around the handle, the bonus is doubled.
    """
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]] # type: ignore
    is_graspable = align_grasp_around_handle(env).float()

    return (is_graspable + 0.1) * torch.abs(door_pos)

"""
Contact Sensor.
"""
def multiply_quat(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    # q and r are tensors of shape (N, 4) representing quaternions
    w1, x1, y1, z1 = q.unbind(dim=-1)
    w2, x2, y2, z2 = r.unbind(dim=-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    
    return torch.stack([w, x, y, z], dim=-1)

def inverse_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def open_with_handle_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the door given by the joint position of the door.

    The bonus is given when the door is open. If the gripper is contact with the handle, the bonus is doubled.
    """
    
    # ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :] # shape = (N, 4)
    # w, x, y, z = ee_tcp_quat.unbind(dim=-1)
    # ee_tcp_quat_inv = torch.stack([w, -x, -y, -z], dim=-1)
    
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]] # type: ignore
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] # (N, 2, 3)
    
    contact_quat_r = contact_sensor.data.quat_w[:, 0] # (N, 4)
    contact_quat_l = contact_sensor.data.quat_w[:, 1] # (N, 4)
    
    rfinger_contact_forces = F.pad(contact_forces[:, 0], (1, 0)) # shape = (N, 4)
    lfinger_contact_forces = F.pad(contact_forces[:, 1], (1, 0)) # shape = (N, 4)
    
    # rotation: q^-1 * contact forces * q
    rfinger_contact_forces = multiply_quat(multiply_quat(inverse_quat(contact_quat_r), rfinger_contact_forces), contact_quat_r)
    lfinger_contact_forces = multiply_quat(multiply_quat(inverse_quat(contact_quat_l), lfinger_contact_forces), contact_quat_l)
    
    
    print(f"right_contact_forces: {rfinger_contact_forces[:, 1:]}")
    print(f"left_contact_forces: {-1 * lfinger_contact_forces[:, 1:]}")
    return align_grasp_around_handle(env)
    
    
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:asset_cfg.joint_ids will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

