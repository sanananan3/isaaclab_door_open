# type: ignore

"""MDP rewards.

Reference: ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/mdp/rewards.py
"""

import torch
import torch.nn.functional as F

from .utils import *
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensor
from omni.isaac.lab.utils.math import matrix_from_quat
from omni.isaac.lab.envs import ManagerBasedRLEnv



def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward the robot for reaching the door handle using inverse-square law."""
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :] # world coordinate of the end-effector
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :] # world coordinate of the handle 

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    

    # print("[info] in approach_ee_handle, handle_pos[0][1] : ", handle_pos[0][1])
    

    # signed_disance_y = torch.sign(handle_pos[0][1] - ee_tcp_pos[0][1]-0.009) 

    direction_to_handle = handle_pos - ee_tcp_pos - 0.01

    door_normal = torch.tensor([0.0, 1.0, 0.0], device=handle_pos.device) # manually change the door normal vector (current: y-axis)

    collision_check = torch.sum(direction_to_handle * door_normal, dim=-1)

    is_valid = torch.logical_and(distance <= threshold, collision_check < 0) 

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2) # square -> more reward for being closer 

    # print("[info] In approach_ee_handle, distance : " ,distance  )
    return torch.where(is_valid, 2 * reward, reward/2)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.
        where `align_z` is the dot product of the z direction of the gripper and the -z direction of the handle
        and `align_x` is the dot product of the x direction of the gripper and the -x direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["handle_frame"].data.target_quat_w[..., 0, :]
    
    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)
    
    # get current x and y direction of the handle
    handle_x, handle_y, handle_z = handle_mat[..., 0],  handle_mat[..., 1] , handle_mat[..., 2]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_y, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 1] , ee_tcp_rot_mat[..., 2]
    
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_z.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_y = torch.bmm(ee_tcp_y.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1) 

    return (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2) * 0.5 +  0.2 * torch.sign(align_y) * align_y**2
    
    

def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Reward the robot's gripper reaching the door handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the one finger is above the handle and the other is below the handle). Otherwise, it returns zero.
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
    is_graspable = (rfinger_pos[:, 2] - handle_pos[:, 2]) * (lfinger_pos[:, 2] - handle_pos[:, 2]) < 0 # one finger above and the other beblow the handle 

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.
    
    (i.e., the one finger is above the handle and the other is below the handle).
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]
    
    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] - handle_pos[:, 2]) * (lfinger_pos[:, 2] - handle_pos[:, 2]) < 0
    
    return is_graspable




def grasp_handle(
    env: ManagerBasedRLEnv, upper_threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    
    # asset_cfg = fr3_finger joint.*

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

    is_valid = distance <= upper_threshold

    return is_valid * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)




def open_door_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the door given by the joint position of the door.

    The bonus is given when the door is open. If the grasp is around the handle, the bonus is doubled.
    """
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    return (is_graspable + 1.0) * torch.abs(door_pos)
    



def illegal_area(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_dist: float = 0.8, max_dist: float = 1.05
) -> torch.Tensor:
    """Penalize the agent if it enters illegal area.
    
    """
    # get the (x, y) position of the robot and the handle
    robot_base_xy = env.scene[asset_cfg.name].data.body_pos_w[..., asset_cfg.body_ids[0], :2]
    handle_xy = env.scene["handle_frame"].data.target_pos_w[..., 0, :2]
    # calculate the Euclidean distance between the robot and the handle
    distance = torch.norm(robot_base_xy - handle_xy, dim=-1, p=2)
    # calculate the opposite handle (-x) direction distance
    opposite_dir_distance = handle_xy[:, 0] - robot_base_xy[:, 0]

    penalty_below_min = torch.clamp(min_dist - distance, min=0.0)  # Positive if too close (too close to the door)
    penalty_above_max = torch.clamp(distance - max_dist, min=0.0)  # Positive if too far (outside workspace boundary)
    penalty_opposite_direction = torch.clamp(opposite_dir_distance - 0.15, min=0.0)
    
    return penalty_below_min + penalty_above_max + penalty_opposite_direction
    
    

"""
Penalty terms.
"""
    

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

"""
Contact Sensor.
"""

def _grasp_force(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] # shape = (N, 2, 3)

    contact_quat_l = contact_sensor.data.quat_w[:, 0] # (N, 4), quat -> quaternion rotation 
    contact_quat_r = contact_sensor.data.quat_w[:, 1] # (N, 4)

    lfinger_contact_forces = F.pad(contact_forces[:, 0], (1, 0)) # (N, 4)
    rfinger_contact_forces = F.pad(contact_forces[:, 1], (1, 0)) # (N, 4)

    # rotation: q^-1 * contact forces * q
    rfinger_contact_forces = multiply_quat(multiply_quat(inverse_quat(contact_quat_r), rfinger_contact_forces), contact_quat_r)
    lfinger_contact_forces = multiply_quat(multiply_quat(inverse_quat(contact_quat_l), lfinger_contact_forces), contact_quat_l)

    # `grasp`: grippers are on the opposite side & both have contact 
    is_graspable = align_grasp_around_handle(env)
    # print(f"right: must be positive      {rfinger_contact_forces[0, 2]}")
    # print(f"left: must be negative       {lfinger_contact_forces[0, 2]}")
    both_contacted = (lfinger_contact_forces[:, 2] < 0) & (rfinger_contact_forces[:, 2] > 0)
    
    # too strong is not allowed either
    lfinger_contact_forces[:, 2] = torch.clamp(lfinger_contact_forces[:, 2], min=-20.0)
    rfinger_contact_forces[:, 2] = torch.clamp(rfinger_contact_forces[:, 2], max=20.0)
    # add +y direction force for right gripper and -y direction force for left gripper
    force = rfinger_contact_forces[:, 2] - lfinger_contact_forces[:, 2]
    return torch.where(is_graspable & both_contacted, force, force * 0.1)
    
    
def rotate_lever_with_handle_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    lever_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    grasp_force = _grasp_force(env, sensor_cfg)
    
    # TODO: proportional to the duration of holding handle
    return grasp_force * 0.1 * (torch.abs(lever_pos) + 1.0)

def open_with_handle_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    grasp_force = _grasp_force(env, sensor_cfg)
    return grasp_force * 0.1 * (torch.abs(door_pos) + 1.0)


def contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] # shape = (N, M, 3)

    non_zero_forces = contact_forces.abs().sum(dim=2).sum(dim=1) > 0  # Shape (N,)
    return non_zero_forces
