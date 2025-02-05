# type: ignore

"""Common functions that can be used to activate certain terminations for door opening task."""

import torch

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv


def sucess_grasp_handle(env: ManagerBasedRLEnv, threshold: float = 0.015) -> torch.Tensor: # previous threshold: 0.01
    """Terminate when the robot's gripper reaching the door handle with the right pose.

    This function returns True if the distance of fingertips to the handle is small enough when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns False.
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

    # ===================================================================================
    # grasping = (lfinger_dist <= threshold) | (rfinger_dist <= threshold)
    grasping = (lfinger_dist <= threshold) & (rfinger_dist <= threshold)
    # ===================================================================================

    return is_graspable & grasping

def success_open_door(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, threshold: float = 0.26) -> torch.Tensor:
    """Terminate when the robot opened the door.
    
    This function returns True if the door joint position is over the threshold. Otherwise, it returns False.
    """
    door_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = sucess_grasp_handle(env)
    
    return torch.abs(is_graspable * door_pos) >= threshold

def fail_illegal_area(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), min_dist: float = 0.75, max_dist: float = 1.1
) -> torch.Tensor:
    """Terminate when the robot enters illegal area.
    
    """
    # get the (x, y) position of the robot and the handle
    robot_base_xy = env.scene[asset_cfg.name].data.body_pos_w[..., asset_cfg.body_ids[0], :2]
    handle_xy = env.scene["handle_frame"].data.target_pos_w[..., 0, :2]
    # calculate the Euclidean distance between the robot and the handle
    distance = torch.norm(robot_base_xy - handle_xy, dim=-1, p=2)
    # calculate the opposite handle (-x) direction distance
    opposite_dir_distance = handle_xy[:, 0] - robot_base_xy[:, 0]
    
    return (distance < min_dist) | (distance > max_dist) | (opposite_dir_distance > 0.2)