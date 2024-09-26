"""Common functions that can be used to activate certain terminations for door opening task."""

import torch

from omni.isaac.lab.envs import ManagerBasedRLEnv


def sucess_grasp_handle(env: ManagerBasedRLEnv, offset: float = 0.04, threshold: float = 0.01) -> torch.Tensor:
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
    grasping = (offset - lfinger_dist <= threshold) & (offset - rfinger_dist <= threshold)

    return is_graspable & grasping