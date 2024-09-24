"""MDP observations.

Reference: ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/mdp/observations.py
"""

import torch

from omni.isaac.lab.sensors import FrameTransformerData
from omni.isaac.lab.envs import ManagerBasedRLEnv

def rel_ee_door_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    object_data: FrameTransformerData = env.scene["handle_frame"].data
    
    return object_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]