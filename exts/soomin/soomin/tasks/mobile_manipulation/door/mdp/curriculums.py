# type: ignore

import torch
from collections.abc import Sequence

from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv

def door_joint_stiffness(env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """TODO: Adapt Door Joint's Stiffness to Lever Handle Status"""
    
    return torch.ones(env_ids, 1)