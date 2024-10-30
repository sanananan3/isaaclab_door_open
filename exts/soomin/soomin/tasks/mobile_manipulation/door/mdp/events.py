# type: ignore

from __future__ import annotations

import torch

from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.managers import SceneEntityCfg

from omni.isaac.lab.envs import ManagerBasedEnv
    
def apply_door_external_torque(
    env: ManagerBasedEnv, 
    env_ids: torch.Tensor, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["door_link", "lever_link"])
):
    """Apply the external `spring` torques applied to the door bodies (door and lever handle).
        TODO: door hinge not applied
    
    """
    
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies
    
    # set forces and torques
    forces = torch.zeros((len(env_ids), num_bodies, 3), device=asset.device)
    torques = torch.zeros_like(forces)
    # torques[..., 0, :] = 10.0
    torques[..., 1, 2] = -10.0
    # set the forces and torques into the buffers
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)