import numpy as np

from omni.isaac.lab.utils import configclass

from .floating_franka import SUMMIT_FRANKA_PANDA_HIGH_PD_CFG
from soomin.envs import mdp
from soomin.tasks.mobile_manipulation.factory.factory_env_cfg import FactoryEnvCfg

@configclass
class FrankaFactoryEnvCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        
        self.actions.mobile_action = mdp.FloatingHolonomicActionCfg(
            asset_name="robot", 
            body_name="robot_base_link",
            x_joint_name="base_joint_x",
            y_joint_name="base_joint_y",
            yaw_joint_name="base_joint_z"
        )