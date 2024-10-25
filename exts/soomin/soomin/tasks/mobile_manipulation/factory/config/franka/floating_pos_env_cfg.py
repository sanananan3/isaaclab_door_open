import numpy as np

from omni.isaac.lab.utils import configclass

from .floating_franka import SUMMIT_FRANKA_PANDA_HIGH_PD_CFG
from soomin.tasks.mobile_manipulation.factory import mdp
from soomin.tasks.mobile_manipulation.factory.factory_env_cfg import FactoryEnvCfg

@configclass
class FrankaFactoryEnvCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        
        self.actions.mobile_action = mdp.JointVelocityActionCfg(
            asset_name="robot", joint_names=["base_joint.*"], scale=1.
        )