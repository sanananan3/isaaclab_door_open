import numpy as np

from omni.isaac.lab.utils import configclass

from .o3dyn import O3DYN_CFG
from soomin.envs.mdp.actions.actions_cfg import HolonomicActionCfg
from soomin.envs.mdp.controllers.holonomic import HolnomicControllerCfg
from soomin.tasks.mobile_manipulation.factory.factory_env_cfg import FactoryEnvCfg

@configclass
class O3DynFactoryEnvCfg(FactoryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = O3DYN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        
        self.actions.mobile_action = HolonomicActionCfg(
            asset_name="robot", 
            fl_joint_name="wheel_fl_joint", 
            fr_joint_name="wheel_fr_joint",
            rl_joint_name="wheel_rl_joint",
            rr_joint_name="wheel_rr_joint",
            scale=(2.34, 2.8, -3.75),
            controller=HolnomicControllerCfg(
                wheel_radius=np.array([0.129, 0.129, 0.129, 0.129]),
                wheel_positions=np.array([
                    [0.7625, 0.5619, 0.0554],
                    [0.7721, -0.5480, 0.0558],
                    [-0.7719, 0.5480, 0.0576],
                    [-0.7623, -0.5619, 0.0581]
                ]),
                wheel_orientations=np.array([
                    [0.5, -0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5],
                    [0.5, -0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5, 0.5]
                ]),
                mecanum_angles=np.array([-135, -45, -45, -135])
            )
        )