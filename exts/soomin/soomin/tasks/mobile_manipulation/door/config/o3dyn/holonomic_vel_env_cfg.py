# type: ignore

from omni.isaac.lab.utils import configclass

from .o3dyn import O3DYN_CFG
from soomin.tasks.mobile_manipulation.door import mdp
from soomin.tasks.mobile_manipulation.door.door_env_cfg import DoorEnvCfg


@configclass
class FrankaDoorEnvCfg(DoorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = O3DYN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        self.actions.arm_action= mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["fr3_joint.*"], scale=1.
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot", 
            joint_names=["fr3_finger_joint.*"],
            open_command_expr={"fr3_finger_joint.*": 0.04},
            close_command_expr={"fr3_finger_joint.*": 0.0},
        )
        
@configclass
class FrankaDoorEnvCfg_PLAY(FrankaDoorEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False