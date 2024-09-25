from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_assets import VELODYNE_VLP_16_RAYCASTER_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() # type: ignore
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

from .summit_franka import SUMMIT_FRANKA_PANDA_CFG, SUMMIT_FRANKA_PANDA_HIGH_PD_CFG
from soomin.tasks.mobile_manipulation.door import mdp
from soomin.tasks.mobile_manipulation.door.door_env_cfg import DoorEnvCfg


@configclass
class FrankaDoorEnvCfg(DoorEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore
        
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/summit_xls_lidar_front/fr3_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/summit_xls_lidar_front/fr3_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/summit_xls_lidar_front/fr3_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/summit_xls_lidar_front/fr3_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        
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