# type: ignore

import torch

from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab_assets import VELODYNE_VLP_16_RAYCASTER_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

from .summit_franka import SUMMIT_FRANKA_PANDA_HIGH_PD_CFG as SUMMIT_FRANKA_CFG
from .floating_franka import SUMMIT_FRANKA_PANDA_HIGH_PD_CFG as FLOATING_FRANKA_CFG
from soomin.tasks.mobile_manipulation.door import mdp
from soomin.tasks.mobile_manipulation.door.door_env_cfg import DoorEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv



# ==============================================================================================================================
        
def update_gripper_action (env: ManagerBasedRLEnv): 
    """
        In single PPO, gripper's training is not working properly. Then, tried to implememt the gripper's action manually. 
        This function is called in the step function of the environment. (play & train)
          
        type(env.cfg) = FrankaDoorEnvCfg / type(env) = ManagerBasedRLEnv
    """
    # for distacnce calculation 
        
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :] # world coordinate of the end-effector
    handle_pos = env.scene["handle_frame"].data.target_pos_w[..., 0, :] # world coordinate of the handle 

    distance = torch.norm(ee_tcp_pos- handle_pos, dim = -1, p = 2)
   #  print("distance :" , distance)

    # for alignment calculation
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    graspable = (rfinger_pos[:, 2] - handle_pos[:, 2]) * (lfinger_pos[:, 2] - handle_pos[:, 2]) < 0 # one finger above and the other beblow the handle 

    is_valid = torch.logical_and(distance<=0.04 , graspable)

    gripper_action = env.action_manager.get_term("gripper_action")

   #  print("joint distance :", env.scene["robot"].data.joint_pos[:, gripper_action._joint_ids])

    for i in range(distance.shape[0]):

        if not hasattr(env, "gripper_locked"):
            env.gripper_locked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


        if is_valid[i] or env.gripper_locked[i]:

            gripper_action._open_command[:] = 0.0
            gripper_action._close_command[:] = 0.0

            env.scene.write_data_to_sim()

            env.gripper_locked[i] = True  
        
    
def reset_gripper_locked(env: ManagerBasedRLEnv) : 
    """
    When episode is terminated, reset the gripper_locked attribute 
    """

    if hasattr(env, "gripper_locked"):
        env.gripper_locked[:] = False
        print("[INFO] IN reset_gripper_locked , gripper_locked attribute is reset to False.")

# ==============================================================================================================================


@configclass
class FrankaDoorEnvCfg(DoorEnvCfg):
    def __post_init__(self):

        # call post_init function of parent 
        super().__post_init__()
        
        self.scene.robot = FLOATING_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ee = end-effector 
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )
        
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*finger",
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ContactFrameTransformer"),
            track_pose=True
        )
        
        self.actions.arm_action= mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["fr3_joint.*"], scale=1.
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot", 
            joint_names=["fr3_finger_joint.*"],
            open_command_expr=  {"fr3_finger_joint.*": 0.04},
            close_command_expr={"fr3_finger_joint.*": 0.04},
        )
        self.actions.mobile_action = mdp.FloatingHolonomicActionCfg(
            asset_name="robot", 
            body_name="robot_base_link",
            x_joint_name="base_joint_x",
            y_joint_name="base_joint_y",
            yaw_joint_name="base_joint_z"
        )

        self.update_gripper_action = update_gripper_action
        self.reset_gripper_locked = reset_gripper_locked

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