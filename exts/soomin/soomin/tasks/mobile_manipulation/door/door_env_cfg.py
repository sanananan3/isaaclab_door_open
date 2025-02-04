# type: ignore
"""Configuration for RL environment of mobile manipulation.

Primary Work: Summit-based Franka emika robot's manipulation for 'door opening' task.

Reference: ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/cabinet_env_cfg.py
"""

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import FrameTransformerCfg, ContactSensorCfg
from omni.isaac.lab.sensors.frame_transformer import OffsetCfg
from omni.isaac.lab.utils import configclass

from . import mdp

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy() 
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# 1. Scene definition
##

@configclass
class FrankaDoorSceneCfg(InteractiveSceneCfg):
    """Configuration for a franka door scene."""
    
    replicate_physics = False
    
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    contact_forces: ContactSensorCfg = MISSING
        
    # objects
    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/lee/Documents/Resources/door.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.4, -1.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0), # 180 deg rotation about z-axis
            joint_pos={
                "door_joint": 0.0,
                "lever_joint": 0.0,
            }
        ),
        actuators={
            "door_joint": ImplicitActuatorCfg(
                joint_names_expr=["door_joint"],
                effort_limit=0.0,
                velocity_limit=0.0,
                stiffness=0.0,
                damping=0.0,
            ),
            "lever_joint": ImplicitActuatorCfg(
                joint_names_expr=["lever_joint"],
                effort_limit=0.0,
                velocity_limit=0.0,
                stiffness=0.0,
                damping=0.0,
                armature=0.001,
                friction=0.3,
            ),
        }
    )
    
    handle_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/door/door/base_link",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/DoorFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/door/door/lever_link",
                name="door_handle",
                offset=OffsetCfg(
                    pos=(0.02, 0.0, 0.02),
                ),
            ),
        ],
    )
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
        
        
##
# 2. MDP settings (actions, obervations, event, rewards, terminations)
##


## 
# 2.1 Action Configurations
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

    mobile_action: ActionTerm | None = None


## 
# 2.2 Observation Configurations
##
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        door_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint", "lever_joint"])},
        )
        door_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint", "lever_joint"])},
        )
        rel_robot_door_distance = ObsTerm(
            func=mdp.rel_robot_door_distance,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["robot_base_link"])}
        )
        rel_ee_door_distance = ObsTerm(func=mdp.rel_ee_door_distance)
        
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    

## 
# 2.3 Event Configurations
##
@configclass
class EventCfg:
    """Configuration for events."""
    
    # on reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_root_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.4, 0.4)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        },
    )
    
    door_external_spring_force = EventTerm(
        func=mdp.apply_door_external_torque, 
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=["door_link", "lever_link"], preserve_order=True)
        }
    )

## 
# 2.4 Reward Configurations
##
@configclass
class RewardsCfg:
    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=4.0, params={"threshold": 0.2}) # threshold : hyperparameter
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=3.0)
    
    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0)
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=3.0)
    
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=3.0,
        params={
            "threshold": 0.06, # previous = 0.03
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger_joint.*"]),
        },
    )
    
    # 3. Mobile action
    illegal_area = RewTerm(
        func=mdp.illegal_area, 
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["robot_base_link"])
        }
    )
    
    # 4. Penalize actions for cosmetic reasons
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    # base_joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.01,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["base_joint.*"]),
    #     }
    # )
    
    # 5. Success Bonus
    rotate_handle_bonus = RewTerm(
        func=mdp.rotate_lever_with_handle_contact,
        weight=4.5,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["lever_joint"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*finger")
        }
    )
    open_door_bonus = RewTerm(
        func=mdp.open_door_bonus,
        weight=3.5,
        params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"])}
    )
    open_handle_contact = RewTerm(
        func=mdp.open_with_handle_contact,
        weight=4.5,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*finger")
        }
    )
    
@configclass
class CurriCulumCfg:
    # door_stiffness = CurrTerm(func=mdp.door_joint_stiffness)
    pass
    
## 
# 2.5 Termination Configurations
##

@configclass
class TerminationsCfg:
    
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    print("[INFO] Termination Criteria : Time out ", time_out)

    # Task Success
    # grasp_door = DoneTerm(func=mdp.sucess_grasp_handle)
    open_door = DoneTerm(
        func=mdp.success_open_door,
        params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"])}
    )
    illegal_area = DoneTerm(
        func=mdp.fail_illegal_area,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["robot_base_link"])}
    )


##
# 3. Tying up the configurations (scene + MDP)
##

@configclass
class DoorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the franka door opening environment."""
    
    # Scene settings
    scene: FrankaDoorSceneCfg = FrankaDoorSceneCfg(num_envs=4096, env_spacing=6.5)
    # Basic settings (base terms)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings (RL terms)
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriCulumCfg = CurriCulumCfg()
    
    
    def __post_init__(self):
        """Post initialization."""
        print("[DEBUG] check for DoorEnvCfg post init")
        # general settings

        # ============= termination criteria ============
        self.decimation = 2
        self.episode_length_s = 8 # 6 -> 8
        # ===============================================

        self.viewer.origin_type = "env"
        self.viewer.eye = (3.0, 0.0, 2.5)
        self.viewer.lookat = (-0.5, -1.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 1000 # 1000Hz -> change to 60 hz for testing , manupulation

        print("[INFO] dt Setting for simulation ", self.sim.dt )

        
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        