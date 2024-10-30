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
# Scene definition
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
            usd_path=f"/home/kist/Documents/Resources/door.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.4, -0.8, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0), # 180 deg rotation about z-axis
            joint_pos={
                "door_joint": 0.0,
            }
        ),
        actuators={
            "door_joint": ImplicitActuatorCfg(
                joint_names_expr=["door_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=2.5,
            ),
            "lever_joint": ImplicitActuatorCfg(
                joint_names_expr=["lever_joint"],
                effort_limit=0.0,
                velocity_limit=0.0,
                stiffness=500.0,
                damping=100.0,
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
                    pos=(0.09, 0.0, 0.0),
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
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    null_command = mdp.NullCommandCfg()
        
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING

    mobile_action: ActionTerm | None = None

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        root_pos = ObsTerm(func=mdp.root_pos_w)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
        handle_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["lever_joint"])},
        )
        handle_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["lever_joint"])},
        )
        rel_ee_door_distance = ObsTerm(func=mdp.rel_ee_door_distance)
        
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
            
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    
@configclass
class EventCfg:
    """Configuration for events."""
    
    # on reset
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    
    reset_root_pos = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.5, 0.5)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        },
    )

    
@configclass
class RewardsCfg:
    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=2.0)
    
    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0)
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": 0.04,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger_joint.*"]),
        },
    )
    
    # 3. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
    
    # 4. Success Bonus
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
    
    
@configclass
class TerminationsCfg:
    # Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Task Success
    # grasp_door = DoneTerm(func=mdp.sucess_grasp_handle)
    open_door = DoneTerm(
        func=mdp.success_open_door,
        params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"])}
    )

@configclass
class DoorEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the franka door opening environment."""
    
    # Scene settings
    scene: FrankaDoorSceneCfg = FrankaDoorSceneCfg(num_envs=4096, env_spacing=6.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriCulumCfg = CurriCulumCfg()
    
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 1.5
        self.viewer.origin_type = "env"
        self.viewer.eye = (3.0, 0.0, 2.5)
        self.viewer.lookat = (-0.5, -1.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 1000  # 1000Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        