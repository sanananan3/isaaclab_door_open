"""Configuration for RL environment of mobile manipulation.

Primary Work: Summit-based Franka emika robot's manipulation for 'door opening' task.

Reference: ~/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/cabinet/cabinet_env_cfg.py
"""

from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab_tasks.manager_based.manipulation.reach.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ActionTermCfg as ActionTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.utils import configclass


##
# Scene definition
##


@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for a default plane scene."""
    
    robot: ArticulationCfg = MISSING # type: ignore
    
    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, -0.105)),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    mobile_action: ActionTerm | None = None

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        
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
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.4, 0.4),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class FactoryEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the mobile test environment."""
    
    # Scene settings
    scene: FactorySceneCfg = FactorySceneCfg(num_envs=4096, env_spacing=6.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        # self.viewer.origin_type = "env"
        # self.viewer.eye = (3.0, 0.0, 2.5)
        # self.viewer.lookat = (-0.5, -1.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz for locomotion
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        