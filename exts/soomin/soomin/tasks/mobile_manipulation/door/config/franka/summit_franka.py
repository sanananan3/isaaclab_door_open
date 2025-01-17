"""Configuration for the Summit-Manipulation robots.

The following configurations are available:

* :obj:`SUMMIT_FRANKA_PANDA_CFG`: Summit base with Franka Emika arm
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

##
# Configuration
#   - Joint position's unit: meter (prismatic) or radian (revolute)
##

SUMMIT_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/lee/Documents/Resources/summit_franka.usd", # 얘는 사실 상 안 쓰이는듯
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.8, 0.0, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),
        joint_pos={
            # base
            "robot_front_left_wheel_joint": 0.0,
            "robot_front_right_wheel_joint": 0.0,
            "robot_back_left_wheel_joint": 0.0,
            "robot_back_right_wheel_joint": 0.0,
            # franka arm
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            # tool
            "fr3_finger_joint.*": 0.04,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base": ImplicitActuatorCfg(
            joint_names_expr=["robot_.*_wheel_joint"],
            velocity_limit=100.0,
            effort_limit=1000.0,
            stiffness=0.0,
            damping=1e5,
        ),
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=100.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0
)
"""Configuration of Franka arm with Franka Hand on a Summit base using implicit actuator models.

The following control configuration is used:

* Base: velocity control
* Arm: position control with damping
* Hand: position control with damping

"""

SUMMIT_FRANKA_PANDA_HIGH_PD_CFG = SUMMIT_FRANKA_PANDA_CFG.copy() # type: ignore
SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.actuators["fr3_shoulder"].stiffness = 400.0
SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.actuators["fr3_shoulder"].damping = 80.0
SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.actuators["fr3_forearm"].stiffness = 400.0
SUMMIT_FRANKA_PANDA_HIGH_PD_CFG.actuators["fr3_forearm"].damping = 80.0
