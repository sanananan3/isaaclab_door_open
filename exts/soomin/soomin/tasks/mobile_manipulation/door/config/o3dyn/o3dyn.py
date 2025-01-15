"""Configuration for the o3dyn mobile robot using dc motor model.

The following configurations are available:

* :obj:`O3DYN_CFG`: Fraunhofer IML's o3dyn model from https://git.openlogisticsfoundation.org/silicon-economy/simulation-model/o3dynsimmodel
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg


O3DYN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/3dynsimmodel/o3dyn.usd",
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
        # pos=(-0.8, 0.0, 0.0),
        # rot=(0.707, 0.0, 0.0, -0.707),
        joint_pos={
            # base
            "wheel_fl_joint": 0.0,
            "wheel_fr_joint": 0.0,
            "wheel_rl_joint": 0.0,
            "wheel_rr_joint": 0.0,
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
    },
)