import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Franka RL agent door opening.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments of spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from soomin.tasks.mobile_manipulation.factory.config.franka.floating_pos_env_cfg import FrankaFactoryEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers import SceneEntityCfg

def main():
    # Initialize the simulation context
    env_cfg = FrankaFactoryEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
        
    # Play the simulator
    print("[INFO]: Setup complete...")
    
    # Simulate physics
    count = 0
    asset_cfg = SceneEntityCfg("robot", joint_names=["base_joint.*"])
    
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 120 == 0:
                joint_vel = env.scene[asset_cfg.name].data.joint_vel[0, :3] # type: ignore
                print(joint_vel)
            if count % 720 == 0:
                # reset
                env.reset()
            # apply base actions to the robot
            direction = (count // 120) % 3
            
            efforts = torch.zeros_like(env.action_manager.action)
            efforts[:, direction] = 1.0
            
            env.step(efforts)
            
            count += 1
    
    env.close()

if __name__ == "__main__":
    main()