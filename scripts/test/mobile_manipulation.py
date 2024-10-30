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

from soomin.tasks.mobile_manipulation.door.config.franka.joint_pos_env_cfg import FrankaDoorEnvCfg
from omni.isaac.lab.envs import ManagerBasedRLEnv

def main():
    # Initialize the simulation context
    env_cfg = FrankaDoorEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
        
    # Play the simulator
    print("[INFO]: Setup complete...")
    
    # Simulate physics
    count = 0
    
    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 1000 == 0:
                env.reset()
            
            efforts = torch.zeros_like(env.action_manager.action)
            env.step(efforts)
            
            count += 1
    
    env.close()

if __name__ == "__main__":
    main()