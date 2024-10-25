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

from soomin.tasks.mobile_manipulation.factory.config.o3dyn.holonomic_vel_env_cfg import O3DynFactoryEnvCfg
from omni.isaac.lab.envs import ManagerBasedEnv

def main():
    # Initialize the simulation context
    env_cfg = O3DynFactoryEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedEnv(cfg=env_cfg)
        
    # Play the simulator
    print("[INFO]: Setup complete...")
    
    # Simulate physics
    count = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            if count == 0:
                start_pos = env.scene["robot"].data.root_state_w[0, 1]
            elif count % 120 == 0:
                reset_pos = env.scene["robot"].data.root_state_w[0, 1]
                # validate holonomic controller's result: base velocity in y axis should be 2 (error range = 0.2)
                assert torch.abs(reset_pos - start_pos - 2) < 0.2
                start_pos = reset_pos
                # reset
                count = 0
                env.reset()
                
            # apply base actions to the robot
            efforts = torch.zeros_like(env.action_manager.action)
            efforts[:, 1] = 1.0
            
            env.step(efforts)
            
            count += 1
    
    env.close()

if __name__ == "__main__":
    main()