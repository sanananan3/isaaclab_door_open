# Isaac Lab Extension Template for Mobile Manipulation: Door Opening Task

This repository provides a template for setting up a custom Isaac Lab extension for mobile manipulation, focusing on door-opening tasks. It integrates reinforcement learning (RL) with environments designed for robotics, as well as tools for simulation and control.

## Features

- **Mobile Manipulation Task**: Designed for a door-opening task using a mobile base and manipulator, such as the Summit-based Franka Panda robot.
- **Custom Isaac Lab Extension**: Built as an extension of Isaac Lab to allow for easy customization and rapid development.
- **Reinforcement Learning**: Includes scripts and environments for RL using the PPO algorithm from the RSL-RL library.
- **Simulation Environment**: Fully integrated with Isaac Lab for high-fidelity simulation of robotic tasks.

> 🗒️ **Note:**
>
> Due to current issues with the **mobile robot's wheels** and **gravity**, the focus has temporarily shifted to training a fixed robot to perform door grasping.
> 
> Once the mobility issues are resolved, the task will be extended to include the mobile robot navigating to and interacting with the door.



## Prerequisites

Before using this template, ensure the following dependencies are installed:

- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html#installing-isaac-lab)
- [Isaac Sim](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html#installing-isaac-sim)
- Python 3.8+
- NVIDIA GPU (for simulation and RL training)
- **USD files (Mobile Robot, Door with Lever)**: Download from [Google Drive](https://drive.google.com/drive/folders/1JjY9h0QxIDsz6-6uHCe5GD9paAyGRRSA?usp=sharing) and place them in the appropriate directory.
    - You should modify `usd_path` for [mobile robot](https://github.com/soom1017/isaaclab_door_open/blob/main/exts/soomin/soomin/tasks/mobile_manipulation/door/config/franka/summit_franka.py#L18) and [door](https://github.com/soom1017/isaaclab_door_open/blob/main/exts/soomin/soomin/tasks/mobile_manipulation/door/door_env_cfg.py#L56).

## Installation

1. Make sure Isaac Sim and Isaac Lab is installed and properly configured.

2. Prepare a python interpreter (**choose one option**):

    #### Option 1: Use the existing Python bundled with Isaac Lab
    As explained in the [documentation](https://isaac-sim.github.io/IsaacLab/source/setup/installation/binaries_installation.html#setting-up-the-conda-environment-optional), the executable `isaaclab.sh` fetches the bundled python. To execute Python scripts, use:
    ```shell
    path-to-isaac-lab/isaaclab.sh -p [.py]
    ```
    #### Option 2: Set up a new conda environment
    ```shell
    path-to-isaac-lab/isaaclab.sh --conda [env_name]    # to create conda env

    conda activate [env_name]
    isaaclab -i                                         # to install isaac lab extensions in conda env
    ```

2. Clone this repository and install the extension library:

    ```shell
    git clone https://github.com/soom1017/isaaclab_door_open.git

    cd exts/soomin
    python -m pip install -e .      # if option1, path-to-isaac-lab/isaaclab.sh -p -m pip install -e .
    ```


## Usage

### Training

To train the robot to grasp the door using the PPO algorithm from RSL-RL:

```shell
python scripts/rsl_rl/train.py --task Template-Isaac-Open-Door-Franka-v0 --num_envs 64
```

### Evaluation
To evaluate a pre-trained model:

```shell
python scripts/rsl_rl/play.py --task Template-Isaac-Open-Door-Franka-Play-v0 --num_envs 1
```

### Experiment Results

- **Tensorboard**: Monitor training progress with Tensorboard.

    ![tensorboard](https://github.com/user-attachments/assets/9e6d1392-b654-42d0-95a0-69d64cbe7356)

- **Play Video**: Showcase the robot interacting with the environment.

    https://github.com/user-attachments/assets/361a4c56-11c7-4ab6-afc2-3f1248816066

