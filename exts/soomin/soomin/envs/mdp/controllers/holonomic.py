# type: ignore

from __future__ import annotations
from dataclasses import MISSING

import carb
import numpy as np
import torch

import osqp
from pxr import Gf
from scipy import sparse
from omni.isaac.core.utils.rotations import quat_to_rot_matrix, euler_angles_to_quat
from omni.isaac.lab.utils import configclass


@configclass
class HolnomicControllerCfg:
    """Configuration for mecanum-wheel holonomic drive controller."""

    wheel_radius: np.ndarray = MISSING
    
    wheel_positions: np.ndarray = MISSING
    
    wheel_orientations: np.ndarray = MISSING
    
    mecanum_angles: np.ndarray = MISSING
        
    wheel_axis: np.ndarray = np.array([1, 0, 0])    # default to x_axis
    
    up_axis: np.ndarray = np.array([0, 0, 1])       # default to z_axis
    
    max_linear_speed: float = 1.0e20
    
    max_angular_speed: float = 1.0e20
    
    max_wheel_speed: float = 1.0e20
    
    linear_gain: float = 1.0
    
    angular_gain: float = 1.0
    
##
#   Main Controller
##

class HolonomicController:
    """Holonomic drive control.

    This controller computes the joint drive commands required to produce the commanded forward, lateral, and yaw speeds of the robot.
    The problem is framed as a quadratic program to minimize the residual "net force" acting on the center of mass.
    
    Reference:
        [1] https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/node-library/nodes/omni-isaac-wheeled_robots/holonomiccontroller-1.html
    """
    
    def __init__(self, cfg: HolnomicControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create input command buffer
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)
        
        # build base
        self.num_wheels = len(self.cfg.wheel_positions)
        self.base_dir = np.zeros((3, self.num_wheels), dtype=float)
        self.wheel_dists_inv = np.zeros((3, self.num_wheels), dtype=float)

        for i in range(self.num_wheels):
            p_0 = self.cfg.wheel_positions[i]
            r_0 = quat_to_rot_matrix(self.cfg.wheel_orientations[i])

            joint_pose = np.zeros((4, 4))
            joint_pose[:3, :3] = r_0.T
            joint_pose[3, :3] = p_0
            joint_pose[3, 3] = 1

            mecanum_angle = self.cfg.mecanum_angles[i]
            mecanum_radius = self.cfg.wheel_radius[i]
            m_rot = Gf.Rotation(
                Gf.Quatf(
                    *euler_angles_to_quat(
                        Gf.Vec3d(*self.cfg.up_axis.tolist()) * mecanum_angle, degrees=True, extrinsic=True
                    )
                )
            )
            j_axis = Gf.Vec3f(
                m_rot.TransformDir(Gf.Matrix4f(joint_pose).TransformDir(Gf.Vec3d(*self.cfg.wheel_axis.tolist())))
            ).GetNormalized()

            self.base_dir[0, i] = j_axis[0] * mecanum_radius
            self.base_dir[1, i] = j_axis[1] * mecanum_radius
            for k in range(2):
                self.wheel_dists_inv[k, i] = p_0[k]

        self.P = sparse.csc_matrix(np.diag(self.cfg.wheel_radius) / np.linalg.norm(self.cfg.wheel_radius))
        self.b = sparse.csc_matrix(np.zeros((6, 1)))
        V = self.base_dir
        W = np.cross(V, self.wheel_dists_inv, axis=0)
        self.A = sparse.csc_matrix(np.concatenate((V, W), axis=0))
        self.l = np.array([0.0, 0.0, -np.inf, -np.inf, -np.inf, 0.0])
        self.u = np.array([0.0, 0.0, np.inf, np.inf, np.inf, 0.0])

        self.prob = osqp.OSQP()

        self.prob.setup(self.P, A=self.A, l=self.l, u=self.u, verbose=False)

        self.prob.solve()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        return 3        # (x = forward, y = lateral, z = yaw) speeds
 
    """
    Operations.
    """

    def set_command(self, command: torch.Tensor):
        """Set target base speed command.

        Args:
            command: The input command in shape (N, 3).
        """
        
        if command.shape[1] != 3:
            raise Exception("command should be of length 3, delta x,y, and angular velocity")
        
        self._command[:] = command
        

    def compute(self) -> torch.Tensor:
        """Computes the target joint/wheel velocity that will yield the desired base speed.

        Returns:
            The target joint/wheel velocity commands in shape (N, num_wheels).
        """
        
        self.joint_commands = torch.zeros((self.num_envs, self.num_wheels), dtype=torch.float32, device=self._device)
        zero_mask = self._command.abs().sum(dim=1) == 0
            
        # Linear and angular commands
        v = self._command[:, :2] * self.cfg.linear_gain  # (N, 2) -> forward and lateral speeds
        w = self._command[:, 2] * self.cfg.angular_gain  # (N,) -> yaw speeds
        
        # Normalize linear velocities if needed
        v_norm = torch.norm(v, dim=1, keepdim=True)  # (N, 1)
        v_normalized = torch.where(v_norm > 0, v / v_norm, v)  # Avoid division by zero
        
        # Clip linear and angular velocities to max limits
        v_clipped = torch.where(v_norm > self.cfg.max_linear_speed, v_normalized * self.cfg.max_linear_speed, v)
        w_clipped = torch.clamp(w, -self.cfg.max_angular_speed, self.cfg.max_angular_speed)

        # Update the problem for each environment
        for i in range(self.num_envs):
            if zero_mask[i]:
                continue
            
            self.l[0:2] = self.u[0:2] = (v_clipped[i] / self.cfg.max_linear_speed).cpu()
            self.l[-1] = self.u[-1] = (w_clipped[i] / self.cfg.max_linear_speed).cpu()

            self.prob.update(l=self.l, u=self.u)
            res = None
            try:
                res = self.prob.solve()
            except Exception as e:
                carb.log_error("error:", e)

            if res is not None:
                values = torch.tensor(res.x.reshape([res.x.shape[0]])) * self.cfg.max_linear_speed

                # Scale wheel speeds if they exceed max wheel speed
                max_value = torch.max(torch.abs(values))
                if max_value > self.cfg.max_wheel_speed:
                    scale = self.cfg.max_wheel_speed / max_value
                    values = values * scale

                self.joint_commands[i] = values

        return self.joint_commands