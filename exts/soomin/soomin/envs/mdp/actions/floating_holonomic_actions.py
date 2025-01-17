from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class FloatingHolonomicAction(ActionTerm):

    cfg: actions_cfg.FloatingHolonomicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, 2)."""
    _offset: torch.Tensor
    """The offset applied to the input action. Shape is (1, 2)."""

    def __init__(self, cfg: actions_cfg.FloatingHolonomicActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse the joint information
        # -- x joint
        x_joint_id, x_joint_name = self._asset.find_joints(self.cfg.x_joint_name)
        if len(x_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the x joint name: {self.cfg.x_joint_name}, got {len(x_joint_id)}"
            )
        # -- y joint
        y_joint_id, y_joint_name = self._asset.find_joints(self.cfg.y_joint_name)
        if len(y_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the y joint name: {self.cfg.y_joint_name}")
        # -- yaw joint
        yaw_joint_id, yaw_joint_name = self._asset.find_joints(self.cfg.yaw_joint_name)
        if len(yaw_joint_id) != 1:
            raise ValueError(f"Found more than one joint match for the yaw joint name: {self.cfg.yaw_joint_name}")
        # parse the body index
        self._body_idx, self._body_name = self._asset.find_bodies(self.cfg.body_name)
        if len(self._body_idx) != 1:
            raise ValueError(f"Found more than one body match for the body name: {self.cfg.body_name}")

        # process into a list of joint ids
        self._joint_ids = [x_joint_id[0], y_joint_id[0], yaw_joint_id[0]]
        self._joint_names = [x_joint_name[0], y_joint_name[0], yaw_joint_name[0]]
        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, 3, device=self.device)

        # save the scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        # obtain current heading
        quat_w = self._asset.data.body_quat_w[:, self._body_idx[0]]
        yaw_w = euler_xyz_from_quat(quat_w)[2]
        # compute joint velocities targets
        self._joint_vel_command[:, 0] = torch.cos(yaw_w) * self.processed_actions[:, 0] - torch.sin(yaw_w) * self.processed_actions[:, 1]  # x
        self._joint_vel_command[:, 1] = torch.sin(yaw_w) * self.processed_actions[:, 0] + torch.cos(yaw_w) * self.processed_actions[:, 1]  # y
        self._joint_vel_command[:, 2] = self.processed_actions[:, 2]  # yaw
        # set the joint velocity targets
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0