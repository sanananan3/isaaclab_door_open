from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.managers.action_manager import ActionTerm

from soomin.envs.mdp.controllers.holonomic import HolonomicController

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class HolonomicAction(ActionTerm):
    
    cfg: actions_cfg.HolonomicActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input (base speed). Shape is (1, 3)."""
    _offset: torch.Tensor
    """The offset applied to the input (base speed). Shape is (1, 3)."""

    def __init__(self, cfg: actions_cfg.HolonomicActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # parse the joint information
        # -- front left wheel joint
        fl_joint_id, fl_joint_name = self._asset.find_joints(self.cfg.fl_joint_name)
        if len(fl_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the fl joint name: {self.cfg.fl_joint_name}, got {len(fl_joint_id)}"
            )
        # -- front right wheel joint
        fr_joint_id, fr_joint_name = self._asset.find_joints(self.cfg.fr_joint_name)
        if len(fr_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the fr joint name: {self.cfg.fr_joint_name}, got {len(fr_joint_id)}"
            )
        # -- rear left wheel joint
        rl_joint_id, rl_joint_name = self._asset.find_joints(self.cfg.rl_joint_name)
        if len(rl_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the fl joint name: {self.cfg.rl_joint_name}, got {len(rl_joint_id)}"
            )
        # -- rear right wheel joint
        rr_joint_id, rr_joint_name = self._asset.find_joints(self.cfg.rr_joint_name)
        if len(rr_joint_id) != 1:
            raise ValueError(
                f"Expected a single joint match for the fr joint name: {self.cfg.rr_joint_name}, got {len(rr_joint_id)}"
            )

        # process into a list of joint ids
        self._joint_ids = [fl_joint_id[0], fr_joint_id[0], rl_joint_id[0], rr_joint_id[0]]
        self._joint_names = [fl_joint_name[0], fr_joint_name[0], rl_joint_name[0], rr_joint_name[0]]
        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # create the holonomic drive controller
        self._holonomic_controller = HolonomicController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )
        
        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel_command = torch.zeros(self.num_envs, 4, device=self.device)     # [fl, fr, rl, rr]

        # save the scale and offset as tensors
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._holonomic_controller.action_dim

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
        # set command into controller
        self._holonomic_controller.set_command(self._processed_actions)

    def apply_actions(self):
        """Calculate wheel speeds given the desired signed vehicle speeds.

            processed_actions (torch.Tensor): (num_envs, 3) shape. 3 = [forward speed, lateral speed, yaw speed]
            _joint_vel_command (torch.Tensor): (num_envs, 4) shape. 4 = [fl, fr, rl, rr]
        """
        # NOT CALLED 

        # compute joint velocity targets
        self._joint_vel_command[:] = self._holonomic_controller.compute()
        # set the joint velocity targets
        self._asset.set_joint_velocity_target(self._joint_vel_command, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
