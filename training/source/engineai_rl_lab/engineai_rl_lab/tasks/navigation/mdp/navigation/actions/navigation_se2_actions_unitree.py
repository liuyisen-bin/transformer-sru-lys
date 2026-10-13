"""G1 navigation action adapter for Unitree's official velocity policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm

from engineai_rl_lab.tasks.navigation.g1.unitree_velocity_policy import (
    UNITREE_POLICY_HISTORY_LENGTH,
    UNITREE_POLICY_INPUT_DIM,
    UNITREE_POLICY_OUTPUT_DIM,
    UnitreeVelocityPolicyBatch,
)

if TYPE_CHECKING:
    from .navigation_se2_actions_unitree_cfg import UnitreeNavigationSE2ActionCfg


UNITREE_SDK_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Exact joint_ids_map from Unitree's velocity/v0 deploy.yaml.
UNITREE_POLICY_TO_SDK = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11,
    17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
UNITREE_POLICY_JOINT_NAMES = [UNITREE_SDK_JOINT_NAMES[index] for index in UNITREE_POLICY_TO_SDK]
UNITREE_POLICY_DEFAULT_JOINT_POS = [
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3,
    0.3, -0.2, -0.2, 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15,
    -0.15, 0.0, 0.0, 0.0, 0.0,
]


class UnitreeNavigationSE2Action(ActionTerm):
    """Map the existing three-dimensional navigation action to 29 joint targets."""

    cfg: UnitreeNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    _TERM_DIMS = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos_rel": 29,
        "joint_vel_rel": 29,
        "last_action": 29,
    }

    def __init__(self, cfg: UnitreeNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._joint_ids, resolved_joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        if resolved_joint_names != cfg.joint_names or len(self._joint_ids) != 29:
            raise ValueError(
                "G1 joint order does not match Unitree velocity/v0. "
                f"Expected {cfg.joint_names}, resolved {resolved_joint_names}."
            )

        self.low_level_position_action_term: ActionTerm = cfg.low_level_position_action.class_type(
            cfg.low_level_position_action, env
        )
        if self.low_level_position_action_term.action_dim != UNITREE_POLICY_OUTPUT_DIM:
            raise ValueError("Unitree low-level position action must contain 29 joints.")

        self._action_dim = 3
        self._init_buffers()
        self._policy = UnitreeVelocityPolicyBatch(cfg.policy_file, self.device)

        actual_default = self._asset.data.default_joint_pos[0, self._joint_ids]
        expected_default = torch.tensor(
            UNITREE_POLICY_DEFAULT_JOINT_POS,
            device=self.device,
            dtype=actual_default.dtype,
        )
        if not torch.allclose(actual_default, expected_default, atol=1.0e-5, rtol=0.0):
            max_error = torch.max(torch.abs(actual_default - expected_default)).item()
            raise ValueError(
                "G1 default pose differs from Unitree deploy.yaml; "
                f"maximum joint error is {max_error:.6f} rad."
            )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._velocity_commands

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        return self._velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        return self._per_env_per_dim_low_pass_alpha

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_policy_actions

    @property
    def low_level_position_actions(self) -> torch.Tensor:
        return self._low_level_policy_actions

    @property
    def prev_low_level_position_actions(self) -> torch.Tensor:
        return self._previous_low_level_policy_actions

    @property
    def active_joint_ids(self) -> list[int]:
        return self._joint_ids

    def process_actions(self, actions: torch.Tensor):
        """Map the existing 3-D action interface to configured physical velocity limits."""

        if actions.shape != self._raw_navigation_actions.shape:
            raise ValueError(
                f"G1 navigation actions must be {tuple(self._raw_navigation_actions.shape)}, "
                f"got {tuple(actions.shape)}."
            )
        self._raw_navigation_actions.copy_(actions)
        normalized = torch.tanh(actions)
        normalized = ((normalized + self._policy_bias) * self._policy_scaling).clamp(-1.0, 1.0)
        commands = torch.where(
            normalized >= 0.0,
            normalized * self._command_max,
            normalized * torch.abs(self._command_min),
        )
        if self.cfg.enable_low_pass_filter:
            alpha = self._per_env_per_dim_low_pass_alpha
            commands = alpha * self._velocity_commands + (1.0 - alpha) * commands
        self._velocity_commands.copy_(commands)

    @torch.inference_mode()
    def apply_actions(self):
        """Run Unitree at 50 Hz and hold its target for four physics ticks."""

        update_ids = torch.nonzero(self._low_level_counter == 0, as_tuple=False).flatten()
        if update_ids.numel() > 0:
            self._previous_low_level_policy_actions[update_ids] = self._low_level_policy_actions[
                update_ids
            ]
            observations = self._build_policy_observations(update_ids)
            policy_actions = self._policy(observations).to(self._low_level_policy_actions.dtype)
            if policy_actions.shape != (update_ids.numel(), UNITREE_POLICY_OUTPUT_DIM):
                raise RuntimeError(
                    f"Unitree policy returned {tuple(policy_actions.shape)}, "
                    f"expected {(update_ids.numel(), UNITREE_POLICY_OUTPUT_DIM)}."
                )
            self._low_level_policy_actions[update_ids] = policy_actions
            self._last_policy_action[update_ids] = policy_actions
            self.low_level_position_action_term.process_actions(self._low_level_policy_actions)

        self.low_level_position_action_term.apply_actions()
        self._low_level_counter.add_(1).remainder_(self.cfg.low_level_decimation)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._raw_navigation_actions[env_ids] = 0.0
        self._velocity_commands[env_ids] = 0.0
        self._low_level_policy_actions[env_ids] = 0.0
        self._previous_low_level_policy_actions[env_ids] = 0.0
        self._last_policy_action[env_ids] = 0.0
        self._history_initialized[env_ids] = False
        for history in self._term_history.values():
            history[env_ids] = 0.0
        self._low_level_counter[env_ids] = 0
        self.low_level_position_action_term.reset(env_ids)

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        self._velocity_commands[env_ids] = 0.0

    def _init_buffers(self):
        self._raw_navigation_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity_commands = torch.zeros_like(self._raw_navigation_actions)
        self._policy_scaling = torch.tensor(
            self.cfg.policy_scaling, device=self.device
        ).expand(self.num_envs, -1).clone()
        self._policy_bias = torch.zeros_like(self._raw_navigation_actions)
        filter_alpha = torch.tensor(self.cfg.low_pass_filter_alpha, device=self.device)
        if filter_alpha.shape != (3,) or not torch.all(
            torch.isfinite(filter_alpha) & (filter_alpha >= 0.0) & (filter_alpha < 1.0)
        ):
            raise ValueError("low_pass_filter_alpha must contain [vx, vy, wz] values in [0, 1).")
        self._per_env_per_dim_low_pass_alpha = filter_alpha.expand(self.num_envs, -1).clone()
        self._command_min = torch.tensor(self.cfg.command_min, device=self.device)
        self._command_max = torch.tensor(self.cfg.command_max, device=self.device)
        self._low_level_policy_actions = torch.zeros(
            self.num_envs, UNITREE_POLICY_OUTPUT_DIM, device=self.device
        )
        self._previous_low_level_policy_actions = torch.zeros_like(
            self._low_level_policy_actions
        )
        self._last_policy_action = torch.zeros_like(self._low_level_policy_actions)
        self._low_level_counter = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._history_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._term_history = {
            name: torch.zeros(
                self.num_envs,
                UNITREE_POLICY_HISTORY_LENGTH,
                dimension,
                device=self.device,
            )
            for name, dimension in self._TERM_DIMS.items()
        }

    def _current_terms(self, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        joint_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        default_pos = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids]
        joint_vel = self._asset.data.joint_vel[env_ids][:, self._joint_ids]
        return {
            "base_ang_vel": self._asset.data.root_ang_vel_b[env_ids] * 0.2,
            "projected_gravity": self._asset.data.projected_gravity_b[env_ids],
            "velocity_commands": self._velocity_commands[env_ids],
            "joint_pos_rel": joint_pos - default_pos,
            "joint_vel_rel": joint_vel * 0.05,
            "last_action": self._last_policy_action[env_ids],
        }

    def _build_policy_observations(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Build Unitree's term-major, oldest-to-newest five-frame history."""

        current_terms = self._current_terms(env_ids)
        was_initialized = self._history_initialized[env_ids]
        flattened_terms: list[torch.Tensor] = []
        for name in self._TERM_DIMS:
            current = current_terms[name]
            history = self._term_history[name][env_ids].clone()
            if torch.any(~was_initialized):
                history[~was_initialized] = current[~was_initialized].unsqueeze(1).expand(
                    -1, UNITREE_POLICY_HISTORY_LENGTH, -1
                )
            if torch.any(was_initialized):
                history[was_initialized, :-1] = history[was_initialized, 1:].clone()
                history[was_initialized, -1] = current[was_initialized]
            self._term_history[name][env_ids] = history
            flattened_terms.append(history.reshape(env_ids.numel(), -1))
        self._history_initialized[env_ids] = True
        observations = torch.cat(flattened_terms, dim=-1)
        if observations.shape[1] != UNITREE_POLICY_INPUT_DIM:
            raise RuntimeError(
                f"Unitree history must flatten to {UNITREE_POLICY_INPUT_DIM}, "
                f"got {observations.shape[1]}."
            )
        return observations


__all__ = [
    "UNITREE_POLICY_DEFAULT_JOINT_POS",
    "UNITREE_POLICY_JOINT_NAMES",
    "UNITREE_POLICY_TO_SDK",
    "UNITREE_SDK_JOINT_NAMES",
    "UnitreeNavigationSE2Action",
]
