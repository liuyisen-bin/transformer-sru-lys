"""Humanoid navigation action backed by a deployed locomotion policy."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from .navigation_se2_actions_humanoid_cfg import HumanoidNavigationSE2ActionCfg


class HumanoidNavigationSE2Action(ActionTerm):
    """Convert navigation velocity commands into humanoid joint targets.

    The observation history and command layout match EngineAI's PM01
    ``rl_walking_example`` deployment runner:

    ``15 * [q-q_default, qd, previous_action, angular_velocity, gravity] + command``.
    """

    cfg: HumanoidNavigationSE2ActionCfg
    _env: ManagerBasedRLEnv

    def __init__(self, cfg: HumanoidNavigationSE2ActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        if not check_file_path(cfg.low_level_policy_file):
            raise FileNotFoundError(f"Policy file '{cfg.low_level_policy_file}' does not exist.")

        self._asset: Articulation = env.scene[cfg.asset_name]
        self._active_joint_ids, active_joint_names = self._asset.find_joints(
            cfg.active_joint_names, preserve_order=True
        )
        if active_joint_names != cfg.active_joint_names:
            raise ValueError(
                "PM01 active joint order does not match the deployment policy. "
                f"Expected {cfg.active_joint_names}, resolved {active_joint_names}."
            )

        self.low_level_position_action_term: ActionTerm = cfg.low_level_position_action.class_type(
            cfg.low_level_position_action, env
        )
        if self.low_level_position_action_term.action_dim != len(cfg.active_joint_names):
            raise ValueError(
                "Low-level position action dimension must match PM01 policy output: "
                f"{self.low_level_position_action_term.action_dim} != {len(cfg.active_joint_names)}."
            )

        self._action_dim = 3
        self._num_policy_actions = len(cfg.active_joint_names)
        expected_observation_dim = self._num_policy_actions * 3 + 6
        if cfg.num_observations != expected_observation_dim:
            raise ValueError(
                f"Expected {expected_observation_dim} single-step observations for "
                f"{self._num_policy_actions} joints, got {cfg.num_observations}."
            )

        self._init_buffers()
        self._load_policy()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_navigation_velocity_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_navigation_velocity_actions

    @property
    def filtered_velocity_commands(self) -> torch.Tensor:
        return self._prev_filtered_velocity_commands

    @property
    def low_pass_alpha_values(self) -> torch.Tensor:
        return self._per_env_per_dim_low_pass_alpha

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def low_level_position_actions(self) -> torch.Tensor:
        return self._low_level_position_actions

    @property
    def prev_low_level_position_actions(self) -> torch.Tensor:
        return self._prev_low_level_position_actions

    @property
    def active_joint_ids(self) -> list[int]:
        return self._active_joint_ids

    def process_actions(self, actions: torch.Tensor):
        """Normalize and filter high-level ``[vx, vy, yaw_rate]`` commands."""
        self._raw_navigation_velocity_actions.copy_(actions)

        if self.cfg.use_raw_actions:
            normalized_actions = actions
        else:
            normalized_actions = actions * self._scale + self._offset

        if self.cfg.policy_distr_type == "gaussian":
            normalized_actions = torch.tanh(normalized_actions)
        elif self.cfg.policy_distr_type == "beta":
            normalized_actions = (normalized_actions - 0.5) * 2.0
        else:
            raise ValueError(f"Unknown policy distribution type: {self.cfg.policy_distr_type}")

        policy_scaling = self._policy_scaling
        if self._policy_scaling_negative is not None:
            policy_scaling = torch.where(
                normalized_actions < 0.0,
                self._policy_scaling_negative,
                self._policy_scaling,
            )
        velocity_commands = (normalized_actions + self._policy_bias) * policy_scaling
        if self.cfg.enable_low_pass_filter:
            alpha = self._per_env_per_dim_low_pass_alpha
            velocity_commands = alpha * self._prev_filtered_velocity_commands + (1.0 - alpha) * velocity_commands

        self._prev_filtered_velocity_commands.copy_(velocity_commands)
        self._processed_navigation_velocity_actions.copy_(velocity_commands)

    @torch.inference_mode()
    def apply_actions(self):
        """Run the 100 Hz humanoid locomotion policy and apply its joint targets."""
        if self._counter % self.cfg.low_level_decimation == 0:
            self._counter = 0
            self._prev_low_level_position_actions.copy_(self._low_level_position_actions)

            policy_input = self._build_policy_input()
            policy_actions = self._run_policy(policy_input)
            policy_actions.clamp_(-self.cfg.action_clip, self.cfg.action_clip)
            self._low_level_position_actions.copy_(policy_actions)
            self.low_level_position_action_term.process_actions(self._low_level_position_actions)

        self.low_level_position_action_term.apply_actions()
        self._counter += 1

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset command filtering, history, and previous actions."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        self._raw_navigation_velocity_actions[env_ids] = 0.0
        self._processed_navigation_velocity_actions[env_ids] = 0.0
        self._prev_filtered_velocity_commands[env_ids] = 0.0
        self._low_level_position_actions[env_ids] = 0.0
        self._prev_low_level_position_actions[env_ids] = 0.0
        self._observation_history[env_ids] = 0.0
        self._history_initialized[env_ids] = False
        self._counter = 0
        self.low_level_position_action_term.reset(env_ids)

    def reset_low_pass_filter(self, env_ids: torch.Tensor):
        self._prev_filtered_velocity_commands[env_ids] = 0.0

    def _init_buffers(self):
        self._raw_navigation_velocity_actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._processed_navigation_velocity_actions = torch.zeros_like(
            self._raw_navigation_velocity_actions
        )
        self._prev_filtered_velocity_commands = torch.zeros_like(
            self._raw_navigation_velocity_actions
        )
        self._per_env_per_dim_low_pass_alpha = torch.full(
            (self.num_envs, self._action_dim),
            self.cfg.low_pass_filter_alpha,
            device=self.device,
        )
        self._low_level_position_actions = torch.zeros(
            self.num_envs, self._num_policy_actions, device=self.device
        )
        self._prev_low_level_position_actions = torch.zeros_like(
            self._low_level_position_actions
        )
        self._observation_history = torch.zeros(
            self.num_envs,
            self.cfg.num_include_obs_steps,
            self.cfg.num_observations,
            device=self.device,
        )
        self._history_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._scale = torch.tensor(self.cfg.scale, device=self.device)
        self._offset = torch.tensor(self.cfg.offset, device=self.device)
        self._policy_scaling = torch.tensor(
            self.cfg.policy_scaling, device=self.device
        ).expand(self.num_envs, -1).clone()
        self._policy_scaling_negative = (
            torch.tensor(self.cfg.policy_scaling_negative, device=self.device)
            .expand(self.num_envs, -1)
            .clone()
            if self.cfg.policy_scaling_negative is not None
            else None
        )
        self._policy_bias = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._observation_scale = torch.tensor(
            self.cfg.observation_scale, device=self.device
        )
        self._command_observation_scale = torch.tensor(
            self.cfg.command_observation_scale, device=self.device
        )
        self._counter = 0

    def _build_policy_input(self) -> torch.Tensor:
        joint_ids = self._active_joint_ids
        joint_pos = self._asset.data.joint_pos[:, joint_ids]
        default_joint_pos = self._asset.data.default_joint_pos[:, joint_ids]
        joint_vel = self._asset.data.joint_vel[:, joint_ids]

        observation = torch.cat(
            (
                joint_pos - default_joint_pos,
                joint_vel,
                self._low_level_position_actions,
                self._asset.data.root_ang_vel_b,
                self._asset.data.projected_gravity_b,
            ),
            dim=-1,
        )
        observation.mul_(self._observation_scale)
        observation.clamp_(-self.cfg.observation_clip, self.cfg.observation_clip)

        self._observation_history.copy_(torch.roll(self._observation_history, shifts=-1, dims=1))
        uninitialized = ~self._history_initialized
        if torch.any(uninitialized):
            self._observation_history[uninitialized] = observation[uninitialized].unsqueeze(1).expand(
                -1, self.cfg.num_include_obs_steps, -1
            )
            self._history_initialized[uninitialized] = True
        self._observation_history[:, -1] = observation

        command = self._processed_navigation_velocity_actions * self._command_observation_scale
        return torch.cat((self._observation_history.flatten(start_dim=1), command), dim=-1)

    def _load_policy(self):
        extension = os.path.splitext(self.cfg.low_level_policy_file)[1].lower()
        if extension == ".mnn":
            try:
                import MNN
            except ImportError as exc:
                raise ImportError(
                    "PM01 locomotion uses an MNN model. Install the `MNN` Python package "
                    "in the Isaac Lab environment."
                ) from exc

            self._policy_type = "mnn"
            self._mnn = MNN
            self._mnn_interpreter = MNN.Interpreter(self.cfg.low_level_policy_file)
            self._mnn_session = self._mnn_interpreter.createSession(
                {
                    "backend": self.cfg.mnn_backend,
                    "precision": self.cfg.mnn_precision,
                    "thread": self.cfg.mnn_num_threads,
                }
            )
            self._mnn_input = self._mnn_interpreter.getSessionInput(self._mnn_session)
            input_dim = self.cfg.num_observations * self.cfg.num_include_obs_steps + self._action_dim
            self._mnn_interpreter.resizeTensor(self._mnn_input, (self.num_envs, input_dim))
            self._mnn_interpreter.resizeSession(self._mnn_session)
            output_shape = self._mnn_interpreter.getSessionOutput(self._mnn_session).getShape()
            if tuple(output_shape) != (self.num_envs, self._num_policy_actions):
                raise ValueError(
                    "Unexpected MNN policy output shape: "
                    f"{output_shape}, expected {(self.num_envs, self._num_policy_actions)}."
                )
        else:
            self._policy_type = "torchscript"
            file_bytes = read_file(self.cfg.low_level_policy_file)
            self._torch_policy = torch.jit.load(file_bytes, map_location=self.device)
            self._torch_policy.eval()

    def _run_policy(self, policy_input: torch.Tensor) -> torch.Tensor:
        if self._policy_type == "torchscript":
            return self._torch_policy(policy_input)

        input_array = policy_input.detach().to("cpu").contiguous().numpy().astype(np.float32, copy=False)
        host_input = self._mnn.Tensor(
            input_array.shape,
            self._mnn.Halide_Type_Float,
            input_array,
            self._mnn.Tensor_DimensionType_Caffe,
        )
        self._mnn_input.copyFrom(host_input)
        self._mnn_interpreter.runSession(self._mnn_session)

        output = self._mnn_interpreter.getSessionOutput(self._mnn_session)
        output_shape = output.getShape()
        host_output = self._mnn.Tensor(
            output_shape,
            self._mnn.Halide_Type_Float,
            np.zeros(output_shape, dtype=np.float32),
            self._mnn.Tensor_DimensionType_Caffe,
        )
        output.copyToHostTensor(host_output)
        output_array = np.asarray(host_output.getData(), dtype=np.float32).reshape(output_shape)
        return torch.from_numpy(output_array.copy()).to(self.device)
