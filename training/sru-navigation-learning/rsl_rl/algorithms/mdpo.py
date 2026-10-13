#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""MDPO (Meta Distilled Policy Optimization) algorithm.

This algorithm uses dual policy learning with mutual distillation to improve
exploration and sample efficiency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from rsl_rl.storage import RolloutStorage

if TYPE_CHECKING:
    from rsl_rl.modules import ActorCritic

# Check if Muon optimizer is available
try:
    from rsl_rl.algorithms.optim import SingleDeviceMuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False

EPSILON = 1e-7


def _kl_gaussian(mu_t: torch.Tensor, std_t: torch.Tensor, mu_s: torch.Tensor, std_s: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence KL(teacher || student) between two diagonal Gaussian distributions.

    Args:
        mu_t: Teacher mean.
        std_t: Teacher standard deviation.
        mu_s: Student mean.
        std_s: Student standard deviation.

    Returns:
        KL divergence summed over action dimensions.
    """
    std_t = std_t.clamp_min(EPSILON)
    std_s = std_s.clamp_min(EPSILON)
    var_t = std_t.square()
    var_s = std_s.square()
    log_ratio = torch.log(std_s) - torch.log(std_t)
    mahalanobis = (var_t + (mu_t - mu_s).square()) / (2.0 * var_s)
    return (log_ratio + mahalanobis - 0.5).sum(dim=-1)


class MDPO:
    """Meta Distilled Policy Optimization algorithm.

    Uses two policies that learn from each other through mutual distillation.
    Each policy is trained on half of the environments, and they share knowledge
    through KL divergence penalties.

    Args:
        actor_critic_1: First actor-critic network.
        actor_critic_2: Second actor-critic network.
        num_learning_epochs: Number of epochs per update.
        num_mini_batches: Number of mini-batches per epoch.
        clip_param: PPO clip parameter.
        value_clip_param: Value function clip parameter.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        value_loss_coef: Coefficient for value loss.
        distill_coef: Coefficient for mutual distillation loss.
        entropy_coef: Coefficient for entropy bonus.
        learning_rate: Initial learning rate.
        min_learning_rate: Minimum learning rate for scheduling.
        weight_decay: Weight decay for optimizer.
        max_grad_norm: Maximum gradient norm for clipping.
        use_clipped_value_loss: Whether to use clipped value loss.
        schedule: Learning rate schedule ('fixed', 'linear', 'cosine', or 'exponential').
        desired_kl: Desired KL divergence (kept for API compatibility).
        device: Device to run on.
        use_muon: Whether to use Muon optimizer for hidden weights.
    """

    actor_critic_1: "ActorCritic"
    actor_critic_2: "ActorCritic"

    def __init__(
        self,
        actor_critic_1: "ActorCritic",
        actor_critic_2: "ActorCritic",
        num_learning_epochs: int = 1,
        num_mini_batches: int = 1,
        clip_param: float = 0.2,
        value_clip_param: float = 0.5,
        gamma: float = 0.998,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        distill_coef: float = 0.02,
        entropy_coef: float = 0.0,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-7,
        weight_decay: float = 0.0,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "linear",
        desired_kl: float = 0.01,
        device: str = "cpu",
        use_muon: bool = True,
        pointnet_freeze_iterations: int = 500,
        pointnet_partial_iterations: int = 2000,
        pointnet_lr_scale: float = 0.1,
        **kwargs,
    ):
        self.device = device

        # Learning rate configuration
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.max_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate
        self.weight_decay = weight_decay
        self.desired_kl = desired_kl
        self.pointnet_freeze_iterations = pointnet_freeze_iterations
        self.pointnet_partial_iterations = pointnet_partial_iterations
        self.pointnet_lr_scale = pointnet_lr_scale
        self._pointnet_training_stage = None

        # Dual policy components
        self.actor_critic_1 = actor_critic_1
        self.actor_critic_2 = actor_critic_2
        self.actor_critic_1.to(self.device)
        self.actor_critic_2.to(self.device)

        # Storage (initialized later)
        self.storage_1: RolloutStorage | None = None
        self.storage_2: RolloutStorage | None = None
        self.indices_1: torch.Tensor | None = None
        self.indices_2: torch.Tensor | None = None
        self._reported_mini_batch_clamp = False

        # Create optimizers
        self.optimizer_1 = self._create_optimizer(self.actor_critic_1, learning_rate, use_muon)
        self.optimizer_2 = self._create_optimizer(self.actor_critic_2, learning_rate, use_muon)

        # Transitions for rollout collection
        self.transition_1 = RolloutStorage.Transition()
        self.transition_2 = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.distill_coef = distill_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def _create_optimizer(
        self, actor_critic: "ActorCritic", learning_rate: float, use_muon: bool
    ) -> torch.optim.Optimizer:
        """Create optimizer with optional Muon support for hidden weights."""
        pointnet_parameters = (
            list(actor_critic.get_pointnet_parameters())
            if hasattr(actor_critic, "get_pointnet_parameters")
            else []
        )
        pointnet_parameter_ids = {id(parameter) for parameter in pointnet_parameters}

        if use_muon and MUON_AVAILABLE:
            hidden_weights, nonhidden_params = self._separate_parameters(actor_critic)
            hidden_weights = [parameter for parameter in hidden_weights if id(parameter) not in pointnet_parameter_ids]
            nonhidden_params = [
                parameter for parameter in nonhidden_params if id(parameter) not in pointnet_parameter_ids
            ]
            param_groups = [
                {
                    "params": hidden_weights,
                    "use_muon": True,
                    "lr": learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": nonhidden_params,
                    "use_muon": False,
                    "lr": learning_rate,
                    "betas": (0.9, 0.999),
                    "weight_decay": self.weight_decay,
                },
            ]
            if pointnet_parameters:
                param_groups.append(
                    {
                        "params": pointnet_parameters,
                        "use_muon": False,
                        "lr": learning_rate * self.pointnet_lr_scale,
                        "betas": (0.9, 0.999),
                        "weight_decay": self.weight_decay,
                    }
                )
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            optimizer._lr_scales = [1.0, 1.0] + (
                [self.pointnet_lr_scale] if pointnet_parameters else []
            )
            print("[MDPO] Using Muon optimizer for hidden weights and AdamW for other parameters")
        else:
            if use_muon and not MUON_AVAILABLE:
                print("[MDPO] Warning: Muon requested but not available. Falling back to Adam.")
            if pointnet_parameters:
                policy_parameters = [
                    parameter for parameter in actor_critic.parameters() if id(parameter) not in pointnet_parameter_ids
                ]
                optimizer = torch.optim.Adam(
                    [
                        {"params": policy_parameters, "lr": learning_rate, "lr_scale": 1.0},
                        {
                            "params": pointnet_parameters,
                            "lr": learning_rate * self.pointnet_lr_scale,
                            "lr_scale": self.pointnet_lr_scale,
                        },
                    ],
                    weight_decay=self.weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(actor_critic.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
            print("[MDPO] Using Adam optimizer")
        return optimizer

    def set_training_iteration(self, iteration: int) -> None:
        """Apply the same perception curriculum to both distilled policies."""
        if not hasattr(self.actor_critic_1, "set_pointnet_training_stage"):
            return
        if iteration < self.pointnet_freeze_iterations:
            stage = "frozen"
        elif iteration < self.pointnet_partial_iterations:
            stage = "partial"
        else:
            stage = "full"
        if stage != self._pointnet_training_stage:
            self.actor_critic_1.set_pointnet_training_stage(stage)
            self.actor_critic_2.set_pointnet_training_stage(stage)
            self._pointnet_training_stage = stage
            encoder_name = getattr(self.actor_critic_1, "perception_encoder_name", "PointNet")
            print(f"[MDPO] {encoder_name} training stage -> {stage} at iteration {iteration}")

    def _separate_parameters(self, actor_critic: "ActorCritic") -> tuple[list, list]:
        """Separate parameters into hidden weights (for Muon) and nonhidden params (for AdamW).

        Hidden weights: All linear layer weights except the last layers of actor and critic.
        Nonhidden params: Biases, gains, last layer weights, and other 1D parameters.
        """
        hidden_weights = []
        nonhidden_params = []

        # Find the last linear layers in actor and critic
        last_actor_layer = self._find_last_linear_layer(actor_critic, "actor")
        last_critic_layer = self._find_last_linear_layer(actor_critic, "critic")

        # Separate parameters
        for param in actor_critic.parameters():
            if param.ndim >= 2:
                is_last_layer = (
                    (last_actor_layer is not None and param is last_actor_layer.weight)
                    or (last_critic_layer is not None and param is last_critic_layer.weight)
                )
                if is_last_layer:
                    nonhidden_params.append(param)
                else:
                    hidden_weights.append(param)
            else:
                nonhidden_params.append(param)

        # Uncomment for debugging: print(f"[MDPO] Parameter separation: {len(hidden_weights)} hidden weights, {len(nonhidden_params)} nonhidden")
        return hidden_weights, nonhidden_params

    @staticmethod
    def _find_last_linear_layer(actor_critic: "ActorCritic", attr_name: str) -> nn.Linear | None:
        """Find the last linear layer in a submodule."""
        if not hasattr(actor_critic, attr_name):
            return None
        module = getattr(actor_critic, attr_name)
        linear_modules = [m for m in module.modules() if isinstance(m, nn.Linear)]
        return linear_modules[-1] if linear_modules else None

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: tuple,
        critic_obs_shape: tuple,
        action_shape: tuple,
    ):
        """Initialize rollout storage for both policies."""
        if num_envs < 2:
            raise ValueError("MDPO requires at least two environments so each policy receives one.")

        num_env_1 = num_envs // 2
        num_env_2 = num_envs - num_env_1

        # Odd indices for policy 1, even indices for policy 2
        self.indices_1 = torch.arange(1, num_envs, 2, device=self.device)
        self.indices_2 = torch.arange(0, num_envs, 2, device=self.device)

        self.storage_1 = RolloutStorage(
            num_env_1, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )
        self.storage_2 = RolloutStorage(
            num_env_2, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def _effective_num_mini_batches(self) -> int:
        """Clamp recurrent mini-batches to the environments owned by each policy."""
        if self.storage_1 is None or self.storage_2 is None:
            raise RuntimeError("MDPO rollout storage must be initialized before update.")

        effective = min(self.num_mini_batches, self.storage_1.num_envs, self.storage_2.num_envs)
        if effective < 1:
            raise ValueError("MDPO needs at least one environment per policy mini-batch.")
        if effective != self.num_mini_batches and not self._reported_mini_batch_clamp:
            print(
                f"[MDPO] Reducing num_mini_batches from {self.num_mini_batches} to {effective} "
                "because environments are split between two policies."
            )
            self._reported_mini_batch_clamp = True
        return effective

    def test_mode(self):
        """Set both policies to evaluation mode."""
        self.actor_critic_1.eval()
        self.actor_critic_2.eval()

    def train_mode(self):
        """Set both policies to training mode."""
        self.actor_critic_1.train()
        self.actor_critic_2.train()

    def act(self, obs: torch.Tensor, critic_obs: torch.Tensor) -> torch.Tensor:
        """Compute actions for all environments.

        Splits observations between the two policies and combines their actions.
        """
        # Split observations
        obs_1, obs_2 = obs[self.indices_1], obs[self.indices_2]
        critic_obs_1, critic_obs_2 = critic_obs[self.indices_1], critic_obs[self.indices_2]

        # Capture hidden states if recurrent
        if self.actor_critic_1.is_recurrent:
            self.transition_1.hidden_states = self.actor_critic_1.get_hidden_states()
        if self.actor_critic_2.is_recurrent:
            self.transition_2.hidden_states = self.actor_critic_2.get_hidden_states()

        # Compute actions and values
        self.transition_1.actions = self.actor_critic_1.act(obs_1).detach()
        self.transition_2.actions = self.actor_critic_2.act(obs_2).detach()
        self.transition_1.values = self.actor_critic_1.evaluate(critic_obs_1).detach()
        self.transition_2.values = self.actor_critic_2.evaluate(critic_obs_2).detach()

        # Store log-probs, means, and stds
        self.transition_1.actions_log_prob = self.actor_critic_1.get_actions_log_prob(self.transition_1.actions).detach()
        self.transition_2.actions_log_prob = self.actor_critic_2.get_actions_log_prob(self.transition_2.actions).detach()
        self.transition_1.action_mean = self.actor_critic_1.action_mean.detach()
        self.transition_1.action_sigma = self.actor_critic_1.action_std.detach()
        self.transition_2.action_mean = self.actor_critic_2.action_mean.detach()
        self.transition_2.action_sigma = self.actor_critic_2.action_std.detach()

        # Store observations
        self.transition_1.observations = obs_1
        self.transition_2.observations = obs_2
        self.transition_1.critic_observations = critic_obs_1
        self.transition_2.critic_observations = critic_obs_2

        # Combine actions to match original environment indices
        full_actions = torch.empty(
            self.indices_1.numel() + self.indices_2.numel(),
            *self.transition_1.actions.shape[1:],
            device=self.device,
        )
        full_actions[self.indices_1] = self.transition_1.actions
        full_actions[self.indices_2] = self.transition_2.actions
        return full_actions

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict):
        """Process environment step and store transitions."""
        rewards_1, rewards_2 = rewards[self.indices_1], rewards[self.indices_2]
        dones_1, dones_2 = dones[self.indices_1], dones[self.indices_2]

        self.transition_1.rewards = rewards_1.clone()
        self.transition_2.rewards = rewards_2.clone()
        self.transition_1.dones = dones_1
        self.transition_2.dones = dones_2

        # Bootstrap on time-out truncations
        if "time_outs" in infos:
            time_outs_1 = infos["time_outs"][self.indices_1].unsqueeze(1).to(self.device)
            time_outs_2 = infos["time_outs"][self.indices_2].unsqueeze(1).to(self.device)
            self.transition_1.rewards += self.gamma * (self.transition_1.values * time_outs_1).squeeze(1)
            self.transition_2.rewards += self.gamma * (self.transition_2.values * time_outs_2).squeeze(1)

        # Add transitions to storage
        self.storage_1.add_transitions(self.transition_1)
        self.storage_2.add_transitions(self.transition_2)

        # Clear transitions and reset hidden states
        self.transition_1.clear()
        self.transition_2.clear()
        self.actor_critic_1.reset(dones_1)
        self.actor_critic_2.reset(dones_2)

    def compute_returns(self, last_critic_obs: torch.Tensor):
        """Compute GAE-lambda returns for each storage."""
        last_critic_obs_1 = last_critic_obs[self.indices_1]
        last_critic_obs_2 = last_critic_obs[self.indices_2]

        last_values_1 = self.actor_critic_1.evaluate(last_critic_obs_1).detach()
        last_values_2 = self.actor_critic_2.evaluate(last_critic_obs_2).detach()

        self.storage_1.compute_returns(last_values_1, self.gamma, self.lam)
        self.storage_2.compute_returns(last_values_2, self.gamma, self.lam)

    def update_dropout_masks(self):
        """Update dropout masks from actor-critics."""
        self.storage_1.saved_dropout_masks_a, self.storage_1.saved_dropout_masks_c = (
            self.actor_critic_1.get_dropout_masks()
        )
        self.storage_2.saved_dropout_masks_a, self.storage_2.saved_dropout_masks_c = (
            self.actor_critic_2.get_dropout_masks()
        )

    def reset_dropout_masks(self):
        """Reset dropout masks in actor-critics."""
        self.actor_critic_1.reset_dropout_masks()
        self.actor_critic_2.reset_dropout_masks()

    def _update_learning_rate(self, iteration: int, max_iterations: int):
        """Update learning rate based on schedule."""
        if self.schedule == "fixed":
            return
        elif self.schedule == "linear":
            # Linear decay to min_lr over first 33% of training
            progress = iteration / (max_iterations * 0.33)
            if progress < 1.0:
                self.learning_rate = self.max_learning_rate - progress * (
                    self.max_learning_rate - self.min_learning_rate
                )
            else:
                self.learning_rate = self.min_learning_rate
        elif self.schedule == "cosine":
            progress = iteration / max_iterations
            self.learning_rate = 0.5 * self.max_learning_rate * (1.0 + math.cos(math.pi * progress))
        elif self.schedule == "exponential":
            progress = iteration / max_iterations
            decay_rate = 5.0
            self.learning_rate = self.min_learning_rate + (
                self.max_learning_rate - self.min_learning_rate
            ) * math.exp(-decay_rate * progress)

        # Apply to optimizers
        scales_1 = getattr(
            self.optimizer_1,
            "_lr_scales",
            [group.get("lr_scale", 1.0) for group in self.optimizer_1.param_groups],
        )
        scales_2 = getattr(
            self.optimizer_2,
            "_lr_scales",
            [group.get("lr_scale", 1.0) for group in self.optimizer_2.param_groups],
        )
        for param_group, scale in zip(self.optimizer_1.param_groups, scales_1):
            param_group["lr"] = self.learning_rate * scale
        for param_group, scale in zip(self.optimizer_2.param_groups, scales_2):
            param_group["lr"] = self.learning_rate * scale

    def _compute_ppo_loss(
        self,
        actions_log_prob: torch.Tensor,
        old_actions_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        value_batch: torch.Tensor,
        target_values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PPO surrogate and value losses.

        Returns:
            Tuple of (surrogate_loss, value_loss, total_policy_loss).
        """
        # Surrogate loss
        ratio = torch.exp(actions_log_prob - old_actions_log_prob.squeeze())
        advantages_squeezed = advantages.squeeze()
        surrogate = -advantages_squeezed * ratio
        surrogate_clipped = -advantages_squeezed * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value loss
        if self.use_clipped_value_loss:
            value_clipped = target_values + (value_batch - target_values).clamp(
                -self.value_clip_param, self.value_clip_param
            )
            value_losses = (value_batch - returns).square()
            value_losses_clipped = (value_clipped - returns).square()
            value_loss = torch.max(value_losses, value_losses_clipped).mean() * 0.5
        else:
            value_loss = (returns - value_batch).square().mean() * 0.5

        # Combined policy loss
        policy_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        return surrogate_loss, value_loss, policy_loss

    def update(self, iteration: int, max_iterations: int) -> tuple[float, float, float]:
        """Perform policy update with mutual distillation.

        Returns:
            Tuple of (mean_value_loss, mean_surrogate_loss, mean_kl_divergence).
        """
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_kl_divergence = 0.0

        # Update learning rate
        self._update_learning_rate(iteration, max_iterations)

        # Each MDPO policy owns half of the environments. Recurrent generators
        # partition by environment, so they cannot create more mini-batches
        # than the per-policy environment count.
        effective_num_mini_batches = self._effective_num_mini_batches()

        # Get mini-batch generators
        if self.actor_critic_1.is_recurrent:
            generator_1 = self.storage_1.reccurent_mini_batch_generator(
                effective_num_mini_batches, self.num_learning_epochs
            )
            generator_2 = self.storage_2.reccurent_mini_batch_generator(
                effective_num_mini_batches, self.num_learning_epochs
            )
        else:
            generator_1 = self.storage_1.mini_batch_generator(
                effective_num_mini_batches, self.num_learning_epochs
            )
            generator_2 = self.storage_2.mini_batch_generator(
                effective_num_mini_batches, self.num_learning_epochs
            )

        num_updates = self.num_learning_epochs * effective_num_mini_batches

        for batch_1, batch_2 in zip(generator_1, generator_2):
            # Unpack mini-batches
            (
                obs_batch_1, critic_obs_batch_1, actions_batch_1, target_values_batch_1,
                advantages_batch_1, returns_batch_1, old_actions_log_prob_batch_1,
                old_mu_batch_1, old_sigma_batch_1, hid_states_batch_1, masks_batch_1,
                dropout_masks_a_1, dropout_masks_c_1,
            ) = batch_1

            (
                obs_batch_2, critic_obs_batch_2, actions_batch_2, target_values_batch_2,
                advantages_batch_2, returns_batch_2, old_actions_log_prob_batch_2,
                old_mu_batch_2, old_sigma_batch_2, hid_states_batch_2, masks_batch_2,
                dropout_masks_a_2, dropout_masks_c_2,
            ) = batch_2

            # Forward pass for Policy 1
            self.actor_critic_1.act(
                obs_batch_1, masks=masks_batch_1, hidden_states=hid_states_batch_1[0], dropout_masks=dropout_masks_a_1
            )
            actions_log_prob_1 = self.actor_critic_1.get_actions_log_prob(actions_batch_1)
            value_batch_1 = self.actor_critic_1.evaluate(
                critic_obs_batch_1, masks=masks_batch_1, hidden_states=hid_states_batch_1[1], dropout_masks=dropout_masks_c_1
            )
            mu_batch_1 = self.actor_critic_1.action_mean
            sigma_batch_1 = self.actor_critic_1.action_std
            entropy_batch_1 = self.actor_critic_1.entropy

            # Forward pass for Policy 2
            self.actor_critic_2.act(
                obs_batch_2, masks=masks_batch_2, hidden_states=hid_states_batch_2[0], dropout_masks=dropout_masks_a_2
            )
            actions_log_prob_2 = self.actor_critic_2.get_actions_log_prob(actions_batch_2)
            value_batch_2 = self.actor_critic_2.evaluate(
                critic_obs_batch_2, masks=masks_batch_2, hidden_states=hid_states_batch_2[1], dropout_masks=dropout_masks_c_2
            )
            mu_batch_2 = self.actor_critic_2.action_mean
            sigma_batch_2 = self.actor_critic_2.action_std
            entropy_batch_2 = self.actor_critic_2.entropy

            # Normalize advantages
            with torch.no_grad():
                advantages_batch_1 = (advantages_batch_1 - advantages_batch_1.mean()) / (advantages_batch_1.std() + EPSILON)
                advantages_batch_2 = (advantages_batch_2 - advantages_batch_2.mean()) / (advantages_batch_2.std() + EPSILON)

            # Compute PPO losses
            surrogate_loss_1, value_loss_1, policy_loss_1 = self._compute_ppo_loss(
                actions_log_prob_1, old_actions_log_prob_batch_1, advantages_batch_1,
                value_batch_1, target_values_batch_1, returns_batch_1, entropy_batch_1
            )
            surrogate_loss_2, value_loss_2, policy_loss_2 = self._compute_ppo_loss(
                actions_log_prob_2, old_actions_log_prob_batch_2, advantages_batch_2,
                value_batch_2, target_values_batch_2, returns_batch_2, entropy_batch_2
            )

            # Compute KL divergence for mutual distillation
            # NOTE: Gradients flow through BOTH student and teacher - this is intentional!
            # In mutual distillation, the KL term regularizes both policies to stay close to each other.

            # Policy 2 as teacher for Policy 1
            self.actor_critic_2.act(
                obs_batch_1, masks=masks_batch_1, hidden_states=hid_states_batch_1[0], dropout_masks=dropout_masks_a_2
            )
            teacher_mu_1 = self.actor_critic_2.action_mean
            teacher_sigma_1 = self.actor_critic_2.action_std

            # Policy 1 as teacher for Policy 2
            self.actor_critic_1.act(
                obs_batch_2, masks=masks_batch_2, hidden_states=hid_states_batch_2[0], dropout_masks=dropout_masks_a_1
            )
            teacher_mu_2 = self.actor_critic_1.action_mean
            teacher_sigma_2 = self.actor_critic_1.action_std

            # KL divergence losses - both policies are regularized to match each other
            distill_kl_1 = _kl_gaussian(teacher_mu_1, teacher_sigma_1, mu_batch_1, sigma_batch_1).mean()
            distill_kl_2 = _kl_gaussian(teacher_mu_2, teacher_sigma_2, mu_batch_2, sigma_batch_2).mean()

            # Total loss with distillation
            total_loss = 0.5 * (
                (policy_loss_1 + self.distill_coef * distill_kl_1)
                + (policy_loss_2 + self.distill_coef * distill_kl_2)
            )

            # Gradient step
            self.optimizer_1.zero_grad(set_to_none=True)
            self.optimizer_2.zero_grad(set_to_none=True)
            total_loss.backward()

            # Clip gradients
            nn.utils.clip_grad_norm_(self.actor_critic_1.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic_2.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic_1.get_critic_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic_2.get_critic_parameters(), self.max_grad_norm)

            self.optimizer_1.step()
            self.optimizer_2.step()

            # Accumulate metrics
            mean_value_loss += (value_loss_1.item() + value_loss_2.item()) * 0.5
            mean_surrogate_loss += (surrogate_loss_1.item() + surrogate_loss_2.item()) * 0.5
            mean_kl_divergence += (distill_kl_1.item() + distill_kl_2.item()) * 0.5

        # Average over all mini-batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_kl_divergence /= num_updates

        # Clear rollouts
        self.storage_1.clear()
        self.storage_2.clear()

        return mean_value_loss, mean_surrogate_loss, mean_kl_divergence
