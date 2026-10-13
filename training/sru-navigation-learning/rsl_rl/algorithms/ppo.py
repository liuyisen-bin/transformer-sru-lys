#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

EPSILON = 1e-7

class PPO:
    actor_critic: ActorCritic

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        value_clip_param=0.5,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        pointnet_freeze_iterations=500,
        pointnet_partial_iterations=1000,
        pointnet_lr_scale=0.1,
        device="cpu",
        **kwargs,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.pointnet_freeze_iterations = pointnet_freeze_iterations
        self.pointnet_partial_iterations = pointnet_partial_iterations
        self.pointnet_lr_scale = pointnet_lr_scale
        self._pointnet_training_stage = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = self._create_optimizer()
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.value_clip_param = value_clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def _create_optimizer(self):
        """Create stable parameter groups before any PointNet layer is frozen."""
        if not hasattr(self.actor_critic, "get_pointnet_parameters"):
            return optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        pointnet_parameters = list(self.actor_critic.get_pointnet_parameters())
        pointnet_parameter_ids = {id(parameter) for parameter in pointnet_parameters}
        policy_parameters = [
            parameter for parameter in self.actor_critic.parameters() if id(parameter) not in pointnet_parameter_ids
        ]
        return optim.Adam(
            [
                {"params": policy_parameters, "lr": self.learning_rate, "lr_scale": 1.0},
                {
                    "params": pointnet_parameters,
                    "lr": self.learning_rate * self.pointnet_lr_scale,
                    "lr_scale": self.pointnet_lr_scale,
                },
            ]
        )

    def set_training_iteration(self, iteration: int) -> None:
        """Apply the perception-encoder curriculum for an absolute iteration."""
        if not hasattr(self.actor_critic, "set_pointnet_training_stage"):
            return
        if iteration < self.pointnet_freeze_iterations:
            stage = "frozen"
        elif iteration < self.pointnet_partial_iterations:
            stage = "partial"
        else:
            stage = "full"
        if stage != self._pointnet_training_stage:
            self.actor_critic.set_pointnet_training_stage(stage)
            self._pointnet_training_stage = stage
            encoder_name = getattr(self.actor_critic, "perception_encoder_name", "PointNet")
            print(f"[PPO] {encoder_name} training stage -> {stage} at iteration {iteration}")

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.eval()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        # print("obs", obs)
        # print("obs shape", obs.shape)
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
        
    def update_dropout_masks(self):
        # update the dropout masks
        self.storage.saved_dropout_masks_a, self.storage.saved_dropout_masks_c = self.actor_critic.get_dropout_masks()
        
    def reset_dropout_masks(self):
        self.actor_critic.reset_dropout_masks()

    def update(self, iter, max_iters):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            dropout_masks_a,
            dropout_masks_c,
        ) in generator:
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0], dropout_masks=dropout_masks_a)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1],
                dropout_masks=dropout_masks_c
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + EPSILON) + EPSILON)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-3, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate * param_group.get("lr_scale", 1.0)
                        
            # normalize advantages
            with torch.no_grad():
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + EPSILON)

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            # print("PPO max ratio: ", ratio.max().item())

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.value_clip_param, self.value_clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean() * 0.5
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean() * 0.5

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip the gradients for actor and critic separately
            nn.utils.clip_grad_norm_(self.actor_critic.get_actor_parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.actor_critic.get_critic_parameters(), self.max_grad_norm)
            
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
