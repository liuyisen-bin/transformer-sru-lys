"""Actor-Critic with PointNet LiDAR encoding and SRU memory."""

from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.modules.actor_critic_sru import LinearConstDropout, MemorySRU, get_activation
from rsl_rl.networks import CrossAttentionFuseModule, PointNetEncoder
from rsl_rl.utils import unpad_trajectories


class ActorCriticSRUPointNet(nn.Module):
    """PointNet -> proprioceptive fusion -> SRU actor-critic."""

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: Optional[list[int]] = None,
        critic_hidden_dims: Optional[list[int]] = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        height_input_dims: tuple[int, int, int] = (64, 7, 7),
        image_input_dims: tuple[int, int, int] = (64, 5, 8),
        num_cameras: int = 1,
        num_points: int = 656,
        point_dim: int = 3,
        pointnet_feature_dim: int = 128,
        pointnet_batch_size: int = 256,
        rnn_type: str = "lstm_sru",
        dropout: float = 0.2,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 1,
        time_embed_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"[ActorCriticSRUPointNet] Ignoring unused arguments: {list(kwargs.keys())}")
        if rnn_type != "lstm_sru":
            print(f"[ActorCriticSRUPointNet] Ignoring rnn_type={rnn_type!r}; LSTM_SRU is always used.")

        actor_hidden_dims = actor_hidden_dims or [256, 256, 256]
        critic_hidden_dims = critic_hidden_dims or [256, 256, 256]
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_points = num_points
        self.point_dim = point_dim
        self.pointnet_feature_dim = pointnet_feature_dim
        self.num_point_features = num_points * point_dim
        self.height_input_dims = height_input_dims
        self.num_height_features = height_input_dims[0] * height_input_dims[1] * height_input_dims[2]

        self.actor_proprioceptive_input_dim = num_actor_obs - self.num_point_features
        self.critic_proprioceptive_input_dim = (
            num_critic_obs - self.num_point_features - self.num_height_features - 1
        )
        if self.actor_proprioceptive_input_dim != self.critic_proprioceptive_input_dim:
            raise ValueError(
                "Actor and critic proprioceptive dimensions differ: "
                f"{self.actor_proprioceptive_input_dim} != {self.critic_proprioceptive_input_dim}"
            )
        if self.actor_proprioceptive_input_dim <= 0:
            raise ValueError("Point-cloud dimensions exceed the actor observation dimensions.")

        self.actor_pointnet = PointNetEncoder(
            point_dim=point_dim,
            feature_dim=pointnet_feature_dim,
            max_batch_size=pointnet_batch_size,
        )
        self.critic_pointnet = PointNetEncoder(
            point_dim=point_dim,
            feature_dim=pointnet_feature_dim,
            max_batch_size=pointnet_batch_size,
        )
        self.attn_height_net = CrossAttentionFuseModule(
            image_dim=height_input_dims[0],
            info_dim=self.critic_proprioceptive_input_dim,
            num_heads=4,
            spatial_dims=(1, height_input_dims[1], height_input_dims[2]),
        )

        self.memory_a = MemorySRU(
            input_size=self.actor_proprioceptive_input_dim + pointnet_feature_dim,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )
        self.memory_c = MemorySRU(
            input_size=self.critic_proprioceptive_input_dim + height_input_dims[0] + pointnet_feature_dim,
            num_layers=rnn_num_layers,
            hidden_size=rnn_hidden_size,
        )
        self.time_layer = nn.Linear(1, time_embed_dim)

        self.linear_dropout_actor = LinearConstDropout(
            rnn_hidden_size, actor_hidden_dims[0], dropout_p=dropout, activation_name=activation
        )
        self.actor = self._make_mlp(actor_hidden_dims, num_actions, activation)
        self.linear_dropout_critic = LinearConstDropout(
            rnn_hidden_size + time_embed_dim,
            critic_hidden_dims[0],
            dropout_p=dropout,
            activation_name=activation,
        )
        self.critic = self._make_mlp(critic_hidden_dims, 1, activation)

        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        self._pointnet_training_stage = "full"
        Normal.set_default_validate_args(False)

        print(
            "[ActorCriticSRUPointNet] "
            f"points={num_points}x{point_dim}, point_feature={pointnet_feature_dim}, "
            f"actor_sru_input={self.actor_proprioceptive_input_dim + pointnet_feature_dim}, "
            f"critic_sru_input={self.critic_proprioceptive_input_dim + height_input_dims[0] + pointnet_feature_dim}"
        )

    @staticmethod
    def _make_mlp(hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
        layers: list[nn.Module] = []
        for index, hidden_dim in enumerate(hidden_dims):
            next_dim = output_dim if index == len(hidden_dims) - 1 else hidden_dims[index + 1]
            layers.append(nn.Linear(hidden_dim, next_dim))
            if index != len(hidden_dims) - 1:
                layers.append(get_activation(activation))
        return nn.Sequential(*layers)

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    @property
    def pointnet_training_stage(self) -> str:
        return self._pointnet_training_stage

    def set_pointnet_training_stage(self, stage: str) -> None:
        self._pointnet_training_stage = stage
        self.actor_pointnet.set_training_stage(stage)
        self.critic_pointnet.set_training_stage(stage)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.set_pointnet_training_stage(self._pointnet_training_stage)
        return self

    def get_pointnet_parameters(self) -> list[nn.Parameter]:
        return list(self.actor_pointnet.parameters()) + list(self.critic_pointnet.parameters())

    def get_actor_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.actor_pointnet.parameters())
            + list(self.linear_dropout_actor.parameters())
            + list(self.actor.parameters())
            + list(self.memory_a.parameters())
            + [self.log_std]
        )

    def get_critic_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.critic_pointnet.parameters())
            + list(self.attn_height_net.parameters())
            + list(self.linear_dropout_critic.parameters())
            + list(self.time_layer.parameters())
            + list(self.critic.parameters())
            + list(self.memory_c.parameters())
        )

    def reset(self, dones=None):
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def _extract_points(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[..., -self.num_point_features :].reshape(
            *observations.shape[:-1], self.num_points, self.point_dim
        )

    def process_actor_input(self, observations: torch.Tensor, masks, hidden_states) -> torch.Tensor:
        batch_mode = masks is not None
        other_obs = observations[..., : -self.num_point_features]
        point_features = self.actor_pointnet(self._extract_points(observations))

        if batch_mode:
            seq_len, batch_size, _ = observations.shape
            other_obs = other_obs.reshape(seq_len, batch_size, self.actor_proprioceptive_input_dim)
            point_features = point_features.reshape(seq_len, batch_size, self.pointnet_feature_dim)
        else:
            other_obs = other_obs.reshape(-1, self.actor_proprioceptive_input_dim)
            point_features = point_features.reshape(-1, self.pointnet_feature_dim)

        combined_features = torch.cat([point_features, other_obs], dim=-1)
        return self.memory_a(combined_features, masks, hidden_states).squeeze(0)

    def process_critic_input(self, observations: torch.Tensor, masks, hidden_states) -> torch.Tensor:
        batch_mode = masks is not None
        tail_size = self.num_height_features + self.num_point_features + 1
        other_obs = observations[..., :-tail_size]
        time_obs = observations[..., -tail_size].unsqueeze(-1)
        height_start = self.num_point_features
        height_obs = observations[..., -(height_start + self.num_height_features) : -height_start]
        points = self._extract_points(observations)

        flat_other_obs = other_obs.reshape(-1, self.critic_proprioceptive_input_dim)
        flat_height_obs = height_obs.reshape(-1, *self.height_input_dims)
        height_features = self.attn_height_net(flat_height_obs, flat_other_obs)
        point_features = self.critic_pointnet(points)

        if batch_mode:
            seq_len, batch_size, _ = observations.shape
            other_obs = flat_other_obs.reshape(seq_len, batch_size, -1)
            height_features = height_features.reshape(seq_len, batch_size, -1)
            point_features = point_features.reshape(seq_len, batch_size, -1)
        else:
            other_obs = flat_other_obs
            point_features = point_features.reshape(-1, self.pointnet_feature_dim)

        combined_features = torch.cat([height_features, point_features, other_obs], dim=-1)
        combined_features = self.memory_c(combined_features, masks, hidden_states)
        time_embed = self.time_layer(time_obs)
        if batch_mode:
            time_embed = unpad_trajectories(time_embed, masks)
        return torch.cat([combined_features.squeeze(0), time_embed], dim=-1)

    def update_distribution(self, combined_features: torch.Tensor):
        mean = self.actor(combined_features)
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        features = self.process_actor_input(observations, masks, hidden_states)
        features = self.linear_dropout_actor(features, dropout_masks)
        self.update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        features = self.process_actor_input(observations, masks, hidden_states)
        features = self.linear_dropout_actor(features, dropout_masks)
        return self.actor(features)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        features = self.process_critic_input(critic_observations, masks, hidden_states)
        features = self.linear_dropout_critic(features, dropout_masks)
        return self.critic(features)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_dropout_masks(self):
        return self.linear_dropout_actor.get_dropout_mask(), self.linear_dropout_critic.get_dropout_mask()

    def reset_dropout_masks(self):
        self.linear_dropout_actor.reset_dropout_mask()
        self.linear_dropout_critic.reset_dropout_mask()

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        exporter = _PointNetSRUExporter(self, normalizer)
        exporter.eval().to("cpu")
        os.makedirs(path, exist_ok=True)
        scripted = torch.jit.script(exporter)
        scripted.save(os.path.join(path, filename))

    def export_onnx(self, path: str, filename: str = "policy.onnx", normalizer=None):
        exporter = _PointNetSRUONNXExporter(self, normalizer)
        exporter.eval().to("cpu")
        os.makedirs(path, exist_ok=True)
        rnn = exporter.rnn
        inputs = (
            torch.zeros(1, self.num_actor_obs),
            torch.zeros(rnn.num_layers, 1, rnn.hidden_size),
            torch.zeros(rnn.num_layers, 1, rnn.hidden_size),
        )
        torch.onnx.export(
            exporter,
            inputs,
            os.path.join(path, filename),
            input_names=["obs", "h_in", "c_in"],
            output_names=["actions", "h_out", "c_out"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "c_in": {1: "batch_size"},
                "actions": {0: "batch_size"},
                "h_out": {1: "batch_size"},
                "c_out": {1: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )


class _PointNetActorPipeline(nn.Module):
    """Shared stateless actor pipeline used by both exporters."""

    def __init__(self, actor_critic: ActorCriticSRUPointNet, normalizer=None):
        super().__init__()
        self.pointnet = copy.deepcopy(actor_critic.actor_pointnet)
        self.rnn = copy.deepcopy(actor_critic.memory_a.rnn)
        self.linear = copy.deepcopy(actor_critic.linear_dropout_actor.linear)
        self.activation = copy.deepcopy(actor_critic.linear_dropout_actor.activation)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()
        self.num_point_features = actor_critic.num_point_features
        self.num_points = actor_critic.num_points
        self.point_dim = actor_critic.point_dim

    def actor_forward(
        self,
        observations: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = self.normalizer(observations)
        other_obs = observations[..., : -self.num_point_features]
        points = observations[..., -self.num_point_features :].reshape(-1, self.num_points, self.point_dim)
        point_features = self.pointnet(points)
        features = torch.cat([point_features, other_obs], dim=-1)
        features, (hidden_state, cell_state) = self.rnn(features.unsqueeze(0), (hidden_state, cell_state))
        features = self.activation(self.linear(features.squeeze(0)))
        return self.actor(features), hidden_state, cell_state


class _PointNetSRUExporter(_PointNetActorPipeline):
    def __init__(self, actor_critic: ActorCriticSRUPointNet, normalizer=None):
        super().__init__(actor_critic, normalizer)
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, observations: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()
        actions, hidden_state, cell_state = self.actor_forward(
            observations, self.hidden_state, self.cell_state
        )
        self.hidden_state[:] = hidden_state
        self.cell_state[:] = cell_state
        return actions

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
        self.cell_state.zero_()


class _PointNetSRUONNXExporter(_PointNetActorPipeline):
    def forward(
        self,
        observations: torch.Tensor,
        hidden_state: torch.Tensor,
        cell_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.actor_forward(observations, hidden_state, cell_state)
