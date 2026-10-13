#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

# per frivik

class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = get_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        # seems that we get better performance without init
        # self.init_memory_weights(self.memory_a, 0.001, 0.)
        # self.init_memory_weights(self.memory_c, 0.001, 0.)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

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

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)
        # self.distribution = Normal(mean, mean * 0.0 + self.log_std.exp())

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        """Export policy as a JIT-scripted module for inference.

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.pt".
            normalizer: Optional normalizer module for observations.
        """
        import os

        exporter = _MLPPolicyExporter(self.actor, normalizer)
        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        scripted = torch.jit.script(exporter)
        scripted.save(filepath)
        print(f"[ActorCritic] Exported JIT policy to: {filepath}")

    def export_onnx(self, path: str, filename: str = "policy.onnx", normalizer=None):
        """Export policy as an ONNX model for inference.

        The ONNX model has the following interface:
            Inputs:
                - obs: Observation tensor of shape (1, num_obs)
            Outputs:
                - actions: Action tensor of shape (1, num_actions)

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.onnx".
            normalizer: Optional normalizer module for observations.
        """
        import os

        exporter = _MLPPolicyONNXExporter(self.actor, normalizer)
        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        # Infer input size from first layer
        num_obs = self.actor[0].in_features

        # Create dummy input
        dummy_obs = torch.zeros(1, num_obs)

        # Export to ONNX
        torch.onnx.export(
            exporter,
            dummy_obs,
            filepath,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "actions": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[ActorCritic] Exported ONNX policy to: {filepath}")
        print(f"  - Input 'obs': shape (1, {num_obs})")
        print(f"  - Output 'actions'")


class _MLPPolicyExporter(nn.Module):
    """JIT-exportable wrapper for non-recurrent ActorCritic."""

    def __init__(self, actor, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))

    @torch.jit.export
    def reset(self):
        pass


class _MLPPolicyONNXExporter(nn.Module):
    """ONNX-exportable wrapper for non-recurrent ActorCritic."""

    def __init__(self, actor, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(self.normalizer(x))


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        print("invalid activation function!")
        return None
