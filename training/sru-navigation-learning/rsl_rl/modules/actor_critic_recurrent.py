#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  Modified by Fan Yang, Per Frivik, Robotic Systems Lab, ETH Zurich 2025
#  SPDX-License-Identifier: BSD-3-Clause

"""Recurrent Actor-Critic module with LSTM/GRU memory."""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic, get_activation
from rsl_rl.networks.sru_memory import LSTM_SRU
from rsl_rl.utils import unpad_trajectories


class ActorCriticRecurrent(ActorCritic):
    """Recurrent Actor-Critic network.

    Extends the base ActorCritic with recurrent memory (LSTM, GRU, or LSTM_SRU).

    Args:
        num_actor_obs: Dimension of actor observations.
        num_critic_obs: Dimension of critic observations.
        num_actions: Number of action dimensions.
        actor_hidden_dims: List of hidden layer dimensions for actor MLP.
        critic_hidden_dims: List of hidden layer dimensions for critic MLP.
        activation: Activation function name.
        rnn_type: Type of RNN ('lstm', 'gru', or 'lstm_sru').
        dropout: Dropout probability.
        rnn_hidden_size: Hidden size of the RNN.
        rnn_num_layers: Number of RNN layers.
        init_noise_std: Initial standard deviation for action noise.
    """

    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        rnn_type="lstm",
        dropout=0.2,
        rnn_hidden_size=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )

        super().__init__(
            num_actor_obs=actor_hidden_dims[0],
            num_critic_obs=critic_hidden_dims[0],
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )

        activation = get_activation(activation)

        self.linear_dropout_actor = LinearConstDropout(
            in_features=rnn_hidden_size,
            out_features=actor_hidden_dims[0],
            dropout_p=dropout,
            activation=activation,
        )
        self.linear_dropout_critic = LinearConstDropout(
            in_features=rnn_hidden_size,
            out_features=critic_hidden_dims[0],
            dropout_p=dropout,
            activation=activation,
        )

        self.memory_a = Memory(num_actor_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        self.memory_c = Memory(num_critic_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)

        print(f"Actor RNN: {self.memory_a}")
        print(f"Critic RNN: {self.memory_c}")

    def reset(self, dones=None):
        """Reset hidden states for terminated episodes."""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        """Sample actions from the policy."""
        input_a = self.memory_a(observations, masks, hidden_states)
        input_a = self.linear_dropout_actor(input_a.squeeze(0), dropout_masks=dropout_masks)
        return super().act(input_a)

    def act_inference(self, observations, masks=None, hidden_states=None, dropout_masks=None):
        """Get deterministic actions for inference."""
        input_a = self.memory_a(observations, masks, hidden_states)
        input_a = self.linear_dropout_actor(input_a.squeeze(0), dropout_masks=dropout_masks)
        return super().act_inference(input_a)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, dropout_masks=None):
        """Evaluate state values."""
        input_c = self.memory_c(critic_observations, masks, hidden_states)
        input_c = self.linear_dropout_critic(input_c.squeeze(0), dropout_masks=dropout_masks)
        return super().evaluate(input_c)

    def get_hidden_states(self):
        """Get current hidden states from actor and critic memories."""
        return self.memory_a.hidden_states, self.memory_c.hidden_states

    def get_dropout_masks(self):
        """Get current dropout masks."""
        return self.linear_dropout_actor.get_dropout_mask(), self.linear_dropout_critic.get_dropout_mask()

    def reset_dropout_masks(self):
        """Reset dropout masks for new episode."""
        self.linear_dropout_actor.reset_dropout_mask()
        self.linear_dropout_critic.reset_dropout_mask()

    def get_actor_parameters(self):
        """Return Actor Parameters including the RNN and dropout layer parameters."""
        return (
            list(self.actor.parameters())
            + list(self.memory_a.parameters())
            + list(self.linear_dropout_actor.parameters())
            + [self.log_std]
        )

    def get_critic_parameters(self):
        """Return Critic Parameters including the RNN and dropout layer parameters."""
        return (
            list(self.critic.parameters())
            + list(self.memory_c.parameters())
            + list(self.linear_dropout_critic.parameters())
        )

    def export_jit(self, path: str, filename: str = "policy.pt", normalizer=None):
        """Export policy as a JIT-scripted module for inference.

        Supports LSTM, GRU, and LSTM_SRU RNN types. The exported module maintains
        internal hidden states and provides a reset() method to clear them.

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.pt".
            normalizer: Optional normalizer module for observations.
        """
        import os

        rnn = self.memory_a.rnn
        rnn_type = type(rnn).__name__.lower()

        # Create appropriate exporter based on RNN type
        if rnn_type == "lstm":
            exporter = _LSTMPolicyExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
        elif rnn_type == "gru":
            exporter = _GRUPolicyExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
        elif rnn_type == "lstm_sru":
            exporter = _LSTMSRUPolicyExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
        else:
            raise NotImplementedError(f"Unsupported RNN type for JIT export: {rnn_type}")

        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)
        scripted = torch.jit.script(exporter)
        scripted.save(filepath)
        print(f"[ActorCriticRecurrent] Exported JIT policy ({rnn_type}) to: {filepath}")

    def export_onnx(self, path: str, filename: str = "policy.onnx", normalizer=None, num_obs: int = None):
        """Export policy as an ONNX model for inference.

        Supports LSTM, GRU, and LSTM_SRU RNN types. For recurrent networks, hidden states
        are exposed as explicit inputs and outputs since ONNX doesn't support stateful ops.

        The ONNX model has the following interface:
            Inputs:
                - obs: Observation tensor of shape (1, num_obs)
                - h_in: Hidden state tensor of shape (num_layers, 1, hidden_size)
                - c_in: Cell state tensor of shape (num_layers, 1, hidden_size) [LSTM/LSTM_SRU only]
            Outputs:
                - actions: Action tensor of shape (1, num_actions)
                - h_out: Updated hidden state
                - c_out: Updated cell state [LSTM/LSTM_SRU only]

        Args:
            path: Directory path to save the exported model.
            filename: Name of the exported file. Defaults to "policy.onnx".
            normalizer: Optional normalizer module for observations.
            num_obs: Number of observations (inferred from memory if not provided).
        """
        import os

        rnn = self.memory_a.rnn
        rnn_type = type(rnn).__name__.lower()

        # Infer num_obs from RNN input size if not provided
        if num_obs is None:
            num_obs = rnn.input_size

        # Create appropriate ONNX exporter based on RNN type
        # Note: Input/output names follow deployment convention: h_in/h_out, c_in/c_out
        if rnn_type == "lstm":
            exporter = _LSTMPolicyONNXExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
            input_names = ["obs", "h_in", "c_in"]
            output_names = ["actions", "h_out", "c_out"]
            # Create dummy inputs for LSTM (obs, hidden, cell)
            dummy_obs = torch.zeros(1, num_obs)
            dummy_hidden = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
            dummy_cell = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
            dummy_inputs = (dummy_obs, dummy_hidden, dummy_cell)
            dynamic_axes = {
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "c_in": {1: "batch_size"},
                "actions": {0: "batch_size"},
                "h_out": {1: "batch_size"},
                "c_out": {1: "batch_size"},
            }
        elif rnn_type == "gru":
            exporter = _GRUPolicyONNXExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
            input_names = ["obs", "h_in"]
            output_names = ["actions", "h_out"]
            # Create dummy inputs for GRU (obs, hidden)
            dummy_obs = torch.zeros(1, num_obs)
            dummy_hidden = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
            dummy_inputs = (dummy_obs, dummy_hidden)
            dynamic_axes = {
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "actions": {0: "batch_size"},
                "h_out": {1: "batch_size"},
            }
        elif rnn_type == "lstm_sru":
            exporter = _LSTMSRUPolicyONNXExporter(
                actor=self.actor,
                rnn=rnn,
                linear_dropout=self.linear_dropout_actor,
                normalizer=normalizer,
            )
            input_names = ["obs", "h_in", "c_in"]
            output_names = ["actions", "h_out", "c_out"]
            # Create dummy inputs for LSTM_SRU (obs, hidden, cell)
            dummy_obs = torch.zeros(1, num_obs)
            dummy_hidden = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
            dummy_cell = torch.zeros(rnn.num_layers, 1, rnn.hidden_size)
            dummy_inputs = (dummy_obs, dummy_hidden, dummy_cell)
            dynamic_axes = {
                "obs": {0: "batch_size"},
                "h_in": {1: "batch_size"},
                "c_in": {1: "batch_size"},
                "actions": {0: "batch_size"},
                "h_out": {1: "batch_size"},
                "c_out": {1: "batch_size"},
            }
        else:
            raise NotImplementedError(f"Unsupported RNN type for ONNX export: {rnn_type}")

        exporter.eval()
        exporter.to("cpu")

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, filename)

        # Export to ONNX
        torch.onnx.export(
            exporter,
            dummy_inputs,
            filepath,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[ActorCriticRecurrent] Exported ONNX policy ({rnn_type}) to: {filepath}")

        # Print model info for deployment
        print(f"  - Input 'obs': shape (1, {num_obs})")
        print(f"  - Input 'h_in': shape ({rnn.num_layers}, 1, {rnn.hidden_size})")
        if rnn_type in ("lstm", "lstm_sru"):
            print(f"  - Input 'c_in': shape ({rnn.num_layers}, 1, {rnn.hidden_size})")
        print(f"  - Output 'actions', 'h_out'" + (", 'c_out'" if rnn_type in ("lstm", "lstm_sru") else ""))
        print(f"  - Note: Initialize h_in/c_in to zeros at episode start")


class _LSTMPolicyExporter(nn.Module):
    """JIT-exportable wrapper for ActorCriticRecurrent with LSTM (optimized)."""

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Register hidden state buffers
        self.register_buffer("hidden_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()

        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = self.activation(self.linear(x.squeeze(0)))
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
        self.cell_state.zero_()


class _GRUPolicyExporter(nn.Module):
    """JIT-exportable wrapper for ActorCriticRecurrent with GRU (optimized)."""

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Register hidden state buffer
        self.register_buffer("hidden_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()

        x = self.normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h
        x = self.activation(self.linear(x.squeeze(0)))
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()


class _LSTMSRUPolicyExporter(nn.Module):
    """JIT-exportable wrapper for ActorCriticRecurrent with LSTM_SRU (optimized)."""

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # Register hidden state buffers for LSTM_SRU
        self.register_buffer("hidden_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(rnn.num_layers, 1, rnn.hidden_size))

    def forward(self, x: torch.Tensor, reset: bool = False) -> torch.Tensor:
        if reset:
            self.hidden_state.zero_()
            self.cell_state.zero_()

        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = self.activation(self.linear(x.squeeze(0)))
        return self.actor(x)

    @torch.jit.export
    def reset(self):
        self.hidden_state.zero_()
        self.cell_state.zero_()


# =============================================================================
# ONNX Policy Exporters
# =============================================================================
# These exporters are designed for ONNX export where hidden states must be
# explicit inputs/outputs (ONNX doesn't support stateful operations).


class _LSTMPolicyONNXExporter(nn.Module):
    """ONNX-exportable wrapper for ActorCriticRecurrent with LSTM.

    Unlike JIT exporters, ONNX requires hidden states as explicit I/O.
    """

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit hidden state I/O.

        Args:
            obs: Observation tensor of shape (batch, num_obs)
            hidden_state: LSTM hidden state of shape (num_layers, batch, hidden_size)
            cell_state: LSTM cell state of shape (num_layers, batch, hidden_size)

        Returns:
            Tuple of (actions, new_hidden_state, new_cell_state)
        """
        x = self.normalizer(obs)
        x, (h_out, c_out) = self.rnn(x.unsqueeze(0), (hidden_state, cell_state))
        x = self.activation(self.linear(x.squeeze(0)))
        actions = self.actor(x)
        return actions, h_out, c_out


class _GRUPolicyONNXExporter(nn.Module):
    """ONNX-exportable wrapper for ActorCriticRecurrent with GRU.

    Unlike JIT exporters, ONNX requires hidden states as explicit I/O.
    """

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with explicit hidden state I/O.

        Args:
            obs: Observation tensor of shape (batch, num_obs)
            hidden_state: GRU hidden state of shape (num_layers, batch, hidden_size)

        Returns:
            Tuple of (actions, new_hidden_state)
        """
        x = self.normalizer(obs)
        x, h_out = self.rnn(x.unsqueeze(0), hidden_state)
        x = self.activation(self.linear(x.squeeze(0)))
        actions = self.actor(x)
        return actions, h_out


class _LSTMSRUPolicyONNXExporter(nn.Module):
    """ONNX-exportable wrapper for ActorCriticRecurrent with LSTM_SRU.

    Unlike JIT exporters, ONNX requires hidden states as explicit I/O.
    """

    def __init__(self, actor, rnn, linear_dropout, normalizer=None):
        super().__init__()
        import copy
        self.actor = copy.deepcopy(actor)
        self.rnn = copy.deepcopy(rnn)
        self.linear = copy.deepcopy(linear_dropout.linear)
        self.activation = copy.deepcopy(linear_dropout.activation)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

    def forward(
        self, obs: torch.Tensor, hidden_state: torch.Tensor, cell_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with explicit hidden state I/O.

        Args:
            obs: Observation tensor of shape (batch, num_obs)
            hidden_state: LSTM_SRU hidden state of shape (num_layers, batch, hidden_size)
            cell_state: LSTM_SRU cell state of shape (num_layers, batch, hidden_size)

        Returns:
            Tuple of (actions, new_hidden_state, new_cell_state)
        """
        x = self.normalizer(obs)
        x, (h_out, c_out) = self.rnn(x.unsqueeze(0), (hidden_state, cell_state))
        x = self.activation(self.linear(x.squeeze(0)))
        actions = self.actor(x)
        return actions, h_out, c_out


class Memory(torch.nn.Module):
    """Memory module supporting LSTM, GRU, and LSTM_SRU.

    Args:
        input_size: Input dimension.
        type: RNN type ('lstm', 'gru', or 'lstm_sru').
        num_layers: Number of RNN layers.
        hidden_size: Hidden state size.
    """

    def __init__(self, input_size, type="lstm", num_layers=1, hidden_size=256):
        super().__init__()
        # RNN type selection
        if type.lower() == "gru":
            rnn_cls = nn.GRU
        elif type.lower() == "lstm":
            rnn_cls = nn.LSTM
        elif type.lower() in ("lstm_sru", "lstm_a_gate"):
            # lstm_a_gate kept for backwards compatibility
            rnn_cls = LSTM_SRU
        else:
            raise ValueError(f"RNN type {type} not supported. Use 'lstm', 'gru', or 'lstm_sru'.")

        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None

    def forward(self, input, masks=None, hidden_states=None):
        """Forward pass through the memory module."""
        batch_mode = masks is not None
        if batch_mode:
            # input (L, B, D)
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # input: (B, D)
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones, use_random_init=True):
        """Reset hidden states for terminated episodes."""
        if self.hidden_states is not None:
            for hidden_state in self.hidden_states:
                if dones.sum() > 0:  # Only do this if there are terminated episodes
                    if use_random_init:
                        # Create a shuffled copy of the current hidden states
                        shuffled_hidden = hidden_state.clone()
                        # Shuffle along the batch dimension
                        batch_size = hidden_state.size(-2)
                        shuffle_indices = torch.randperm(batch_size, device=hidden_state.device)
                        shuffled_hidden = shuffled_hidden[..., shuffle_indices, :]

                        # Assign shuffled hidden states to terminated environments
                        hidden_state[..., dones == 1, :] = shuffled_hidden[..., dones == 1, :]
                    else:
                        # Use zero initialization
                        hidden_state[..., dones == 1, :] = 0.0


class SimpleConsistentDropout(nn.Module):
    """Dropout that maintains consistent masks across a trajectory."""

    def __init__(self, p):
        super(SimpleConsistentDropout, self).__init__()
        self.p = p
        self.scale_factor = 1.0 / (1.0 - p)

    def forward(self, x, dropout_masks=None):
        if self.training:
            if dropout_masks is None:
                dropout_masks = torch.empty_like(x).bernoulli_(1 - self.p)
            out = x * dropout_masks * self.scale_factor
            return out, dropout_masks
        else:
            return x, None


class LinearConstDropout(nn.Module):
    """Linear layer with activation and consistent dropout.

    Combines a Linear layer, an activation, and consistent dropout that
    maintains the same mask across a trajectory.
    """

    def __init__(self, in_features, out_features, dropout_p, activation):
        super(LinearConstDropout, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.dropout = SimpleConsistentDropout(p=dropout_p)
        self.dropout_masks = None

    def forward(self, x, dropout_masks=None):
        x = self.linear(x)
        x = self.activation(x)
        if dropout_masks is None:
            x, self.dropout_masks = self.dropout(x, dropout_masks=self.dropout_masks)
        else:
            x, _ = self.dropout(x, dropout_masks)
        return x

    def get_dropout_mask(self):
        return self.dropout_masks

    def reset_dropout_mask(self):
        self.dropout_masks = None
