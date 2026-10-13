# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Custom RSL-RL configuration classes for navigation tasks."""

from dataclasses import MISSING
from typing import Literal, Optional

from isaaclab.utils import configclass


@configclass
class RslRlPpoActorCriticCfg:
    """Configuration for the PPO actor-critic networks with navigation extensions."""

    class_name: str = "ActorCritic"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    rnn_type: str = "lstm"
    """The type of RNN to use."""

    rnn_hidden_size: int = 256
    """The hidden size of the RNN."""

    rnn_num_layers: int = 1
    """The number of layers in the RNN."""

    dropout: float = 0.0
    """The dropout rate for the first layer of the actor and critic networks."""

    # Visual inputs
    num_cameras: int = 1
    """Number of depth cameras encoded into the observation (1 or 2)."""
    
    image_input_dims: tuple[int, int, int] = (64, 5, 8)
    """Encoded depth feature shape as (C, H, W)."""
    
    height_input_dims: tuple[int, int, int] = (64, 7, 7)
    """Encoded height scan feature shape as (C, H, W). Default is (64, 7, 7) for 64*7*7=3136 features."""

    # Point-cloud inputs
    num_points: int = 656
    """Number of fixed LiDAR points at each policy step."""

    point_dim: int = 3
    """Number of coordinates per point."""

    pointnet_feature_dim: int = 128
    """Dimension of the global PointNet feature."""

    pointnet_batch_size: int = 256
    """Maximum number of point clouds encoded in one PointNet chunk."""


@configclass
class RslRlRangeImageActorCriticCfg(RslRlPpoActorCriticCfg):
    """Configuration for the ordered LiDAR range-image CNN policy."""

    range_image_shape: tuple[int, int] = (16, 41)
    """LiDAR scan shape as (vertical channels, horizontal samples)."""

    range_image_feature_dim: int = 128
    """Dimension of the global range-image feature."""

    range_image_batch_size: int = 256
    """Maximum number of scans encoded in one CNN chunk."""

    range_image_max_distance: float = 5.0
    """Range used to normalize XYZ values inside the CNN."""

    transformer_num_layers: int = 0
    """Number of Transformer encoder layers after the range-image CNN."""

    transformer_num_heads: int = 4
    """Number of self-attention heads in each Transformer layer."""

    transformer_feedforward_dim: int = 512
    """Hidden dimension of the Transformer feed-forward block."""

    transformer_dropout: float = 0.1
    """Dropout used inside Transformer encoder layers."""


@configclass
class RslRlPpoAlgorithmCfg:
    """Configuration for the PPO algorithm."""

    class_name: str = MISSING
    """The algorithm class name. Default is PPO."""

    value_loss_coef: float = MISSING
    """The coefficient for the value loss."""

    use_clipped_value_loss: bool = MISSING
    """Whether to use clipped value loss."""

    clip_param: float = MISSING
    """The clipping parameter for the policy."""

    value_clip_param: float = 0.2
    """The value clipping parameter. Default is 0.2."""

    entropy_coef: float = MISSING
    """The coefficient for the entropy loss."""

    num_learning_epochs: int = MISSING
    """The number of learning epochs per update."""

    num_mini_batches: int = MISSING
    """The number of mini-batches per update."""

    learning_rate: float = MISSING
    """The learning rate for the policy."""

    schedule: str = MISSING
    """The learning rate schedule."""

    gamma: float = MISSING
    """The discount factor."""

    lam: float = MISSING
    """The lambda parameter for Generalized Advantage Estimation (GAE)."""

    desired_kl: float = MISSING
    """The desired KL divergence."""

    max_grad_norm: float = MISSING
    """The maximum gradient norm."""

    pointnet_freeze_iterations: int = 500
    """Keep PointNet fully frozen before this iteration."""

    pointnet_partial_iterations: int = 2000
    """Train only the final PointNet layers until this iteration."""

    pointnet_lr_scale: float = 0.1
    """PointNet learning-rate multiplier relative to the policy learning rate."""


@configclass
class RslRlOnPolicyRunnerCfg:
    """Configuration of the runner for on-policy algorithms."""

    seed: Optional[int] = 42
    """The seed for the experiment. Default is 42."""

    device: str = "cuda:0"
    """The device for the rl-agent. Default is cuda:0."""

    num_steps_per_env: int = MISSING
    """The number of steps per environment per update."""

    max_iterations: int = MISSING
    """The maximum number of iterations."""

    empirical_normalization: bool = MISSING
    """Whether to use empirical normalization."""

    policy: RslRlPpoActorCriticCfg = MISSING
    """The policy configuration."""

    algorithm: RslRlPpoAlgorithmCfg = MISSING
    """The algorithm configuration."""

    reward_shifting_value: float = 0.0
    """The value to shift the reward by. Default is 0.0."""

    ##
    # Checkpointing parameters
    ##

    save_interval: int = MISSING
    """The number of iterations between saves."""

    experiment_name: str = MISSING
    """The experiment name."""

    run_name: str = ""
    """The run name. Default is empty string."""

    ##
    # Logging parameters
    ##

    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    """The logger to use. Default is tensorboard."""

    neptune_project: str = "isaaclab"
    """The neptune project name. Default is "isaaclab"."""

    wandb_project: str = "isaaclab"
    """The wandb project name. Default is "isaaclab"."""

    ##
    # Loading parameters
    ##

    resume: bool = False
    """Whether to resume. Default is False."""

    load_run: str = ".*"
    """The run directory to load. Default is ".*" (all)."""

    load_checkpoint: str = "model_.*.pt"
    """The checkpoint file to load. Default is "model_.*.pt" (all)."""
