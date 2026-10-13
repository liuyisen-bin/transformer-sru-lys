"""Range-image CNN agent configurations for PM01 navigation."""

from isaaclab.utils import configclass

from engineai_rl_lab.tasks.navigation.config.rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRangeImageActorCriticCfg,
)


def _range_image_policy() -> RslRlRangeImageActorCriticCfg:
    return RslRlRangeImageActorCriticCfg(
        class_name="ActorCriticSRURangeImage",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_hidden_size=512,
        rnn_type="lstm_sru",
        rnn_num_layers=1,
        dropout=0.2,
        # 须与雷达一致：num_points = H*W，max_distance = RayCasterCfg.max_distance。
        num_points=656,
        point_dim=3,
        height_input_dims=(64, 7, 7),
        range_image_shape=(16, 41),
        range_image_feature_dim=128,
        range_image_batch_size=256,
        range_image_max_distance=5.0,
    )


@configclass
class PM01NavRangeImagePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner using an ordered LiDAR range-image CNN."""

    num_steps_per_env = 16
    max_iterations = 8000
    save_interval = 400
    logger = "tensorboard"
    seed = 60
    wandb_project = "isaaclab_nav_pm01"
    experiment_name = "pm01_navigation_range_image_ppo"
    empirical_normalization = False
    reward_shifting_value = 0.05
    policy: RslRlRangeImageActorCriticCfg = _range_image_policy()
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.02,
        use_clipped_value_loss=True,
        clip_param=0.2,
        value_clip_param=0.2,
        entropy_coef=0.00375,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        pointnet_freeze_iterations=200,
        pointnet_partial_iterations=400,
        pointnet_lr_scale=0.1,
    )
