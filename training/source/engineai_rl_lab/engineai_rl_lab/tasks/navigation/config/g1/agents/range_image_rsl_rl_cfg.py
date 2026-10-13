"""Range-image PPO runner for G1 navigation."""

from isaaclab.utils import configclass

from ...pm01.agents.range_image_rsl_rl_cfg import (
    PM01NavRangeImagePPORunnerCfg,
    _range_image_policy,
)


def _transformer_sru_policy():
    """Keep the proven G1 SRU policy and add a two-layer spatial Transformer."""

    policy = _range_image_policy()
    policy.transformer_num_layers = 2
    policy.transformer_num_heads = 4
    policy.transformer_feedforward_dim = 512
    policy.transformer_dropout = 0.1
    return policy


@configclass
class G1NavRangeImagePPORunnerCfg(PM01NavRangeImagePPORunnerCfg):
    """Reuse the proven G1 SRU settings and add a spatial Transformer."""

    save_interval = 100
    wandb_project = "isaaclab_nav_g1_transformer_sru"
    experiment_name = "g1_29dof_unitree_transformer_sru_ppo"
    policy = _transformer_sru_policy()


__all__ = ["G1NavRangeImagePPORunnerCfg"]
