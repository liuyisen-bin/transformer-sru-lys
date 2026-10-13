"""Gym registration for G1 Transformer-SRU with Unitree locomotion."""

import gymnasium as gym

from . import agents, navigation_env_cfg


def _register(task_id: str, env_cfg, runner_cfg) -> None:
    gym.register(
        id=task_id,
        entry_point="engineai_rl_lab.tasks.navigation:NavigationEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register(
    "Isaac-Navigation-PPO-G1-29DOF-v0",
    navigation_env_cfg.G1NavigationEnvCfg,
    agents.range_image_rsl_rl_cfg.G1NavRangeImagePPORunnerCfg,
)

__all__ = [
    "navigation_env_cfg",
    "agents",
]
