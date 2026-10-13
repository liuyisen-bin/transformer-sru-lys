#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Play a trained navigation policy with automatic checkpoint loading.

Usage:
    python training/scripts/nav/play.py --task Isaac-Navigation-PPO-G1-29DOF-v0 [options]

Arguments:
    --task                   Task name (defaults to the G1 navigation task)
    --checkpoint             Path to model checkpoint (.pt file)
    --use_last_checkpoint    Use latest checkpoint from logs (default behavior)
    --num_envs              Number of parallel environments
    --video                 Enable video recording
    --video_length          Video length in steps (default: 200)

Examples:
    python training/scripts/nav/play.py --task Isaac-Navigation-PPO-G1-29DOF-v0
    python training/scripts/nav/play.py --task Isaac-Navigation-PPO-G1-29DOF-v0 --checkpoint path/to/model.pt
    python training/scripts/nav/play.py --task Isaac-Navigation-PPO-G1-29DOF-v0 --num_envs 16 --export_onnx --checkpoint path/to/model.pt

Note: Automatically finds latest checkpoint if --checkpoint not specified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Prefer the SRU package in this workspace over a stale PYTHONPATH checkout.
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_RSL_RL = _WORKSPACE_ROOT / "sru-navigation-learning"
_LOCAL_ENGINEAI_RL_LAB = _WORKSPACE_ROOT / "source" / "engineai_rl_lab"
if _LOCAL_RSL_RL.is_dir():
    sys.path.insert(0, str(_LOCAL_RSL_RL))
if _LOCAL_ENGINEAI_RL_LAB.is_dir():
    sys.path.insert(0, str(_LOCAL_ENGINEAI_RL_LAB))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-PPO-G1-29DOF-v0", choices=["Isaac-Navigation-PPO-G1-29DOF-v0"], help="Name of the G1 navigation task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="Use last checkpoint from logs.")
parser.add_argument("--export_jit", action="store_true", default=False, help="Export policy as JIT module.")
parser.add_argument("--export_onnx", action="store_true", default=False, help="Export policy as ONNX model.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / rsl_rl imports require a launched simulation.
import gymnasium as gym
import os
import re
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import engineai_rl_lab.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_onnx
from engineai_rl_lab.tasks.navigation.config.g1.navigation_env_cfg import G1NavigationEnvCfg_PLAY


class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Adapt Isaac Lab's TensorDict API to the SRU RSL-RL 2.x API."""

    def get_observations(self):
        observations = super().get_observations()
        return observations["policy"], {"observations": observations}

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        extras["observations"] = observations
        return observations["policy"], rewards, dones, extras


def find_latest_checkpoint(log_path: str, checkpoint_pattern: str = "model_.*.pt") -> str:
    """Find the latest checkpoint file in the log directory.

    Args:
        log_path: Base log directory path
        checkpoint_pattern: Regex pattern for checkpoint files

    Returns:
        Path to the latest checkpoint file
    """
    if not os.path.exists(log_path):
        raise ValueError(f"Log path does not exist: {log_path}")

    run_dirs = []
    for entry in os.scandir(log_path):
        if entry.is_dir() and re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", entry.name):
            run_dirs.append(entry.name)

    if not run_dirs:
        raise ValueError(f"No run directories found in: {log_path}")

    run_dirs.sort()
    latest_run = run_dirs[-1]
    run_path = os.path.join(log_path, latest_run)

    checkpoint_files = []
    for f in os.listdir(run_path):
        if re.match(checkpoint_pattern, f):
            checkpoint_files.append(f)

    if not checkpoint_files:
        raise ValueError(f"No checkpoint files matching '{checkpoint_pattern}' found in: {run_path}")

    checkpoint_files.sort(key=lambda m: f"{m:0>15}")
    latest_checkpoint = checkpoint_files[-1]

    return os.path.join(run_path, latest_checkpoint)


def load_checkpoint_with_fallback(runner: OnPolicyRunner, checkpoint_path: str, load_optimizer: bool = True):
    """Load checkpoint with fallback for PyTorch compatibility issues.

    Args:
        runner: RSL-RL runner instance
        checkpoint_path: Path to checkpoint file
        load_optimizer: Whether to load optimizer state
    """
    print(f"[INFO] Loading checkpoint from: {checkpoint_path}")

    loaded_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if runner.is_mdpo:
        runner.alg.actor_critic_1.load_state_dict(loaded_dict["model_state_dict"], strict=True)
        runner.alg.actor_critic_2.load_state_dict(loaded_dict["model_state_dict"], strict=True)
    else:
        runner.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"], strict=True)

    if runner.empirical_normalization:
        runner.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
        runner.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])

    if load_optimizer:
        if runner.is_mdpo:
            runner.alg.optimizer_1.load_state_dict(loaded_dict["optimizer_state_dict"])
        else:
            runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])

    runner.current_learning_iteration = loaded_dict["iter"]
    print(f"[INFO] Loaded checkpoint from iteration {loaded_dict['iter']}")


def export_policy_jit(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as JIT module to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    print(f"[INFO] Exporting JIT policy to: {export_dir}")
    actor_critic.export_jit(path=export_dir, filename="policy.pt", normalizer=normalizer)
    print(f"[INFO] JIT export complete!")


def export_policy_onnx(runner: OnPolicyRunner, checkpoint_path: str):
    """Export policy as ONNX model to an 'export' folder next to the checkpoint.

    Args:
        runner: RSL-RL runner instance with loaded policy
        checkpoint_path: Path to the checkpoint file (used to determine export location)
    """
    checkpoint_dir = os.path.dirname(checkpoint_path)
    export_dir = os.path.join(checkpoint_dir, "export")

    if runner.is_mdpo:
        actor_critic = runner.alg.actor_critic_1
    else:
        actor_critic = runner.alg.actor_critic

    normalizer = runner.obs_normalizer if runner.empirical_normalization else None

    if not hasattr(actor_critic, "export_onnx"):
        raise NotImplementedError(
            f"ONNX export not implemented for {type(actor_critic).__name__}. "
            "Please add an export_onnx method to this module."
        )

    print(f"[INFO] Exporting ONNX policy to: {export_dir}")
    actor_critic.export_onnx(path=export_dir, filename="policy.onnx", normalizer=normalizer)
    print(f"[INFO] ONNX export complete!")


def main():
    """Play navigation policy with RSL-RL."""
    if str(args_cli.device).startswith("cuda"):
        torch.cuda.set_device(args_cli.device)

    spec = gym.spec(args_cli.task)
    agent_cfg_class = spec.kwargs.get("rsl_rl_cfg_entry_point")

    # Playback only needs a small terrain set.  The registered environment uses
    # the full training terrain (6 x 30 tiles), which can exhaust GPU memory on
    # desktop cards even when --num_envs is set to 1.
    env_cfg = G1NavigationEnvCfg_PLAY()
    agent_cfg: RslRlOnPolicyRunnerCfg = agent_cfg_class()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = LegacyRslRlVecEnvWrapper(env)

    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = find_latest_checkpoint(log_root_path, checkpoint_pattern="model_.*.pt")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    load_checkpoint_with_fallback(runner, resume_path)

    if args_cli.export_jit:
        export_policy_jit(runner, resume_path)

    if args_cli.export_onnx:
        export_policy_onnx(runner, resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
