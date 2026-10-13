#!/usr/bin/env python3
# Copyright (c) 2022-2025, Fan Yang and Per Frivik, ETH Zurich.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Train a navigation policy using RSL-RL (PPO).

Usage:
    python training/scripts/nav/train.py --task Isaac-Navigation-PPO-G1-29DOF-v0 --num_envs <num> [options]

Arguments:
    --task               Task name (defaults to the G1 navigation task)
    --num_envs           Number of parallel environments
    --smoke_test         Use the reduced G1 PLAY environment for a G1 chain smoke test
    --visual_demo        Use one obstacle-free G1 environment for low-memory visualization
    --seed               Random seed
    --max_iterations     Training iterations
    --run_name           Custom run name for logging
    --resume             Resume training from a checkpoint
    --checkpoint         Explicit checkpoint path (implies --resume)
    --load_run           Run-directory regex used when resolving a checkpoint
    --load_checkpoint    Checkpoint-file regex used when resolving a checkpoint
    --no_load_optimizer  Restore policy weights without optimizer state
    --video              Enable video recording
    --video_length       Video length in steps (default: 200)
    --video_interval     Recording interval in steps (default: 2000)

Examples:
    python training/scripts/nav/train.py --task Isaac-Navigation-PPO-G1-29DOF-v0 --num_envs 2048 --headless
    python training/scripts/nav/train.py --task Isaac-Navigation-PPO-G1-29DOF-v0 --resume
    python training/scripts/nav/train.py --checkpoint checkpoints/model_24999.pt --max_iterations 30000

Logs saved to: logs/rsl_rl/<experiment_name>/<timestamp>/
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

parser = argparse.ArgumentParser(description="Train a navigation policy with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--smoke_test",
    action="store_true",
    help="For the G1 29DOF task only, use its reduced PLAY environment for a chain smoke test.",
)
parser.add_argument(
    "--visual_demo",
    action="store_true",
    help="For G1 only, use one small obstacle-free terrain for low-memory visualization.",
)
parser.add_argument("--task", type=str, default="Isaac-Navigation-PPO-G1-29DOF-v0", choices=["Isaac-Navigation-PPO-G1-29DOF-v0"], help="Name of the G1 navigation task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Run name appended to the log directory.")
parser.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path; implies --resume.")
parser.add_argument("--load_run", type=str, default=None, help="Run-directory regex used to find a checkpoint.")
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Checkpoint-file regex used to select a model within the run.",
)
parser.add_argument(
    "--no_load_optimizer",
    action="store_true",
    help="Load policy weights and iteration only; initialize a fresh optimizer.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / rsl_rl imports require a launched simulation.
import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import engineai_rl_lab.tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


class LegacyRslRlVecEnvWrapper(RslRlVecEnvWrapper):
    """Adapt Isaac Lab's TensorDict API to the SRU RSL-RL 2.x API."""

    def get_observations(self):
        observations = super().get_observations()
        return observations["policy"], {"observations": observations}

    def step(self, actions):
        observations, rewards, dones, extras = super().step(actions)
        extras["observations"] = observations
        return observations["policy"], rewards, dones, extras


if str(args_cli.device).startswith("cuda"):
    torch.cuda.set_device(args_cli.device)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train navigation policy with RSL-RL."""
    g1_smoke_test = args_cli.smoke_test and args_cli.task == "Isaac-Navigation-PPO-G1-29DOF-v0"
    g1_visual_demo = args_cli.visual_demo and args_cli.task == "Isaac-Navigation-PPO-G1-29DOF-v0"
    if args_cli.smoke_test and args_cli.visual_demo:
        raise ValueError("Use only one of --smoke_test or --visual_demo.")
    if g1_visual_demo:
        from engineai_rl_lab.tasks.navigation.config.g1.navigation_env_cfg import G1NavigationEnvCfg_VISUAL

        env_cfg = G1NavigationEnvCfg_VISUAL()
    elif g1_smoke_test:
        from engineai_rl_lab.tasks.navigation.config.g1.navigation_env_cfg import G1NavigationEnvCfg_PLAY

        env_cfg = G1NavigationEnvCfg_PLAY()
    else:
        env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")

    if g1_smoke_test or g1_visual_demo:
        agent_cfg.algorithm.num_mini_batches = 1
        print(
            "[INFO] G1 reduced run: overriding algorithm.num_mini_batches=1; "
            "formal training configuration is unchanged."
        )

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.resume or args_cli.checkpoint is not None:
        agent_cfg.resume = True
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.load_checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.load_checkpoint

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = LegacyRslRlVecEnvWrapper(env)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Resolve the checkpoint before creating the new run directory so lookup
    # cannot pick the empty destination run.
    resume_path = None
    if agent_cfg.resume:
        if args_cli.checkpoint is not None:
            resume_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Checkpoint does not exist: {resume_path}")
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=agent_cfg.load_run,
                checkpoint=agent_cfg.load_checkpoint,
            )
        print(f"[INFO] Resuming training from checkpoint: {resume_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if resume_path is not None:
        runner.load(resume_path, load_optimizer=not args_cli.no_load_optimizer)
        print(f"[INFO] Restored checkpoint at iteration {runner.current_learning_iteration}")

    runner.add_git_repo_to_log(__file__)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # On resume, max_iterations is the final iteration, not additional iterations.
    num_learning_iterations = agent_cfg.max_iterations
    if resume_path is not None:
        num_learning_iterations -= runner.current_learning_iteration
        if num_learning_iterations <= 0:
            raise ValueError(
                f"Checkpoint iteration {runner.current_learning_iteration} already reached "
                f"the requested max_iterations={agent_cfg.max_iterations}."
            )

    try:
        runner.learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=True)
    except KeyboardInterrupt:
        interrupted_path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}.pt")
        runner.save(interrupted_path)
        print(f"\n[INFO] Training interrupted; checkpoint saved to: {interrupted_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
