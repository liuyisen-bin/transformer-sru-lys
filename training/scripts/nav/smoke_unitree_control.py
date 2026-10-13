#!/usr/bin/env python3
"""Exercise the actual navigation action adapter on flat ground without a learned navigator."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile

# Isaac's fast shutdown may exit before Python can propagate a failed assertion.
# Keep the pass/fail decision in a small parent process that does not import Isaac.
if __name__ == "__main__" and "--_status_path" not in sys.argv:
    with tempfile.TemporaryDirectory(prefix="unitree-control-smoke-") as temporary:
        status_path = Path(temporary) / "result.json"
        worker = subprocess.run([sys.executable, sys.argv[0], *sys.argv[1:],
                                 "--_status_path", str(status_path)])
        report = json.loads(status_path.read_text()) if status_path.is_file() else {}
        success = worker.returncode == 0 and report.get("passed") is True
        print(f"[control-smoke] {'PASS' if success else 'FAIL'}", flush=True)
        raise SystemExit(0 if success else 1)

TRAINING = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRAINING / "source" / "engineai_rl_lab"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=30.0)
parser.add_argument("--output", type=Path, default=TRAINING.parent / "experiments/unitree_control_smoke")
parser.add_argument("--_status_path", type=Path, help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 20.0 <= args.duration <= 120.0:
    parser.error("duration must be in [20, 120] seconds")
app = AppLauncher(args).app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.envs.manager_based_env_cfg import DefaultEventManagerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from engineai_rl_lab.tasks.navigation.assets.g1 import G1_29DOF_CFG, G1_FOOT_FRICTION
from engineai_rl_lab.tasks.navigation.config.g1.navigation_env_cfg import G1ActionsCfg, G1EventCfg, G1_PHYSICS_MATERIAL
from engineai_rl_lab.tasks.navigation.g1.unitree_velocity_policy import get_unitree_velocity_policy_path

BASE_CASES = {
    "stand": (0.0, 0.0, 0.0),
    "turn_left": (0.0, 0.0, 1.0),
    "turn_right": (0.0, 0.0, -1.0),
    "forward": (0.4, 0.0, 0.0),
    "backward": (-0.25, 0.0, 0.0),
    "strafe_left": (0.0, 0.3, 0.0),
    "strafe_right": (0.0, -0.3, 0.0),
    "curve_left": (0.3, 0.0, 0.6),
    "curve_right": (0.3, 0.0, -0.6),
}
HEADINGS = (0, 90, 180, 270)
CASES = {f"{name}_yaw{yaw}": command for yaw in HEADINGS for name, command in BASE_CASES.items()}


@configclass
class FlatScene(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(
        physics_material=G1_PHYSICS_MATERIAL))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class EmptyObservations:
    pass


@configclass
class MaterialEvents(DefaultEventManagerCfg):
    # Check the actual startup randomizer cannot overwrite the calibrated feet.
    physics_material = G1EventCfg().physics_material


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = ManagerBasedEnvCfg(
        scene=FlatScene(num_envs=len(CASES), env_spacing=12.0),
        observations=EmptyObservations(), actions=G1ActionsCfg(), decimation=40,
        events=MaterialEvents(),
        sim=sim_utils.SimulationCfg(dt=0.005, render_interval=40, device=args.device,
                                   physics_material=G1_PHYSICS_MATERIAL), seed=0,
    )
    env = ManagerBasedEnv(cfg)
    try:
        env.reset()
        robot = env.scene["robot"]
        adapter = env.action_manager.get_term("velocity_command")
        state = robot.data.root_state_w.clone()
        for i in range(len(CASES)):
            yaw = math.radians(HEADINGS[i // len(BASE_CASES)])
            state[i, 3:7] = torch.tensor([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)], device=env.device)
        robot.write_root_pose_to_sim(state[:, :7])
        env.sim.forward()
        total_mass = robot.root_physx_view.get_masses().sum(dim=1)
        assert torch.allclose(total_mass, torch.full_like(total_mass, 35.11514), atol=1e-3, rtol=0), "Unexpected G1 runtime mass"
        materials = robot.root_physx_view.get_material_properties()
        offset = 0
        foot_shapes = 0
        for name, link_path in zip(robot.body_names, robot.root_physx_view.link_paths[0]):
            count = robot._physics_sim_view.create_rigid_body_view(link_path).max_shapes
            if name in ("left_ankle_roll_link", "right_ankle_roll_link"):
                actual = materials[:, offset:offset + count, :2]
                assert torch.allclose(actual, torch.full_like(actual, G1_FOOT_FRICTION), atol=1e-6, rtol=0), "Foot calibration was overwritten"
                foot_shapes += count
            offset += count
        assert foot_shapes == 8 and offset == materials.shape[1]
        assert adapter.cfg.command_min == [-0.5, -0.3, -1.0]
        assert adapter.cfg.command_max == [1.0, 0.3, 1.0]
        assert adapter.cfg.enable_low_pass_filter
        assert torch.allclose(robot.data.default_joint_armature, torch.full_like(
            robot.data.default_joint_armature, 0.01), atol=1e-7, rtol=0)
        assert torch.all(adapter.low_pass_alpha_values[:, 2] == 0)
        target = torch.tensor(list(CASES.values()), device=env.device)
        limits = torch.where(target >= 0, adapter._command_max, adapter._command_min.abs())
        raw = torch.atanh((target / limits).clamp(-1 + 1e-6, 1 - 1e-6))
        initial_position = robot.data.root_pos_w.clone()
        fallen = torch.zeros(len(CASES), dtype=torch.bool, device=env.device)
        max_tilt = torch.zeros(len(CASES), device=env.device)
        first_fall = [None] * len(CASES)
        velocities = []
        finite = True
        with (args.output / "telemetry.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "case", "phase", "cmd_vx", "cmd_vy", "cmd_wz",
                             "actual_vx", "actual_vy", "actual_wz", "height", "tilt_deg"])
            for step in range(round(args.duration / env.step_dt)):
                time = step * env.step_dt
                active = 2.0 <= time < args.duration - 5.0
                with torch.inference_mode():
                    env.step(raw if active else torch.zeros_like(raw))
                cmd = adapter.processed_actions
                expected_yaw = target[:, 2] if active else torch.zeros_like(target[:, 2])
                assert torch.allclose(cmd[:, 2], expected_yaw, atol=2e-6, rtol=0), "Unexpected yaw smoothing/limit"
                velocity = torch.cat((robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]), dim=1)
                tilt = torch.rad2deg(torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1, 1)))
                finite &= bool(torch.isfinite(robot.data.root_state_w).all() and
                               torch.isfinite(adapter.low_level_actions).all())
                now_fallen = (robot.data.root_pos_w[:, 2] < .45) | (tilt > 45)
                for i in (now_fallen & ~fallen).nonzero().flatten().tolist():
                    first_fall[i] = time + env.step_dt
                fallen |= now_fallen
                max_tilt = torch.maximum(max_tilt, tilt)
                if args.duration - 10.0 <= time < args.duration - 5.0:
                    velocities.append(velocity.clone())
                rows = torch.cat((cmd, velocity, robot.data.root_pos_w[:, 2:3], tilt[:, None]), dim=1).tolist()
                for name, row in zip(CASES, rows):
                    writer.writerow([round(time + env.step_dt, 3), name,
                                     "active" if active else "zero", *row])
        mean = torch.stack(velocities).mean(0).cpu().tolist()
        displacement = torch.linalg.vector_norm((robot.data.root_pos_w - initial_position)[:, :2], dim=1).tolist()
        checks = []
        for i, command in enumerate(CASES.values()):
            if command == (0., 0., 0.):
                check = max(abs(v) for v in mean[i]) < .1 and displacement[i] < .3
            else:
                check = all(abs(c) < 1e-8 or v * c > abs(c) * (.1 if j == 2 else .03)
                            for j, (v, c) in enumerate(zip(mean[i], command)))
            checks.append(check)
        results = []
        for i, (name, command) in enumerate(CASES.items()):
            result = dict(case=name, requested_velocity=command, mean_active_velocity=mean[i],
                          fallen=bool(fallen[i]), first_fall_s=first_fall[i], max_tilt_deg=max_tilt[i].item(),
                          final_displacement_m=displacement[i], directional_response=checks[i],
                          passed=finite and not bool(fallen[i]) and checks[i])
            results.append(result)
            print("[control-smoke] " + json.dumps(result), flush=True)
        policy_path = get_unitree_velocity_policy_path()
        report = dict(passed=all(r["passed"] for r in results), duration_s=args.duration,
                      policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                      finite=finite, physics_hz=1.0 / env.physics_dt,
                      low_level_hz=1.0 / (env.physics_dt * adapter.cfg.low_level_decimation),
                      command_hz=1.0 / env.step_dt,
                      actuators={name: type(actuator).__name__ for name, actuator in robot.actuators.items()},
                      command_min=adapter.cfg.command_min, command_max=adapter.cfg.command_max,
                      low_pass_alpha=adapter.cfg.low_pass_filter_alpha, armature=0.01,
                      total_mass_kg=total_mass[0].item(), foot_friction=G1_FOOT_FRICTION,
                      material_randomization=True, headings_deg=HEADINGS, cases=results)
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if args._status_path is not None:
            args._status_path.write_text(json.dumps(report))
        return report["passed"]
    finally:
        env.close()


if __name__ == "__main__":
    try:
        passed = main()
    finally:
        app.close()
    raise SystemExit(0 if passed else 1)
