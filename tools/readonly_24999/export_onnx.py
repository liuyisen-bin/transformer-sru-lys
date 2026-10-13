#!/usr/bin/env python3
"""Export the frozen navigation actor, including explicit SRU state; no ROS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/sru-navigation-learning'))
from rsl_rl.modules import ActorCriticSRURangeImage
from rsl_rl.modules.actor_critic_sru_pointnet import _PointNetSRUONNXExporter


def load_actor(checkpoint):
    config = yaml.load((checkpoint.parent / (checkpoint.stem + '_config') / 'agent.yaml').read_text(), Loader=yaml.FullLoader)
    if config['empirical_normalization']:
        raise ValueError('This exporter requires the recorded no-normalizer contract')
    kwargs = dict(config['policy'])
    assert kwargs.pop('class_name') == 'ActorCriticSRURangeImage'
    actor = ActorCriticSRURangeImage(1984, 5121, 3, **kwargs)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    actor.load_state_dict(saved['model_state_dict'], strict=True)
    actor.eval().requires_grad_(False)
    return actor, saved['iter']


def session(path):
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = 2
    opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opt, providers=['CPUExecutionProvider'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/model_24999.pt')
    p.add_argument('--output', type=Path, default=ROOT / 'checkpoints/model_24999.onnx')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(24999)
    # Avoid exporting PyTorch's fused Transformer fastpath as an unsupported op.
    torch.backends.mha.set_fastpath_enabled(False)
    actor, iteration = load_actor(args.checkpoint)
    wrapper = _PointNetSRUONNXExporter(actor).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    inputs = (torch.zeros(1, 1984), torch.zeros(1, 1, 512), torch.zeros(1, 1, 512))
    torch.onnx.export(wrapper, inputs, str(args.output), dynamo=False,
                      input_names=['obs', 'h_in', 'c_in'],
                      output_names=['actions', 'h_out', 'c_out'],
                      opset_version=17, do_constant_folding=True)
    graph = onnx.load(str(args.output))
    meta = {
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'observation_layout': 'base_lin_vel[3],base_ang_vel[3],projected_gravity[3],previous_raw_action[3],goal_unit_xyz_log1p_distance[4],lidar_xyz_div5[1968]',
        'lidar_shape': '16x41x3, vertical-major',
        'state_contract': 'h,c float32[1,1,512]; reset to zero on new episode; carry outputs each step',
        'action_contract': 'raw navigation action[3]; command mapping/filter are outside this graph',
        'normalization': 'No empirical obs normalization. Original encoder internal divide-by-5 preserved.',
        'intended_use': 'Navigation actor; this file does not contain motor control',
    }
    onnx.helper.set_model_props(graph, meta)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, str(args.output))
    runtime = session(args.output)
    h = np.zeros((1, 1, 512), np.float32); c = h.copy()
    ht = torch.zeros(1, 1, 512); ct = ht.clone()
    actor.memory_a.hidden_states = None
    previous = np.zeros(3, np.float32)
    errors = np.zeros(3); original_error = 0.
    rng = np.random.default_rng(24999)
    el, az = np.meshgrid(np.deg2rad(np.linspace(-45, 0, 16)), np.deg2rad(np.linspace(-60, 60, 41)), indexing='ij')
    rays = np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)], -1).reshape(656, 3)
    with torch.inference_mode():
        for step in range(64):
            points = rays * rng.uniform(.1, 1., (656, 1))
            points[rng.random(656) < .2] = 0
            obs = np.r_[rng.normal(0, .05, 6), [0, 0, -1], previous,
                        [1, 0, 0, np.log(6)], points.ravel()].astype(np.float32)[None]
            out = runtime.run(None, {'obs': obs, 'h_in': h, 'c_in': c})
            ref = wrapper(torch.from_numpy(obs), ht, ct)
            native = actor.act_inference(torch.from_numpy(obs)).numpy()
            for i in range(3):
                assert np.isfinite(out[i]).all()
                errors[i] = max(errors[i], float(np.max(abs(out[i] - ref[i].numpy()))))
            original_error = max(original_error, float(np.max(abs(native - ref[0].numpy()))))
            previous = out[0][0]; h, c = out[1:]; ht, ct = ref[1:]
    assert max(errors) < 2e-4, errors
    assert original_error < 2e-5, original_error
    report = {**meta, 'iteration': int(iteration), 'onnx_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'onnx_bytes': args.output.stat().st_size, 'torch_version': torch.__version__,
              'onnxruntime_version': ort.__version__, 'opset': 17,
              'inputs': {v.name: v.shape for v in runtime.get_inputs()},
              'outputs': {v.name: v.shape for v in runtime.get_outputs()},
              'consecutive_test_steps': 64, 'max_abs_errors_actions_h_c': errors.tolist(),
              'max_original_actor_vs_wrapper_error': original_error, 'passed': True}
    args.output.with_suffix('.export.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
