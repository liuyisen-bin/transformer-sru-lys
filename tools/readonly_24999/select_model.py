#!/usr/bin/env python3
"""Select a validated navigation ONNX without changing the low-level controller."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / 'deploy/transformer_sru_real/contract.json'
INPUTS = {'obs': [1, 1984], 'h_in': [1, 1, 512], 'c_in': [1, 1, 512]}
OUTPUTS = {'actions': [1, 3], 'h_out': [1, 1, 512], 'c_out': [1, 1, 512]}


def select_model(model, contract_path=DEFAULT_CONTRACT):
    model = Path(model).resolve(strict=True)
    relative_model = model.relative_to(ROOT).as_posix()
    contract_path = Path(contract_path).resolve(strict=True)
    contract = json.loads(contract_path.read_text())
    report = json.loads(model.with_suffix('.export.json').read_text())
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if report.get('passed') is not True or report.get('consecutive_test_steps', 0) < 64:
        raise ValueError('A passing 64-step export report is required')
    if report.get('onnx_sha256') != digest:
        raise ValueError('ONNX hash does not match its export report')
    errors = report.get('max_abs_errors_actions_h_c', [])
    native_error = report.get('max_original_actor_vs_wrapper_error', float('inf'))
    if (len(errors) != 3 or any(not math.isfinite(v) or not 0 <= v < 2e-4 for v in errors)
            or not math.isfinite(native_error) or not 0 <= native_error < 2e-5):
        raise ValueError('Export numerical comparison did not pass')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    runtime = ort.InferenceSession(str(model), sess_options=options,
                                   providers=['CPUExecutionProvider'])
    for tensors, expected in ((runtime.get_inputs(), INPUTS), (runtime.get_outputs(), OUTPUTS)):
        if ({v.name: v.shape for v in tensors} != expected
                or any(v.type != 'tensor(float)' for v in tensors)):
            raise ValueError('ONNX names, shapes or dtypes do not match the deployment interface')
    # Preserve the released observation preprocessing and action semantics.
    baseline = json.loads((ROOT / 'checkpoints/model_24999.export.json').read_text())
    metadata = runtime.get_modelmeta().custom_metadata_map
    for key in ('observation_layout', 'lidar_shape', 'state_contract', 'action_contract', 'normalization'):
        if metadata.get(key) != baseline[key] or report.get(key) != baseline[key]:
            raise ValueError(f'Incompatible model metadata: {key}')
    outputs = runtime.run(list(OUTPUTS), {k: np.zeros(v, np.float32) for k, v in INPUTS.items()})
    if any(not np.isfinite(v).all() for v in outputs):
        raise ValueError('ONNX inference produced non-finite output')
    contract.update(high_model=relative_model, high_sha256=digest)
    # A failed validation never changes the active contract.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=contract_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(contract, indent=2) + '\n')
        temporary.chmod(contract_path.stat().st_mode & 0o777)
        os.replace(temporary, contract_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'contract': str(contract_path), 'high_model': relative_model, 'high_sha256': digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', type=Path, required=True, help='Model inside this repository, with its .export.json')
    parser.add_argument('--contract', type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(json.dumps(select_model(args.onnx, args.contract), indent=2))


if __name__ == '__main__':
    main()
