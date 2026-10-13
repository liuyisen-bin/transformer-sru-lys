"""Model selection must preserve controller settings and reject invalid exports."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import onnx

from select_model import ROOT, DEFAULT_CONTRACT, select_model


class SelectModelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.contract = self.directory / 'contract.json'
        shutil.copyfile(DEFAULT_CONTRACT, self.contract)
        self.before = self.contract.read_bytes()
        self.model = self.directory / 'new_model.onnx'
        shutil.copyfile(ROOT / 'checkpoints/model_24999.onnx', self.model)
        self.report_path = self.model.with_suffix('.export.json')
        self.report = json.loads((ROOT / 'checkpoints/model_24999.export.json').read_text())
        self.write_report()

    def write_report(self):
        self.report_path.write_text(json.dumps(self.report))

    def rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            select_model(self.model, self.contract)
        self.assertEqual(self.contract.read_bytes(), self.before)

    def test_select_and_restore_preserve_controller(self):
        select_model(self.model, self.contract)
        updated = json.loads(self.contract.read_text())
        expected = json.loads(self.before)
        expected['high_model'] = self.model.relative_to(ROOT).as_posix()
        expected['high_sha256'] = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.assertEqual(updated, expected)
        select_model(ROOT / 'checkpoints/model_24999.onnx', self.contract)
        self.assertEqual(json.loads(self.contract.read_text()), json.loads(self.before))

    def test_report_missing(self):
        self.report_path.unlink()
        self.rejected()

    def test_model_hash_mismatch(self):
        self.report['onnx_sha256'] = '0' * 64
        self.write_report()
        self.rejected()

    def test_failed_or_incomplete_export(self):
        for changes in ({'passed': False}, {'consecutive_test_steps': 1},
                        {'max_abs_errors_actions_h_c': [0, float('nan'), 0]},
                        {'max_original_actor_vs_wrapper_error': 1.0}):
            with self.subTest(changes=changes):
                original = self.report.copy()
                self.report.update(changes)
                self.write_report()
                self.rejected()
                self.report = original

    def test_wrong_interface_with_matching_hash(self):
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node('Identity', ['obs'], ['actions'])], 'wrong_interface',
            [onnx.helper.make_tensor_value_info('obs', onnx.TensorProto.FLOAT, [1, 1984])],
            [onnx.helper.make_tensor_value_info('actions', onnx.TensorProto.FLOAT, [1, 1984])])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=8)
        onnx.save(model, self.model)
        self.report['onnx_sha256'] = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.write_report()
        self.rejected()

    def test_changed_preprocessing_with_matching_hash(self):
        model = onnx.load(self.model)
        metadata = {p.key: p.value for p in model.metadata_props}
        metadata['normalization'] = 'different preprocessing'
        onnx.helper.set_model_props(model, metadata)
        onnx.save(model, self.model)
        self.report.update(normalization=metadata['normalization'],
                           onnx_sha256=hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.write_report()
        self.rejected()


if __name__ == '__main__':
    unittest.main()
