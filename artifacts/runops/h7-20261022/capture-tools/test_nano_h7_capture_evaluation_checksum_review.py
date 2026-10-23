"""Capture-only tests; synthetic saved outputs, no scientific evaluation."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load('capture_review', '/private/tmp/nano_h7_capture_evaluation_checksum_review.py')
fixture = load('capture_original_tests', '/private/tmp/test_nano_h7_capture_evaluation.py')
fixture.h = h
CaptureRegressionTests = fixture.CaptureTests


class ChecksumReviewTests(unittest.TestCase):
    def inventory(self, root, mutate=lambda payload: payload):
        data = {'results/h7_screen_evaluation.json': b'{"synthetic":true}\n',
                'results/h7_screen_evaluation.log': b''}
        for name, value in data.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        prefix = fixture.sums(data)
        payload = prefix + f'{fixture.sha(prefix)}  results/SHA256SUMS.tmp\n'.encode()
        (root / 'results/SHA256SUMS').write_bytes(mutate(payload))
        return h.RESULT_FILES.copy()

    def test_exact_temporary_prefix_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root)
            before = (root / 'results/SHA256SUMS').read_bytes()
            anomaly = h.verify_inventory(root, 'results/SHA256SUMS', files)
            self.assertEqual(len(anomaly), 1)
            self.assertEqual(anomaly[0]['type'], 'frozen_checksum_temporary_self_entry')
            self.assertEqual((root / 'results/SHA256SUMS').read_bytes(), before)

    def test_rejects_non_matching_temporary_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root, lambda b: b.rsplit(b'\n', 2)[0] + b'\n' + b'0'*64 + b'  results/SHA256SUMS.tmp\n')
            with self.assertRaisesRegex(ValueError, 'prefix digest'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_rejects_non_final_temporary_entry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root, lambda b: b.splitlines(keepends=True)[-1] + b''.join(b.splitlines(keepends=True)[:-1]))
            with self.assertRaisesRegex(ValueError, 'must follow'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_rejects_missing_real_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root, lambda b: b''.join(b.splitlines(keepends=True)[1:]))
            with self.assertRaisesRegex(ValueError, 'must follow'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_rejects_real_temporary_member(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root)
            (root / 'results/SHA256SUMS.tmp').write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'Unexpected checksum temporary'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_rejects_unknown_missing_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root, lambda b: b.replace(b'results/SHA256SUMS.tmp', b'results/OTHER.tmp'))
            with self.assertRaisesRegex(ValueError, 'Member digest mismatch'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_rejects_changed_real_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = self.inventory(root)
            (root / 'results/h7_screen_evaluation.json').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Member digest mismatch'):
                h.verify_inventory(root, 'results/SHA256SUMS', files)

    def test_end_to_end_synthetic_temporary_bug(self):
        original_sums = fixture.sums
        def bug_sums(files):
            prefix = original_sums(files)
            if set(files) == h.RESULT_FILES - {'results/SHA256SUMS'}:
                return prefix + f'{fixture.sha(prefix)}  results/SHA256SUMS.tmp\n'.encode()
            return prefix
        with tempfile.TemporaryDirectory() as temp, patch.object(fixture, 'sums', side_effect=bug_sums):
            args = fixture.fixture(Path(temp))
            original_sha = h.op.digest(args.archive)
            with patch.object(fixture.frozen, '_load_model', side_effect=AssertionError('model loading forbidden')), \
                 patch.object(fixture.frozen, '_load_screen', side_effect=AssertionError('screen access forbidden')), \
                 patch.object(fixture.frozen.torch, 'load', side_effect=AssertionError('checkpoint loading forbidden')):
                summary = h.validate(args)
            self.assertEqual(h.op.digest(args.archive), original_sha)
            self.assertEqual(summary['stored_result_validation']['decision'], 'PASS')
            self.assertEqual(len(summary['result_checksum_anomalies']), 1)


if __name__ == '__main__':
    unittest.main()
