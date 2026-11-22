"""Behavioral regression checks for the research artifact contract.

Runs in temporary package copies. Corruption fixtures are never real ledger
events. No training, network call, or mutation of the source package is needed.
Run: python3 -B checks/test_research_package.py
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
import validate_research_package as validator

ROOT = Path(__file__).resolve().parents[1]


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class ResearchPackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = tempfile.TemporaryDirectory(prefix='nanolm-contract-baseline-')
        cls.source = Path(cls.snapshot.name) / 'package'
        shutil.copytree(ROOT, cls.source, ignore=shutil.ignore_patterns('__pycache__'))

    @classmethod
    def tearDownClass(cls):
        cls.snapshot.cleanup()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='nanolm-contract-test-')
        self.root = Path(self.temporary.name) / 'package'
        shutil.copytree(self.source, self.root)
        self.log = self.root / 'ledger_events.jsonl'

    def tearDown(self):
        self.temporary.cleanup()

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def event(self, decision='D-900', suffix='first', previous=None):
        return {'schema_version': 2, 'decision_id': decision,
                'event_id': f'evt-fixture-{decision}-{suffix}',
                'previous_event_id': previous, 'recorded_at_utc': '2030-01-01T00:00:00Z',
                'status': 'reviewed', 'decision': 'Synthetic contract-test decision',
                'rationale': 'A temporary fixture exercises the event contract.',
                'alternatives_considered': ['Preserve the existing event history'],
                'evidence_type': 'engineering', 'source_refs': ['README.md'],
                'next_gate': 'Temporary test only', 'supersedes_ids': [],
                'affected_model': 'No model; temporary validation fixture',
                'permission_class': 'review_only', 'outcome': 'document_only',
                'open_questions': [],
                'provenance': {'source_commit': '3f9ef1a1b5643acda9ee4cc2a780d4fdcbf4bce8',
                               'actor': 'validation fixture',
                               'evidence_note': 'Temporary test input, never a scientific result.'}}

    def append(self, event):
        with self.log.open('a') as stream:
            stream.write(json.dumps(event) + '\n')

    def human(self, decision):
        with (self.root / '02_DECISION_LEDGER.md').open('a') as stream:
            stream.write(f'\n| {decision} | Temporary validation fixture. |\n')

    def with_event(self):
        event = self.event()
        self.human(event['decision_id'])
        return event

    def assert_passes(self):
        report = validator.validate(self.root)
        self.assertEqual(report['status'], 'PASS', report['errors'])
        return report

    def assert_fails(self, text):
        report = validator.validate(self.root)
        self.assertEqual(report['status'], 'FAIL', report)
        self.assertTrue(any(text in message for message in report['errors']), report['errors'])
        return report

    def test_current_package_passes(self):
        self.assert_passes()

    def test_new_decision_and_later_revision_are_accepted(self):
        first = self.with_event()
        self.append(first)
        revision = self.event(suffix='revision', previous=first['event_id'])
        revision['rationale'] = 'A later review extends the same decision without deleting its past.'
        self.append(revision)
        report = self.assert_passes()
        self.assertEqual(report['latest_events']['D-900'], revision['event_id'])

    def test_revision_of_legacy_decision_is_accepted(self):
        last = next(e for e in reversed(self.events()) if e['decision_id'] == 'D-026')
        self.append(self.event('D-026', previous=last['event_id']))
        self.assert_passes()

    def test_duplicate_event_id_is_rejected(self):
        first = self.with_event()
        self.append(first)
        self.append(copy.deepcopy(first))
        self.assert_fails('duplicate event_id')

    def test_orphan_revision_is_rejected(self):
        event = self.with_event()
        event['previous_event_id'] = 'evt-does-not-exist'
        self.append(event)
        self.assert_fails('latest prior event')

    def test_stale_revision_pointer_is_rejected(self):
        first = self.with_event()
        self.append(first)
        second = self.event(suffix='second', previous=first['event_id'])
        self.append(second)
        self.append(self.event(suffix='third', previous=first['event_id']))
        self.assert_fails('latest prior event')

    def test_new_legacy_shape_cannot_bypass_provenance(self):
        event = self.with_event()
        for key in ['schema_version', 'previous_event_id', 'provenance']:
            del event[key]
        self.append(event)
        self.assert_fails('new events require schema_version 2')

    def test_wrong_enum_is_rejected(self):
        event = self.with_event()
        event['status'] = 'proved superior'
        self.append(event)
        self.assert_fails('schema')

    def test_wrong_field_type_is_rejected_without_crashing(self):
        event = self.with_event()
        event['source_refs'] = 'README.md'
        self.append(event)
        self.assert_fails('schema')

    def test_non_object_event_is_rejected_without_crashing(self):
        self.append(['not an event object'])
        self.assert_fails('schema')

    def test_unknown_field_is_rejected(self):
        event = self.with_event()
        event['unsupported_promise'] = True
        self.append(event)
        self.assert_fails('Additional properties')

    def test_new_event_requires_provenance(self):
        event = self.with_event()
        del event['provenance']
        self.append(event)
        self.assert_fails('provenance')

    def test_wrong_source_commit_shape_is_rejected(self):
        event = self.with_event()
        event['provenance']['source_commit'] = 'main'
        self.append(event)
        self.assert_fails('source_commit')

    def test_missing_source_file_is_rejected(self):
        event = self.with_event()
        event['source_refs'] = ['unavailable-evidence.md']
        self.append(event)
        self.assert_fails('unresolved file')

    def test_malformed_remote_source_is_rejected(self):
        event = self.with_event()
        event['source_refs'] = ['https://[broken']
        self.append(event)
        self.assert_fails('malformed reference')

    def test_source_cannot_escape_package(self):
        event = self.with_event()
        event['source_refs'] = ['../outside.md']
        self.append(event)
        self.assert_fails('escapes the portable package')

    def test_missing_required_document_reports_failure(self):
        (self.root / '02_DECISION_LEDGER.md').unlink()
        self.assert_fails('missing 02_DECISION_LEDGER.md')

    def test_new_documents_are_checked_automatically(self):
        (self.root / '99_FIXTURE.md').write_text('[Bad target](missing-file.md)\n')
        self.assert_fails('99_FIXTURE.md: unresolved file')

    def test_bad_local_heading_is_rejected(self):
        with (self.root / 'README.md').open('a') as stream:
            stream.write('\n[Missing heading](02_DECISION_LEDGER.md#no-such-heading)\n')
        self.assert_fails('unresolved heading')

    def test_legacy_prefix_edit_is_detected(self):
        rows = self.events()
        rows[0]['rationale'] = 'A changed history must not silently be accepted.'
        self.log.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assert_fails('pinned historical bytes')

    def test_complete_prior_ledger_is_preserved_on_append(self):
        baseline = Path(self.temporary.name) / 'prior.jsonl'
        baseline.write_bytes(self.log.read_bytes())
        first = self.with_event()
        self.append(first)
        report = validator.validate(self.root, baseline)
        self.assertEqual(report['status'], 'PASS', report['errors'])
        self.assertEqual(report['full_baseline_append_only'], 'PASS')

    def test_rewriting_post_legacy_history_fails_full_baseline_check(self):
        first = self.with_event()
        self.append(first)
        baseline = Path(self.temporary.name) / 'prior.jsonl'
        baseline.write_bytes(self.log.read_bytes())
        lines = self.log.read_bytes().splitlines(keepends=True)
        changed = json.loads(lines[-1])
        changed['rationale'] = 'This rewrite is locally coherent but violates the prior committed record.'
        self.log.write_bytes(b''.join(lines[:-1]) + (json.dumps(changed) + '\n').encode())
        # Fixed legacy-prefix protection alone cannot detect this later edit.
        self.assert_passes()
        report = validator.validate(self.root, baseline)
        self.assertEqual(report['status'], 'FAIL', report)
        self.assertEqual(report['full_baseline_append_only'], 'FAIL')
        self.assertTrue(any('byte-exact extension' in error for error in report['errors']))

    def test_malformed_json_line_reports_failure(self):
        with self.log.open('a') as stream:
            stream.write('{not JSON}\n')
        self.assert_fails('invalid JSON')

    def test_duplicate_json_key_is_rejected(self):
        event = self.with_event()
        serialized = json.dumps(event)
        serialized = serialized[:-1] + ', "status": "confirmed"}'
        with self.log.open('a') as stream:
            stream.write(serialized + '\n')
        self.assert_fails('duplicate JSON object key')

    def test_timestamp_requires_utc_and_valid_calendar(self):
        for timestamp in ['2030-01-01T00:00:00+01:00', '2030-02-31T00:00:00Z']:
            with self.subTest(timestamp=timestamp):
                before = self.log.read_bytes()
                event = self.event()
                event['recorded_at_utc'] = timestamp
                self.append(event)
                self.assert_fails('schema')
                self.log.write_bytes(before)

    def test_recorded_time_cannot_go_backwards(self):
        event = self.with_event()
        event['recorded_at_utc'] = '1990-01-01T00:00:00Z'
        self.append(event)
        self.assert_fails('earlier than the preceding append')

    def test_hypothesis_cannot_be_labeled_confirmed(self):
        event = self.with_event()
        event.update(status='confirmed', outcome='confirmed', evidence_type='hypothesis')
        self.append(event)
        self.assert_fails('schema')

    def test_confirmed_claim_requires_artifact(self):
        event = self.with_event()
        event.update(status='confirmed', outcome='confirmed')
        self.append(event)
        self.assert_fails('evidence_artifacts')

    def test_bad_evidence_checksum_is_rejected(self):
        event = self.with_event()
        event['provenance']['evidence_artifacts'] = [{'path': 'README.md', 'sha256': '0' * 64}]
        self.append(event)
        self.assert_fails('SHA-256 mismatch')

    def test_valid_evidence_checksum_is_accepted(self):
        previous_count = self.assert_passes()['local_evidence_hashes_checked']
        artifact = self.root / 'temporary-evidence.json'
        artifact.write_text('{"synthetic_fixture": true}\n')
        event = self.with_event()
        event.update(status='confirmed', outcome='confirmed')
        event['provenance']['evidence_artifacts'] = [{'path': artifact.name, 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}]
        self.append(event)
        report = self.assert_passes()
        self.assertEqual(report['local_evidence_hashes_checked'], previous_count + 1)

    def test_unknown_superseded_decision_is_rejected(self):
        event = self.with_event()
        event['supersedes_ids'] = ['D-999']
        self.append(event)
        self.assert_fails('different earlier decision')

    def test_duplicate_human_row_is_rejected(self):
        self.human('D-026')
        self.assert_fails('duplicate human decision rows')

    def test_default_scripts_leave_every_package_byte_unchanged(self):
        before = hashes(self.root)
        for name in ['validate_research_package.py', 'repo_lightweight_checks.py',
                     'formal_sanity.py', 'transport_anchor_pilot.py']:
            with self.subTest(script=name):
                process = subprocess.run([sys.executable, '-B', str(self.root / 'checks' / name)],
                                         text=True, capture_output=True, timeout=45,
                                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertEqual(hashes(self.root), before)

    def test_explicit_output_refuses_existing_files(self):
        output = Path(self.temporary.name) / 'existing.json'
        sentinel = b'Frozen prior result; do not overwrite.\n'
        output.write_bytes(sentinel)
        for name in ['validate_research_package.py', 'formal_sanity.py', 'transport_anchor_pilot.py']:
            with self.subTest(script=name):
                process = subprocess.run([sys.executable, '-B', str(self.root / 'checks' / name), '--out', str(output)],
                                         text=True, capture_output=True, timeout=45)
                self.assertNotEqual(process.returncode, 0)
                self.assertIn('FileExistsError', process.stderr)
                self.assertEqual(output.read_bytes(), sentinel)

    def test_remainder_check_fails_if_function_is_made_linear(self):
        # Independent mutation: remove curvature from f, leaving the checker alone.
        path = self.root / 'checks/repo_lightweight_checks.py'
        tree = ast.parse(path.read_text())
        changed = False
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'check_nonzero_second_order_drift':
                for inner in node.body:
                    if isinstance(inner, ast.FunctionDef) and inner.name == 'f':
                        inner.body = [ast.Return(value=ast.parse('theta[0]', mode='eval').body)]
                        changed = True
        self.assertTrue(changed, 'Mutation did not locate the function under test')
        namespace = {'__name__': 'mutation_fixture'}
        exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
        with self.assertRaises(AssertionError):
            namespace['check_nonzero_second_order_drift']()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResearchPackageTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    print(json.dumps({'suite': 'research_package_contract', 'tests_run': result.testsRun,
                      'failures': len(result.failures), 'errors': len(result.errors),
                      'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                      'scope': 'Temporary corruption fixtures, append-history semantics, read-only behavior and numerical-check mutation.'}, indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
