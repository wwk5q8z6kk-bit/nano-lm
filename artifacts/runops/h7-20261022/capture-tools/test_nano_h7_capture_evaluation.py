import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('capture', '/private/tmp/nano_h7_capture_evaluation.py')
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
from nano_ai.contract import FIELD_ORDER
from nano_ai.training import evaluate_evidence_query_h7 as frozen
from nano_ai.training.evaluate_pointer import _empty_raw_bucket, _finish_raw_bucket

def jb(value):
    return (json.dumps(value, sort_keys=True) + '\n').encode()

def sha(value):
    return hashlib.sha256(value).hexdigest()

def tar_bytes(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode='w:gz') as tar:
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            tar.addfile(member, io.BytesIO(payload))
    return out.getvalue()

def sums(files):
    return ''.join(f'{sha(payload)}  {name}\n' for name, payload in sorted(files.items())).encode()

def synthetic_raw(error_rows):
    rows=[]
    aggregate=_empty_raw_bucket()
    by_field={name.value:_empty_raw_bucket() for name in FIELD_ORDER}
    for index in range(1000):
        fields=[]
        for name in by_field:
            state='supported' if index%2 else 'absent'
            gold=[{'start':1,'end':4,'text':'foo','speaker':'patient'}]
            exact=index>=error_rows
            spans=gold if exact else [{'start':0,'end':4,'text':'xfoo','speaker':'patient'}]
            field={'field':name,'raw_state':state,'gold_state':state,'state_correct':True,
                   'span_exact':exact,'joint_exact':exact,'presented':True,'wrong_presented':not exact,
                   'raw_spans':spans,'gold_spans':gold}
            fields.append(field)
            for bucket in (aggregate,by_field[name]):
                bucket['total']+=1;bucket['decoded']+=1
                for key in ('state_correct','span_exact','joint_exact','presented','wrong_presented'):
                    bucket[key]+=int(field[key])
        rows.append({'example_id':f'invented-{index}','variant':'base','target_field':'chief_complaint',
                     'target_state':None,'decode_status':'decoded','decode_error':None,'fields':fields})
    return {'items':1000,'decoded_items':1000,'decode_failure_items':0,'decode_failure_rate':0.,
            'fields':_finish_raw_bucket(aggregate),'by_field':{name:_finish_raw_bucket(bucket) for name,bucket in by_field.items()},
            'wrong_presented_field_count':aggregate['wrong_presented'],'item_diagnostics':rows}

def fixture(root, mutate=None):
    authority = root / 'authority'
    authority.mkdir()
    training_res = {'attempt_id': 'synthetic-training'}
    (authority / 'training.reservation.json').write_bytes(jb(training_res))
    train_terminal = {'attempt_id': training_res['attempt_id'], 'reservation_sha256': sha(jb(training_res)),
                      'status': 'completed', 'exit_code': 0, 'error': None, 'elapsed_seconds': 100.0}
    candidates = [{'label': f'h7-seed-{seed}', 'sha256': str(i)*64, 'artifact_bytes': 100,
                   'seed': seed, 'epoch': 1, 'report_sha256': str(i+2)*64} for i,seed in enumerate(h.CONTRACT['treatment_seeds'],1)]
    training = {'terminal': train_terminal, 'runtime': h.RUNTIME, 'candidates': candidates,
                'all_attempt_members_verified': True, 'all_treatment_members_verified': True,
                'treatment_archive_sha256': '9'*64}
    training_path = root / 'training-capture.json'
    training_path.write_bytes(jb(training))
    res = {'schema': 'nano.h7-operational-reservation.v1', 'experiment': h.op.EXPERIMENT,
           'stage': 'evaluation', 'attempt_id': 'synthetic-evaluation', 'canonical_authority': str(authority.resolve()),
           'package': h.op.PACKAGES['evaluation'], 'runner_sha256': h.RUNNER_SHA256,
           'kernel_metadata': {'is_private': True, 'enable_gpu': False, 'enable_tpu': False, 'dataset_sources':['synthetic/package','synthetic/treatments']},
           'stage_dataset':'synthetic/package', 'treatment_dataset':'synthetic/treatments',
           'verified_private_datasets':['synthetic/package','synthetic/treatments'], 'treatment_archive_sha256':'9'*64,
           'max_runtime_seconds': 14300, 'campaign_runtime_cap_seconds':14400, 'authorization_cost_usd':0,
           'training_terminal':train_terminal}
    reservation_path = authority / 'evaluation.reservation.json'
    reservation_path.write_bytes(jb(res))
    started = {'attempt_id':res['attempt_id'], 'stage':'evaluation', 'started_at_utc':'2026-10-22T20:00:00Z', 'reservation_sha256':sha(jb(res))}
    terminal = {**started,'finished_at_utc':'2026-10-22T20:03:20Z','elapsed_seconds':200.0,
                'status':'completed','phase':'completed','exit_code':0,'error':None,'retry_authorized':False}
    metrics = {'joint_exact_accuracy':.8,'state_accuracy':.9,'decode_failure_items':0}
    mechanism = {'binding_error_fields':100,'disjoint_or_speaker_mismatch_fields':5,'field_collapse_count':0}
    arms = [{'candidate':{'seed':None,'checkpoint_sha256':h.CONTRACT['control_checkpoint_sha256'],'checkpoint_bytes':13168916},'metrics':metrics,'mechanism':mechanism}]
    comparisons = []
    for prior in candidates:
        candidate = {'label':prior['label'],'seed':prior['seed'],'epoch':prior['epoch'],'checkpoint_sha256':prior['sha256'],'checkpoint_bytes':prior['artifact_bytes'],'training_report_sha256':prior['report_sha256']}
        arms.append({'candidate':candidate,'metrics':metrics.copy(),'mechanism':{**mechanism,'binding_error_fields':60}})
        comparisons.append({'treatment_seed':prior['seed'],'bars':dict.fromkeys(h.comparison.H7_BAR_NAMES,True),
                            'all_pass_bars':True,'safety_or_state_violation':False,'boundary_improvement_bars_met':True,
                            'span_exact_on_shared_state_correct':{'denominator':1000,'control_numerator':800,'control_rate':.8,'treatment_numerator':900,'treatment_rate':.9,'absolute_gain':.1},
                            'binding_error_fields':{'control':100,'treatment':60,'relative_reduction':.4},
                            'state_accuracy_delta':0.,'disjoint_or_speaker_mismatch_delta':0})
    result = {'schema_version':h.CONTRACT['schema_version'],'experiment_id':h.CONTRACT['experiment_id'],
              'status':'complete','decision':'PASS','arms':arms,'comparisons':comparisons,
              'runtime':{'device':'cpu','batch_size':32,'python':'3.13.15','torch':'2.11.0+cpu','cuda':None,'gpu':None},
              'screen':{'manifest_sha256':h.CONTRACT['screen_manifest_sha256'],'records_sha256':h.CONTRACT['screen_records_sha256'],'records':1000},
              'termination':{'control_scoring_passes':1,'treatment_scoring_passes':2,'third_seed_allowed':False,'threshold_sweep_performed':False,'h6_development_opened':False}}
    for index, arm in enumerate(arms):
        raw=synthetic_raw(200 if index==0 else 100)
        arm['raw']=raw
        arm['metrics']={key:raw['fields'][key] for key in ('joint_exact_accuracy','state_accuracy')}
        arm['metrics']['decode_failure_items']=0
        arm['mechanism']=frozen._mechanism_summary(raw)
    result['comparisons']=[frozen._comparison(arms[0],arm) for arm in arms[1:]]
    result['decision']=frozen._decision(result['comparisons'])
    package = h.ROOT / 'artifacts/nano_h7/execution-package-v1/nano-h7-kaggle-evaluation.tar.gz'
    files = {'evidence/started.json':jb(started),'evidence/terminal.json':jb(terminal),'evidence/reservation.json':jb(res),
             'evidence/runtime.json':jb(h.RUNTIME), 'evidence/runner.log':b'SYNTHETIC UNIT FIXTURE\n',
             'evidence/screen_evaluation.started.json':jb({'event':'screen_evaluation','started_at_utc':'2026-10-22T20:00:10Z'}),
             'evidence/package-validation.json':jb({'manifest_sha256':h.op.PACKAGES['evaluation']['manifest_sha256'],'full_tree_verified':True,'frozen_archive_sha256_provenance':h.op.PACKAGES['evaluation']['archive_sha256']}),
             'evidence/RUN_H7_KAGGLE_OPERATIONAL.sh':h.expected_wrapper(package).encode(),
             'evidence/runtime_probe.py':h.op.RUNTIME_PROBE.encode(),'evidence/event_marker.py':h.op.EVENT_MARKER.encode()}
    if mutate:
        mutate(result, files)
    result_files = {'results/h7_screen_evaluation.json':jb(result),'results/h7_screen_evaluation.log':b'SYNTHETIC UNIT FIXTURE\n'}
    result_files['results/SHA256SUMS'] = sums(result_files)
    inner = tar_bytes(result_files)
    inner_name = h.op.PACKAGES['evaluation']['result']
    files[inner_name] = inner
    files[inner_name+'.sha256'] = f'{sha(inner)}  {inner_name}\n'.encode()
    files.update({'nano-h7-evaluation/'+name:payload for name,payload in result_files.items()})
    files['ATTEMPT_SHA256SUMS'] = sums(files)
    archive = root / 'attempt.tar.gz'
    archive.write_bytes(tar_bytes(files))
    archive.with_name(archive.name+'.sha256').write_text(f'{h.op.digest(archive)}  {archive.name}\n')
    return SimpleNamespace(archive=archive,destination=root/'capture',training_capture=training_path,reservation=reservation_path,authority=authority,evaluation_package=package)

class CaptureTests(unittest.TestCase):
    def test_complete_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            args = fixture(Path(temp))
            with patch.object(frozen, '_load_model', side_effect=AssertionError('model loading forbidden')), \
                 patch.object(frozen, '_load_screen', side_effect=AssertionError('screen access forbidden')), \
                 patch.object(frozen.torch, 'load', side_effect=AssertionError('checkpoint loading forbidden')):
                summary = h.validate(args)
            self.assertEqual(summary['stored_result_validation']['decision'],'PASS')
            self.assertEqual(summary['total_training_and_evaluation_runtime_seconds'],300)
            self.assertEqual([arm['decoded_fields'] for arm in summary['recorded_row_validation']['arms']], [5000]*3)
    def test_failed_terminal_preserves_failure_and_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            def mutate(result,files):
                terminal = json.loads(files['evidence/terminal.json'])
                terminal['status'] = 'failed'
                files['evidence/terminal.json'] = jb(terminal)
            args = fixture(Path(temp),mutate)
            cli = [str(args.archive),str(args.destination),'--training-capture',str(args.training_capture),
                   '--reservation',str(args.reservation),'--authority',str(args.authority),'--evaluation-package',str(args.evaluation_package)]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(h.main(cli),1)
            self.assertTrue((args.destination/'CAPTURE_FAILURE.json').is_file())
            self.assertTrue((args.destination/'evidence/runner.log').is_file())
    def test_candidate_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            def mutate(result,files): result['arms'][1]['candidate']['checkpoint_sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'admitted training'):
                h.validate(fixture(Path(temp),mutate))
    def test_changed_operational_shell(self):
        with tempfile.TemporaryDirectory() as temp:
            def mutate(result,files): files['evidence/RUN_H7_KAGGLE_OPERATIONAL.sh'] += b'echo extra\n'
            with self.assertRaisesRegex(ValueError,'frozen derivation'):
                h.validate(fixture(Path(temp),mutate))
    def test_missing_event(self):
        with tempfile.TemporaryDirectory() as temp:
            def mutate(result,files): del files['evidence/screen_evaluation.started.json']
            with self.assertRaisesRegex(ValueError,'inventory'):
                h.validate(fixture(Path(temp),mutate))
    def test_sidecar_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            args.archive.with_name(args.archive.name+'.sha256').write_text('0'*64+'  attempt.tar.gz\n')
            with self.assertRaisesRegex(ValueError,'sidecar mismatch'): h.validate(args)
            self.assertFalse(args.destination.exists())
    def test_reservation_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp))
            duplicate = Path(temp)/'different.json'
            res=json.loads(args.reservation.read_bytes()); res['attempt_id']='another'
            duplicate.write_bytes(jb(res));args.reservation=duplicate
            with self.assertRaisesRegex(ValueError,'canonical operator authority'):h.validate(args)
    def test_malformed_archive_members(self):
        for case in ('traversal','duplicate','symlink'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); archive=root/'malformed.tar.gz'
                with tarfile.open(archive,'w:gz') as tar:
                    names = ['../escape'] if case=='traversal' else ['safe','safe'] if case=='duplicate' else ['link']
                    for name in names:
                        member=tarfile.TarInfo(name)
                        if case=='symlink':member.type=tarfile.SYMTYPE;member.linkname='/tmp'
                        tar.addfile(member,io.BytesIO())
                with self.assertRaises(ValueError):h.extract_files(archive,root/'extracted')
                self.assertEqual(list((root/'extracted').iterdir()),[])
    def test_strict_json(self):
        for data in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):h.parse_json(data)
    def test_raw_row_tamper(self):
        mutations = {
            'missing row':lambda result:result['arms'][1]['raw']['item_diagnostics'].pop(),
            'duplicate id':lambda result:result['arms'][1]['raw']['item_diagnostics'][1].update(example_id='invented-0'),
            'state flag':lambda result:result['arms'][1]['raw']['item_diagnostics'][0]['fields'][0].update(state_correct=False,joint_exact=False),
            'field aggregate':lambda result:result['arms'][1]['raw']['fields'].update(state_correct=4999),
            'span flag':lambda result:result['arms'][1]['raw']['item_diagnostics'][0]['fields'][0].update(span_exact=True,joint_exact=True),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                def mutate(result,files):mutation(result)
                with self.assertRaises(ValueError):h.validate(fixture(Path(temp),mutate))
    def test_failure_row_remains_in_denominator_and_valid_reject_is_captured(self):
        with tempfile.TemporaryDirectory() as temp:
            def mutate(result,files):
                arm=result['arms'][1];raw=arm['raw'];row=raw['item_diagnostics'][0]
                aggregate={key:raw['fields'][key] for key in _empty_raw_bucket()}
                by_field={name:{key:bucket[key] for key in _empty_raw_bucket()} for name,bucket in raw['by_field'].items()}
                for field in row['fields']:
                    for bucket in (aggregate,by_field[field['field']]):
                        bucket['decoded']-=1
                        for key in ('state_correct','span_exact','joint_exact','presented','wrong_presented'):
                            bucket[key]-=int(field[key])
                row.update(fields=None,decode_status='failure',decode_error='invented failure')
                raw.update(decoded_items=999,decode_failure_items=1,decode_failure_rate=.001,
                           fields=_finish_raw_bucket(aggregate),by_field={name:_finish_raw_bucket(bucket) for name,bucket in by_field.items()},
                           wrong_presented_field_count=aggregate['wrong_presented'])
                arm['metrics']={key:raw['fields'][key] for key in ('joint_exact_accuracy','state_accuracy')}
                arm['metrics']['decode_failure_items']=1
                arm['mechanism']=frozen._mechanism_summary(raw)
                result['comparisons']=[frozen._comparison(result['arms'][0],arm) for arm in result['arms'][1:]]
                result['decision']=frozen._decision(result['comparisons'])
            result=h.validate(fixture(Path(temp),mutate))
            self.assertEqual(result['stored_result_validation']['decision'],'REJECT')
            self.assertEqual(result['recorded_row_validation']['arms'][1]['decoded_fields'],4995)
            self.assertEqual(result['recorded_row_validation']['arms'][1]['field_denominator'],5000)

if __name__ == '__main__':unittest.main()
