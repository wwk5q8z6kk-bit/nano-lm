from pathlib import Path, PurePosixPath
import argparse, dataclasses, hashlib, json, re, shutil, sys, tarfile
ROOT=Path('/Users/mac/Projects/nano-lm')
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_h7_kaggle as op
from nano_ai.training.evaluate_evidence_query_h7 import _validate_treatment_report
p=argparse.ArgumentParser()
p.add_argument('archive', type=Path)
p.add_argument('destination', type=Path)
a=p.parse_args()
sha=op.digest(a.archive)
sidecar=a.archive.with_name(a.archive.name+'.sha256').read_text().strip().split()
assert sidecar == [sha,a.archive.name], 'attempt archive hash mismatch'
a.destination.mkdir(parents=True,exist_ok=False)
with tarfile.open(a.archive,'r:gz') as t:
    names=set()
    for m in t.getmembers():
        assert m.isfile() and op.safe_name(m.name) and m.name not in names, 'unsafe/duplicate/non-file capture member'
        names.add(m.name)
        dest=a.destination/m.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        with t.extractfile(m) as source, dest.open('xb') as output: shutil.copyfileobj(source,output)
sums={}
for line in (a.destination/'ATTEMPT_SHA256SUMS').read_text().splitlines():
    match=re.fullmatch(r'([0-9a-f]{64})  (.+)',line)
    assert match and op.safe_name(match[2]) and match[2] not in sums
    assert op.digest(a.destination/match[2])==match[1], 'attempt member digest mismatch'
    sums[match[2]]=match[1]
assert set(sums)==names-{'ATTEMPT_SHA256SUMS'}, 'attempt inventory mismatch'
def read(name): return json.loads((a.destination/'evidence'/name).read_bytes())
reservation=json.loads((ROOT/'artifacts/runops/h7-20261022/training.reservation.json').read_bytes())
assert read('reservation.json')==reservation
terminal=read('terminal.json')
assert terminal['attempt_id']==reservation['attempt_id'] and terminal['stage']=='training'
assert terminal['status']=='completed' and terminal['phase']=='completed' and terminal['exit_code']==0 and terminal['error'] is None
assert 0<terminal['elapsed_seconds']<14400
assert terminal['reservation_sha256']==op.digest(ROOT/'artifacts/runops/h7-20261022/training.reservation.json')
assert read('runtime.json')=={'python':'3.13.15','torch':'2.11.0+cpu','torch_cuda':None,'cuda_available':False,'tokenizers':'0.22.2','pytest':'9.0.2'}
for seed in (20260805,20260806): assert read(f'training_seed_{seed}.started.json')['event']==f'training_seed_{seed}'
result=a.destination/'nano-h7-treatment-results.tar.gz'
result_sha=op.digest(result)
assert result.with_name(result.name+'.sha256').read_text().strip().split()==[result_sha,result.name]
treatments=a.destination/'verified-treatments'
op.extract_treatments(result,result_sha,treatments)
candidates=[]
for seed in (20260805,20260806):
    report=treatments/f'results/seed-{seed}/training_report.json'
    r=json.loads(report.read_bytes())
    assert r['seed']==seed
    candidate=_validate_treatment_report(report,expected_report_sha256=op.digest(report))
    for epoch in r['epochs']:
        checkpoint=report.parent/epoch['checkpoint']['filename']
        assert op.digest(checkpoint)==epoch['checkpoint']['sha256'] and checkpoint.stat().st_size==epoch['checkpoint']['bytes']
    candidates.append({**dataclasses.asdict(candidate),'path':str(candidate.path)})
assert len({candidate['sha256'] for candidate in candidates}) == 2, 'treatment checkpoints must be distinct'
summary={'attempt_archive_sha256':sha,'attempt_archive_bytes':a.archive.stat().st_size,'terminal':terminal,'runtime':read('runtime.json'),'treatment_archive_sha256':result_sha,'treatment_archive_bytes':result.stat().st_size,'candidates':candidates,'all_attempt_members_verified':True,'all_treatment_members_verified':True,'remaining_runtime_seconds':int(14400-terminal['elapsed_seconds'])}
(a.destination/'CAPTURE_VALIDATION.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
print(json.dumps(summary,indent=2,default=str))
