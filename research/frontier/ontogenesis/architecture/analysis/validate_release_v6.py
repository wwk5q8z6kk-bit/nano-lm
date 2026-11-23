"""Bounded V6 release gate. No model training or external access.

Default is read-only. --out creates a new receipt exclusively. Historical V5
scientific code/results are checked for preservation, not rerun wholesale.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
DOC_CHANGES={
 'README.md','01_ARCHITECTURE_AND_THEOREMS.md','02_DECISION_LEDGER.md',
 '03_EXPERIMENTS_AND_EXECUTION.md','04_EVIDENCE_AND_REFERENCES.md',
 '05_GAP_CLOSURE_AND_FORMAL_REVIEW.md','06_VALIDATION_AND_UNRESOLVED_WORK.md',
 '07_IMPLEMENTATION_CONTRACT.md','08_LITERATURE_AUDIT.md',
 '10_TRAINING_AND_CONTROLLED_UPDATE_CONTRACT.md',
 '13_CAUSAL_MEMORY_AND_RESOURCE_FRONTIER.md','14_NATIVE_UPDATE_COMPARISON.md',
}

def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,msg):
 if not ok:raise AssertionError(msg)
def read(path):return json.loads((ROOT/path).read_text())
def strict_json(raw):
 def pairs(items):
  result={}
  for k,v in items:
   require(k not in result,'duplicate JSON key '+k);result[k]=v
  return result
 def bad(value):raise ValueError('nonfinite JSON '+value)
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def command(args):
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
 process=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,env=env,
   capture_output=True,text=True,timeout=60)
 require(process.returncode==0,'command failed: '+repr(args)+'\n'+process.stderr[-7000:])
 return process
def unit_count(process,n):
 counts=re.findall(r'^Ran (\d+) tests? in [\d.]+s$',process.stderr,re.M)
 require(counts==[str(n)] and process.stderr.rstrip().endswith('OK'),'unexpected unittest receipt')
 require('skipped=' not in process.stderr,'skips not accepted')
 return {'status':'PASS','tests':n,'stdout_sha256':sha(process.stdout.encode()),
         'stderr_sha256':sha(process.stderr.encode())}

def preservation():
 baseline=read('evidence/v5_preservation_baseline.json')['files']
 require(len(baseline)==208,'V5 source inventory changed')
 old_docs=ROOT/'evidence/history_v6/v5_documents'
 same=[];extended=[];results=[]
 for row in baseline:
  path=row['path'];current=(ROOT/path).read_bytes()
  if path=='README.md':
   old=(ROOT/'evidence/history_v6/README_v5.md.txt').read_bytes()
   require(sha(old)==row['sha256'],'historical README changed')
  elif path=='ledger_events.jsonl':
   old=(ROOT/'evidence/history_v6/ledger_events_v5.jsonl.txt').read_bytes()
   require(sha(old)==row['sha256'] and current.startswith(old),'V5 event prefix changed')
   extended.append(path)
  elif path in DOC_CHANGES:
   old=(old_docs/path).read_bytes()
   require(sha(old)==row['sha256'] and old in current,'historical chapter not preserved: '+path)
   extended.append(path)
  else:
   require(len(current)==row['bytes'] and sha(current)==row['sha256'],'changed V5 source: '+path)
   same.append(path)
  if path.startswith('results/'):
   require(sha(current)==row['sha256'],'historical result changed');results.append(path)
 require(len(results)==27,'V5 result count changed')
 first=(ROOT/'evidence/history_v6/first_principles_LEDGER_before_v6.md.txt').read_bytes()
 shared=(ROOT/'evidence/history_v6/shared_LEDGER_before_v6.md.txt').read_bytes()
 require((ROOT.parent/'LEDGER-FIRST-PRINCIPLES.md').read_bytes().startswith(first),'first-principles prefix lost')
 require((ROOT.parent/'LEDGER.md').read_bytes()==shared,'shared ledger changed')
 return {'status':'PASS','prior_files':len(baseline),'byte_identical_files':len(same),
         'prior_result_files':len(results),'extended_historical_files':extended,
         'first_principles_prefix_bytes':len(first),'shared_ledger_bytes':len(shared)}

def sources():
 m=read('evidence/source_manifest_v6.json');seen=set()
 for row in m['files']:
  raw=(ROOT/row['local_path']).read_bytes()
  require(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'source content mismatch')
  require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==row['git_blob_sha'],'Git blob mismatch')
  seen.add((row['source_commit'],row['upstream_path']))
 require(len(m['files'])==m['source_copy_count']==231,'source copy count')
 require(len(seen)==m['distinct_commit_path_identities']==217,'distinct source count')
 return {'status':'PASS','copies':len(m['files']),'distinct_commit_path_identities':len(seen),'bytes':sum(x['bytes'] for x in m['files'])}

def ledger():
 import jsonschema
 lines=(ROOT/'ledger_events.jsonl').read_bytes().splitlines()
 records=[strict_json(x) for x in lines]
 require(len(records)==58,'event count')
 ids=set();latest={};validator=jsonschema.Draft202012Validator(read('ledger_event.schema.json'),format_checker=jsonschema.FormatChecker())
 for i,e in enumerate(records):
  validator.validate(e)
  require(e['event_id'] not in ids,'duplicate event');ids.add(e['event_id'])
  if e.get('schema_version')==2:
   require(e['previous_event_id']==latest.get(e['decision_id']),'previous event linkage')
  latest[e['decision_id']]=e['event_id']
  if i>=46:
   for a in e['provenance']['evidence_artifacts']:
    require(sha((ROOT/a['path']).read_bytes())==a['sha256'],'event artifact hash '+a['path'])
   for p in e['source_refs']:
    require((ROOT/p).exists(),'event source path missing '+p)
 human=set(re.findall(r'\bD-\d{3}\b',(ROOT/'02_DECISION_LEDGER.md').read_text()))
 require(len(human)==73 and len(latest)==50,'decision counts')
 require(set(latest)<=human,'machine event missing from human ledger')
 return {'status':'PASS','events':58,'human_ids':73,'machine_ids':50,'prior_prefix_events':46,'new_events':12}

def literature():
 old=read('results/literature_audit_v5.json')['records'];new=read('results/literature_audit_v6.json')['records']
 require(new[:len(old)]==old and len(old)==43 and len(new)==48,'literature history/count')
 require(len({x['id'] for x in new})==48,'duplicate literature ID')
 return {'status':'PASS','identities':48,'preserved_prior_records':43,'new_primary_records':5}

def reanalysis():
 result={}
 for kind,script,saved in [
  ('native','analysis/reconcile_upstream_v6.py','results/upstream_reanalysis_v6.json'),
  ('correction','analysis/reconcile_correction_v6.py','results/correction_audit_v6.json'),
  ('context','analysis/reconcile_context_v6.py','results/context_source_diagnostics_v6_original.json')]:
  current=strict_json(command([script]).stdout);old=read(saved)
  if kind=='context':
   # Packaged runner has relocated paths/stdout-only behavior; its hash is new.
   require(current['diagnostic_source_sha256']==sha((ROOT/script).read_bytes()),'current context runner hash')
   current.pop('diagnostic_source_sha256');old.pop('diagnostic_source_sha256')
  require(current==old,kind+' saved reanalysis disagreement')
  result[kind]={'status':'PASS','saved_result_sha256':sha((ROOT/saved).read_bytes())}
 result['status']='PASS'
 return result

def generation():
 process=command(['-m','unittest','discover','-s','experiments','-p','test_generative_reasoning_v6.py','-v'])
 unit=unit_count(process,31)
 sys.path.insert(0,str(ROOT/'experiments'))
 import generative_reasoning_v6 as g
 dataset=g.load_dataset(ROOT/'evidence/generated_relations_v6')
 require(dataset.manifest_sha256=='8cffe7ed4e5cd308cf675974cc30382462c82085720645502a0107c270b0706c','published dataset identity')
 report=g.instrument_controls(dataset);saved=read('results/generative_reasoning_checks_v6_final.json')
 require(saved['generator_and_scorer_source_sha256']==g.source_sha(),'final scorer hash')
 require(saved['test_source_sha256']==sha((ROOT/'experiments/test_generative_reasoning_v6.py').read_bytes()),'generation test hash')
 for key in report['controls']:
  require(report['controls'][key]==saved['controls'][key],key+' controls differ')
 require(saved['unit_validation']['tests_run']==31 and saved['unit_validation']['successful'],'saved generation tests')
 old=read('results/generative_reasoning_checks_v6.json')
 require(old['generator_and_scorer_source_sha256']==sha((ROOT/'evidence/history_v6/initial_instrument/generative_reasoning_v6.py').read_bytes()),'initial source custody')
 require(old['dataset_manifest_sha256']==saved['dataset_manifest_sha256'],'dataset changed across repair')
 require(saved['model_execution_status']=='NOT_RUN' and saved['contamination_status']=='NOT_CHECKED','overclaimed model/contamination status')
 return {'status':'PASS','unit_tests':unit['tests'],'worlds':96,'records':480,'dataset_sha256':dataset.manifest_sha256,
         'model_execution':'NOT_RUN','contamination':'NOT_CHECKED','all_saved_controls_identical':True}

def independent_generation():
 current=strict_json(command(['checks/review_generative_reasoning_v6.py']).stdout)
 old=read('results/generation_independent_corrected_adversarial_results.json')
 require(current==old,'independent adversarial saved output mismatch')
 require(current['check_count']==21 and current['failure_count']==0,'independent checks')
 return {'status':'PASS','checks':21,'source_sha256':current['source_sha256']}

def temporal():
 process=command(['experiments/test_temporal_revision_scoring_v6.py'])
 unit_count(process,20)
 current=strict_json(process.stdout);old=read('results/temporal_revision_checks_v6.json')
 require(current==old,'temporal saved result mismatch')
 require(current['independent_exhaustive_plans']==1350,'temporal exhaustive plans')
 return {'status':'PASS','tests':20,'independent_tiny_history_plans':1350,'source_sha256':current['source_sha256']}

def architecture_math():
 # Independent integer inventory from named tensor shapes, not the compact formula alone.
 d,f,h,kv,hd,L,V=1280,3456,20,4,64,20,32768
 shapes=[(d,h*hd),(d,kv*hd),(d,kv*hd),(h*hd,d),(d,2*f),(f,d),(d,),(d,),(hd,),(hd,)]
 import math
 total=V*d+L*sum(math.prod(s) for s in shapes)+d
 require(total==386062080,'native parameter inventory')
 cache=2*L*1*kv*512*hd*2
 require(cache==10485760,'GQA cache tensor budget')
 require(234950*64*512==7698841600,'training exposure')
 return {'status':'PASS','native_parameters':total,'hypothetical_KV_bytes':cache,'training_target_presentations':7698841600,
         'scope':'Integer accounting only; proof review recorded separately. No checkpoint load or serving-memory measurement.'}

def document_links():
 names=['README.md','00_FULL_ARCHITECTURE.md','15_NATIVE_BASELINE_AND_RESEARCH_UPDATE.md',
        '16_GENERATIVE_REASONING_EVALUATION.md','17_CORRECTION_MEASUREMENT_AND_PROTECTION.md']
 count=0
 for name in names:
  text=(ROOT/name).read_text()
  require(text.count('```')%2==0,'unclosed code fence '+name)
  for target in re.findall(r'\]\(([^)\s]+)\)',text):
   if '://' in target or target.startswith('#'):continue
   rel=target.split('#',1)[0]
   if rel=='results/validation_v6_final.json':continue  # this gate creates the receipt afterward
   require((ROOT/rel).exists(),'broken local link '+name+': '+rel);count+=1
 return {'status':'PASS','current_documents':len(names),'local_links':count}

def inventory():
 return {str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in ROOT.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts}

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--out',type=Path)
 args=parser.parse_args()
 if args.out is not None:require(not args.out.exists(),'receipt destination exists')
 before=inventory();start=time.monotonic();groups={}
 for name,fn in [('v5_preservation',preservation),('frozen_sources',sources),('ledger',ledger),
                 ('literature',literature),('source_reanalysis',reanalysis),('generation',generation),
                 ('independent_generation',independent_generation),('temporal',temporal),
                 ('architecture_math',architecture_math),('current_document_links',document_links)]:
  groups[name]=fn()
 require(before==inventory(),'default validation changed package bytes')
 report={'release':'Nano LM V6','classification':'BOUNDED_SOURCE_AND_INSTRUMENT_VALIDATION_NOT_MODEL_TRAINING',
         'status':'PASS','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'runtime':{'python':platform.python_version(),'platform':platform.platform()},
         'group_count':len(groups),'groups':groups,'seconds':round(time.monotonic()-start,3),
         'read_only_default_verified':True,'model_execution':'NOT_RUN','architecture_adoption':'NOT_DECIDED',
         'validator_sha256':sha(Path(__file__).read_bytes()),
         'limitations':['No complete historical V5 rerun. Unchanged sources/results are verified by exact preservation.',
                        'No trained checkpoint, actual tokenizer, corpus, model scoring, paid compute or remote CI execution.',
                        'Consistency checks and hashes are not independent backend attestation or proof of scientific truth.']}
 if args.out:
  with args.out.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
