"""Deterministic read-only arithmetic audit. No model import, fitting, or inference.

Reads immutable source copies and writes derived review JSON beside this script.
Run from repository workspace root: python v6_review/upstream_evidence/reanalyse.py
"""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path

ARCH = Path(__file__).resolve().parents[1]
HERE = ARCH / "evidence/upstream_v6/native"
P = HERE / 'sources/research/frontier/ontogenesis'
load = lambda f: json.loads((P / f).read_text())
M = json.loads((HERE / 'manifest.json').read_text())
for f in M['files']:
    raw = (ARCH / f['local_path']).read_bytes()
    assert len(raw) == f['expected_bytes']
    assert hashlib.sha256(raw).hexdigest() == f['sha256']
    assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == f['git_blob_sha']

XXL = 'dense-XXL-lr0.002-s1000-tok7.7e+09-muon-dp4-fl0.01-hs-v2'
XL = 'dense-XL-lr0.001-s1000-tok4.2e+09-dp4-hs-v2'
r = load('results/scale2/' + XXL + '.json')
b = load('results/scale2/' + XXL + '.bench.json')['tasks']
nc = load('results/scale2/' + XXL + '.bench-nocontext.json')['tasks']
old = load('results/scale1/' + XL + '.json')
ob = load('results/scale1/' + XL + '.bench.json')['tasks']
v = load('results/scale2/q13-verdict.json')
s135 = load('results/lm/eval-reference-SmolLM2-135M.json')['tasks']
s360 = load('results/lm/eval-reference-SmolLM2-360M.json')['tasks']
s135nc = load('results/lm/eval-reference-SmolLM2-135M-nocontext.json')['tasks']
s360nc = load('results/lm/eval-reference-SmolLM2-360M-nocontext.json')['tasks']
py3 = load('results/lm/eval-pythia410m-step3000.json')['tasks']
py4 = load('results/lm/eval-pythia410m-step4000.json')['tasks']
assert r['finished'] and r['steps'] == r['curve'][-1]['step']
assert r['steps'] * r['bs'] * r['seq'] == r['train_tokens'] == r['curve'][-1]['tokens']
assert r['code_sha256'] == hashlib.sha256((HERE/'historical_sources/12316003862ff4117c26583a65ea808e1ff1e2a8/research/frontier/ontogenesis/tpu/train_jax.py').read_bytes()).hexdigest()[:16]
q2 = .79 <= r['curve'][-1]['mix'] <= .83
q3a = all(b[k]['acc'] > ob[k]['acc'] for k in ('arc_easy', 'piqa'))
q3b = abs(b['arc_easy']['acc']-.521) <= .03 and abs(b['piqa']['acc']-.668) <= .03 and b['lambada']['ppl'] > 10.8
assert [v[k]['held'] for k in ('Q2','Q3a','Q3b')] == [q2,q3a,q3b]
mix = sum(w*r['curve'][-1][k] for k,w in {'web':.62,'textbook':.13,'math':.13,'code':.12}.items())
assert abs(mix-r['curve'][-1]['mix']) <= .0001
lr_table = []
for lr in ('.002','.004','.008'):
    x=load(f'results/scale2/dense-L-lr0{lr}-s1000-tok8e+08-muon-dp4-fl0.01-hs-v2.json')
    lr_table.append({'lr':x['lr'],'mix':x['curve'][-1]['mix'],'finished':x['finished'],'seed':x['seed']})
chosen = min(lr_table,key=lambda x:x['mix'])['lr']
assert float((P/'results/scale2/chosen-lr.txt').read_text()) == min(chosen,.004) == r['lr']
benchmark=[]
for k,x in b.items():
    n=x['n']; assert all(d[k]['n']==n for d in (ob,nc,s135,s360,s135nc,s360nc,py3,py4))
    a,c=x['acc'],s135[k]['acc']
    # Descriptive independent-binomial approximation only. Same items require paired outputs.
    se=math.sqrt((a*(1-a)+c*(1-c))/n)
    benchmark.append({'task':k,'n':n,'ours211_acc':ob[k]['acc'],'ours386_acc':a,
                      'ours386_acc_norm':x.get('acc_norm'),'smol135_acc':c,'smol360_acc':s360[k]['acc'],
                      'ours386_minus_smol135':round(a-c,4),
                      'ours386_context_gain':round(a-nc[k]['acc'],4) if k!='winogrande' else None,
                      'smol135_context_gain':round(c-s135nc[k]['acc'],4) if k!='winogrande' else None,
                      'smol360_context_gain':round(s360[k]['acc']-s360nc[k]['acc'],4) if k!='winogrande' else None,
                      'nocontext_ours_minus_smol135':round(nc[k]['acc']-s135nc[k]['acc'],4),
                      'approx_unpaired_se_gap':round(se,6),'approx_unpaired_z':round((a-c)/se,3),
                      'pythia_step3000_acc':py3[k]['acc'],'pythia_step4000_acc':py4[k]['acc']})

arms=('looped','dense','tf-4x2-d64','tf-8x1-d64'); seeds=range(21,26)
counts={}; details=[]
for arm in arms:
    counts[arm]={}
    for label,lr in [('1e-3',.001),('3e-4',.0003),('3e-3',.003)]:
        n=0
        for s in seeds:
            file=f'results/nested/{arm}-s{s}.json' if label=='1e-3' else f'results/nested-lr/{arm}-lr{label}-s{s}.json'
            d=load(file)
            assert (d['arm'],d['lr'],d['seed'],d['steps']) == (arm,lr,s,26000)
            assert [x[0] for x in d['curve']]==list(range(1000,26001,1000))
            passed=[x[0] for x in d['curve'] if x[2]['h2']>=.9]
            n+=bool(passed)
            details.append({'arm':arm,'lr':lr,'seed':s,'params':d['params'],'first90':min(passed) if passed else None,
                            'final_curve_h2':d['curve'][-1][2]['h2'],'separate_final_eval_h2':d['eval']['h2'],
                            'final_h1':d['eval']['h1'],'seconds':d['train_seconds']})
        counts[arm][label]=n
best={a:max(c.values()) for a,c in counts.items()}
pairs=[(best['looped'],best['dense']),(best['tf-4x2-d64'],best['tf-8x1-d64'])]
decision='reuse advantage stands' if all(a>=b+2 for a,b in pairs) else 'explained by learning rate' if all(abs(a-b)<=1 for a,b in pairs) else 'mixed'
nv=load('results/nested-lr/verdict.json');assert nv['counts']==counts and nv['decision']==decision

ft={a:load('results/forward_transfer/'+a+'.json') for a in ('fresh','dense','dense_replay','loop','loop_replay','menu','menu_replay')}
fv=load('results/forward_transfer/verdict.json')
primary={}
for a,rows in ft.items():
    assert [x['seed'] for x in rows] == [9101,9102,9103]
    if a=='fresh':continue
    xs=[x['by_experience']['24']['novel_comp/embed/k8'] for x in rows]
    mean=round(sum(xs)/len(xs),4)
    assert fv['table_primary'][a]==[mean,xs]
    primary[a]={'mean':mean,'per_seed':xs,'query_predictions_per_seed':8*(23-8),
                'shared_model_params':rows[0]['params'],'additional_stream_task_params':24*16,
                'stored_examples':rows[0]['stored_examples'],
                'stream_retention':round(sum(x['by_experience']['24']['stream_retained'] for x in rows)/3,4)}
embed_max={a:max(round(sum(x['by_experience'][str(n)][f'novel_prim/embed/k{k}'] for x in rows)/3,4) for n in (0,2,4,8,16,24) for k in (2,4,8,16)) for a,rows in ft.items() if a!='fresh'}
full_gain={a:round(max(round(sum(x['by_experience']['24'][f'novel_prim/full/k{k}'] for x in rows)/3,4)-round(sum(x['by_experience']['0'][f'novel_prim/full/k{k}'] for x in ft['fresh'])/3,4) for k in (2,4,8,16)),4) for a,rows in ft.items() if a!='fresh'}
assert embed_max==fv['predictions']['T7']['novel_prim_embed_max']
assert full_gain==fv['predictions']['T7']['novel_prim_full_minus_fresh_max']
fcheck={
 'T1': all(all(row['by_experience']['0'][f'{kind}/full/k{k}'] <= .1 for kind in ('novel_comp','novel_prim') for k in (2,4,8,16)) for row in ft['fresh']),
 'T2':primary['dense']['mean']<=.15,
 'T3':.15<=primary['dense_replay']['mean']<=.50,
 'T4':primary['loop_replay']['mean']>=.6 and primary['loop_replay']['mean']-primary['dense_replay']['mean']>=.2,
 'T5':primary['menu_replay']['mean']>=.9 and sum(x['by_experience']['24']['novel_comp/embed/k2'] for x in ft['menu_replay'])/3>=.6 and primary['menu']['mean']>=.7,
 'T6':all(primary[a]['mean']-round(sum(x['by_experience']['4']['novel_comp/embed/k8'] for x in ft[a])/3,4)>=.2 for a in ('loop_replay','menu_replay')),
 'T7':all(x<=.15 for x in embed_max.values()) and all(x<=.05 for x in full_gain.values())}
assert all(fv['predictions'][k]['held']==val for k,val in fcheck.items())
qa=load('results/q14/dense-L-lr0.002-s1000-tok8e+08-muon-dp4-fl0.01-hs-v2.json')
qc=load('results/q14/dense-L128-lr0.002-s1000-tok8e+08-muon-dp4-fl0.01-splash-hs-v2.json')
speed=qc['curve'][-1]['tokens_per_second']/qa['curve'][-1]['tokens_per_second']
assert not qc['finished'] and speed<1.25

ldo={s:load(f'results/lm/store-ldo-{s}.json') for s in ('S','M','L')}
cache={s:load(f'results/lm/cache-exchange-{s}.json') for s in ('S','M')}
for x in ldo.values():
    assert x['store_tokens']==48000000 and x['store_share_of_training_file']==.06
    for dom,row in x['bits_per_byte'].items():
        assert round(row['model']-row['ldo'],4)==x['gain_leave_document_out'][dom]
        assert round(row['model']-row['uninformed_real_weights'],4)==x['failing_control_gain'][dom]
    assert all(v<0 for v in x['failing_control_gain'].values()) and x['failing_control_passed']
for x in cache.values():
    for dom,row in x['bits_per_byte'].items():assert round(row['model']-row['with_cache'],4)==x['gain'][dom]
    assert x['causality_check_passed']
store_gate=ldo['L']['gain_leave_document_out']['mix']<.01
cache_gate=all(cache['M']['gain'][dom]<.01 for dom in ('web','textbook','math','code'))
assert store_gate and cache_gate

words=[f'w{i}' for i in range(30)]
stride=lambda w:{tuple(w[i:i+13]) for i in range(0,max(0,len(w)-12),7)}
allgrams=lambda w:{tuple(w[i:i+13]) for i in range(max(0,len(w)-12))}
shift=['prefix']+words
assert allgrams(words)&allgrams(shift) and not stride(words)&stride(shift)

out={'cutoff':M['cutoff'],'verified_sources':len(M['files']),'verified_bytes':sum(x['expected_bytes'] for x in M['files']),
 'q13':{'params':r['params'],'train_tokens':r['train_tokens'],'steps':r['steps'],
        'tokens_per_parameter':r['train_tokens']/r['params'],'approx_6ND_flops':6*r['params']*r['train_tokens'],
        'smol135_token_exposure_ratio':2e12/r['train_tokens'],'smol135_approx_6ND_flops_ratio':6*135e6*2e12/(6*r['params']*r['train_tokens']),
        'estimated_corpus_token_coverage_fraction':1-math.exp(-r['train_tokens']/r['corpus_tokens']),
        'finished':r['finished'],'mix':r['curve'][-1]['mix'],'weighted_rounded_domain_mix':mix,
        'mix_improvement_211_to_386':old['curve'][-1]['mix']-r['curve'][-1]['mix'],
        'lr_table':lr_table,'Q2':q2,'Q3a':q3a,'Q3b':q3b,
        'benchmark_total_n':sum(x['n'] for x in b.values()),'benchmark':benchmark,
        'wall_hours_to_final_eval':r['curve'][-1]['hours'],'kernel_tokens_per_second':r['curve'][-1]['tokens_per_second'],
        'wall_tokens_per_second_to_final_eval':r['train_tokens']/(r['curve'][-1]['hours']*3600)},
 'nested_lr':{'counts':counts,'best':best,'decision':decision,'detail':details},
 'forward_transfer':{'primary':primary,'predicates':fcheck,'chance_reference':1/23,
                     'replay_training_examples_processed':300*23*sum(range(1,25)),
                     'non_replay_training_examples_processed':300*23*24},
 'q14':{'speed_ratio':speed,'W1':False,'W2':'unmeasured','W3':'unmeasured','planned_tokens_metadata':qc['train_tokens'],
        'completed_tokens_last_curve':qc['curve'][-1]['tokens'],'completed_steps':qc['curve'][-1]['step']},
 'store_cache':{'ldo_mix_gains':{s:x['gain_leave_document_out']['mix'] for s,x in ldo.items()},
                'ldo_101M_domain_gains':ldo['L']['gain_leave_document_out'],
                'ldo_101M_control_gains':ldo['L']['failing_control_gain'],
                'ldo_scoped_closure':store_gate,'cache_scoped_closure_40M_512context':cache_gate,
                'cache_40M_domain_gains':cache['M']['gain'],
                'caveat':'One checkpoint per size, first-version sampler; LDO is post-hoc complement fitting, not joint-training repair.'},
 'stride7_overlap_counterexample':{'full_overlap_windows':len(allgrams(words)&allgrams(shift)),'sampled_overlap_windows':len(stride(words)&stride(shift)),
                                 'construction':'30 unique words versus same 30 words with one word prepended'},
 'limitations':['Arithmetic/source audit only; no training or inference rerun.',
                'Rounded aggregate scores cannot support exact paired significance tests.',
                'Approximate 6ND compute and corpus coverage are calculations, not measured FLOPs or actual unique tokens.',
                'WinoGrande no-context result is a degenerate tie baseline, excluded from context-gain interpretation.']}
print(json.dumps(out, indent=2, allow_nan=False))
