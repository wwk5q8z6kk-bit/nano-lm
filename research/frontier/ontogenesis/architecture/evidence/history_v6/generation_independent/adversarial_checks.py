"""Independent synthetic adversarial checks. No model, tokenizer, or corpus run.

The forged receipt fixtures intentionally demonstrate that declarations do not
attest to model execution. Their output is labelled synthetic throughout.
"""
from __future__ import annotations
import copy, hashlib, importlib, json, pathlib, re, sys, tempfile
from collections import Counter, defaultdict

HERE=pathlib.Path(__file__).resolve().parent
snapshot=pathlib.Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE/'initial_snapshot'
out=pathlib.Path(sys.argv[2]).resolve() if len(sys.argv)>2 else HERE/'initial_adversarial_results.json'
sys.path.insert(0,str(snapshot))
g=importlib.import_module('generative_reasoning_v6')
checks=[]
def add(name,passed,details=None):
    checks.append({'name':name,'passed':bool(passed),'details':details})
def rejected(f):
    try:f()
    except Exception as e:return {'rejected':True,'exception':type(e).__name__,'message':str(e)}
    return {'rejected':False}
def plain_oracle(prompt):
    values=dict(re.findall(r'(entity_[0-9a-f]{12}) has value (value_[0-9a-f]{12})\.',prompt))
    pointers=dict(re.findall(r'(entity_[0-9a-f]{12}) points to (entity_[0-9a-f]{12})\.',prompt))
    q=prompt.split('\n\nQuery: ',1)[1].split('\nAnswer:',1)[0]
    entity=re.search(r'entity_[0-9a-f]{12}',q)[0]
    if 'entity reached from' in q:entity=pointers.get(entity)
    return values.get(entity,'UNKNOWN')

with tempfile.TemporaryDirectory(prefix='nanolm-independent-d-') as temp:
    d=g.generate_dataset(pathlib.Path(temp)/'data')
    rm=g.make_run_manifest(d,'independent_synthetic_controls_not_model')
    pb={x['id']:x for x in d.public};ob={x['id']:x for x in d.oracle}
    preds=g.control_predictions(d,lambda view:plain_oracle(view.prompt))
    perfect=g.score(d,preds,rm)
    groups=defaultdict(dict)
    for x in d.oracle:groups[x['world_id']][x['condition']]=x
    expected={('development','all'):32,('development','direct_retrieval'):16,('development','two_hop'):16,
              ('evaluation','all'):64,('evaluation','direct_retrieval'):32,('evaluation','two_hop'):32}
    denominators={}
    for (sp,fam),n in expected.items():
        m=perfect['metrics'][sp][fam]
        assert m['worlds']==n and m['records']==5*n
        assert m['primary_counterfactual_joint']=={'correct':n,'denominator':n,'rate':1.0}
        assert m['unknown_abstention']['denominator']==2*n
        assert all(x['denominator']==n for x in m['condition_policy_accuracy'].values())
        denominators[sp+'/'+fam]={'worlds':n,'records':5*n,'abstention_denominator':2*n}
    add('all_split_family_world_denominators',True,denominators)
    add('independent_visible_prompt_oracle_all480',all(plain_oracle(pb[x['id']]['prompt'])==x['expected_answer'] for x in d.oracle))
    for world,rows in groups.items():
        a,b,perm,nc,broken=[rows[k] for k in g.CONDITIONS]
        assert a['query_entity']==b['query_entity'] and a['expected_answer']!=b['expected_answer']
        assert sum(x!=y for x,y in zip(a['facts'],b['facts']))==1
        assert sorted(map(g.fact_tuple,a['facts']))==sorted(map(g.fact_tuple,perm['facts'])) and a['facts']!=perm['facts']
        assert nc['facts']==[] and nc['expected_answer']=='UNKNOWN' and broken['expected_answer']=='UNKNOWN'
    add('all96_worlds_counterfactual_permutation_absence_semantics',True)
    subset=[p for p in preds if ob[p['id']]['condition']!='full_b']
    miss=g.score(d,subset,rm)
    add('all_full_b_missing_keeps_primary_denominators',all(miss['metrics'][sp][fam]['primary_counterfactual_joint']['correct']==0 and miss['metrics'][sp][fam]['primary_counterfactual_joint']['denominator']==n for (sp,fam),n in expected.items()),miss['denominators'])
    leak=copy.deepcopy(preds)
    for p in leak:
        if ob[p['id']]['condition']=='no_context':p['raw_text']=ob[p['id']]['a_target']
    gain=g.score(d,leak,rm)
    add('original_target_gain_distinct_from_unknown_policy',all(gain['metrics'][sp][fam]['original_target_contextual_gain']['paired_difference']==0 and gain['metrics'][sp][fam]['condition_policy_accuracy']['no_context']['rate']==0 and perfect['metrics'][sp][fam]['original_target_contextual_gain']['paired_difference']==1 for sp,fam in expected))
    same=copy.deepcopy(preds)
    for p in same:
        if ob[p['id']]['condition']=='full_b':p['raw_text']=ob[p['id']]['a_target']
    st=g.score(d,same,rm)
    add('same_A_answer_in_B_fails_every_pair',all(st['metrics'][sp][fam]['primary_counterfactual_joint']['correct']==0 for sp,fam in expected))
    for name,mutation in [('duplicate_id',lambda ps:ps+[ps[0]]),('unknown_id',lambda ps:[{**ps[0],'id':'f'*24}]),('prompt_hash',lambda ps:[{**ps[0],'receipt':{**ps[0]['receipt'],'prompt_sha256':'0'*64}}])]:
        x=rejected(lambda:g.score(d,mutation(copy.deepcopy(preds)),rm));add(name,x['rejected'],x)
    bad=copy.deepcopy(d.oracle);bad[0]['expected_answer']='value_000000000000'
    x=rejected(lambda:g.validate_dataset(d.public,bad));add('gold_only_tamper',x['rejected'],x)
    publicfile=pathlib.Path(temp)/'data/public.jsonl';publicfile.write_bytes(publicfile.read_bytes()+b'\n')
    x=rejected(lambda:g.load_dataset(pathlib.Path(temp)/'data'));add('on_disk_public_hash_tamper',x['rejected'],x)

    # Genuine consistency gap: coherently mutate one world IN MEMORY, keeping
    # the original manifest and its hash unchanged. No all-file/hash forgery.
    altered=copy.deepcopy(d);ap={x['id']:x for x in altered.public}
    world=next(x['world_id'] for x in altered.oracle if x['family']=='direct_retrieval')
    old=next(x['expected_answer'] for x in altered.oracle if x['world_id']==world and x['condition']=='full_a')
    new='value_deadbeefcafe'
    for row in altered.oracle:
        if row['world_id']!=world:continue
        row['a_target']=new
        for f in row['facts']:
            if f['object']==old:f['object']=new
        prompt=g.render_prompt(row['facts'],row['family'],row['query_entity']);ap[row['id']]['prompt']=prompt
        row['expected_answer']=plain_oracle(prompt);row['prompt_sha256']=g.text_sha(prompt)
        row['prompt_utf8_bytes']=len(prompt.encode());row['context_utf8_bytes']=len(g.context_text(row['facts']).encode())
    semantic=rejected(lambda:g.validate_dataset(altered.public,altered.oracle))
    result_box={}
    def run_altered():
        result_box['r']=g.score(altered,g.control_predictions(altered,lambda v:plain_oracle(v.prompt)),rm)
    x=rejected(run_altered)
    add('in_memory_dataset_mutation_rejected_before_scoring',x['rejected'],{
        **x,'semantics_still_valid':not semantic['rejected'],
        'claimed_public_file_hash':d.manifest['files']['public.jsonl']['sha256'],
        'actual_altered_public_hash':g.sha(g.jsonl(altered.public)),
        'reported_dataset_hash_unchanged':bool(result_box) and result_box['r']['dataset_manifest_sha256']==d.manifest_sha256})

    identity={'checkpoint_sha256':'0'*64,'tokenizer_sha256':'1'*64,'model_config_sha256':'2'*64,'adapter_source_sha256':'3'*64,'context_limit':100}
    modelrun=g.make_run_manifest(d,'FORGED_SYNTHETIC_RECEIPT_NOT_MODEL_EXECUTION','model_checkpoint_evaluation',identity)
    row=next(x for x in d.oracle if x['condition']=='full_a')
    def fake_prediction(raw,stop='eos',ids=None,output=None):
        ids=[7,8,9] if ids is None else ids;output=[5,6] if output is None else output
        return {'id':row['id'],'status':'generated','raw_text':raw,'receipt':g.model_receipt(pb[row['id']]['prompt'],modelrun,ids,output,stop)}
    newline_cases=[]
    for raw in ('\n'+row['expected_answer'],row['expected_answer']+'\n\n','\n'+row['expected_answer']+'\n'):
        box={}
        def run_bad_stop():box['r']=g.score(d,[fake_prediction(raw)],modelrun)
        x=rejected(run_bad_stop)
        newline_cases.append({'raw_repr':repr(raw),**x,'credited_correct':bool(box) and box['r']['metrics'][row['split']][row['family']]['condition_policy_accuracy']['full_a']['correct']})
    add('eos_cannot_bypass_first_newline_contract',all(x['rejected'] for x in newline_cases),newline_cases)
    x=rejected(lambda:g.score(d,[fake_prediction(row['expected_answer'],ids=list(range(69)))],modelrun))
    add('model_generated_oversize_receipt_rejected',x['rejected'],x)
    box=g.score(d,[fake_prediction(row['expected_answer'],stop='max_new_tokens',output=list(range(32)))],modelrun)
    add('budget_exact_text_is_not_success',box['metrics'][row['split']][row['family']]['condition_policy_accuracy']['full_a']['correct']==0,box['denominators'])
    x=rejected(lambda:g.score(d,preds,modelrun));add('unmodified_control_receipt_relabel_rejected',x['rejected'],x)
    forged=[]
    for x in d.public:
        forged.append({'id':x['id'],'status':'generated','raw_text':plain_oracle(x['prompt']),
                       'receipt':g.model_receipt(x['prompt'],modelrun,[1,2],[3],'eos')})
    fr=g.score(d,forged,modelrun)
    add('receipt_self_assertion_boundary_demonstration',True,{
        'fixture_only_no_model_run':True,'accepted_synthetic_model_shaped_receipts':True,
        'reported_execution_status':fr['model_execution_status'],
        'evaluation_primary':fr['metrics']['evaluation']['all']['primary_counterfactual_joint'],
        'expected_scope':'Receipt consistency is not independent model/tokenizer/decode attestation.'})

    class AliasAdapter:
        identity=identity
        def reset(self):self.ids=[7,8,9]
        def encode_prompt(self,v):self.preflight_copy=list(self.ids);return self.ids
        def generate(self,v):
            self.ids.pop()
            return g.Completion(plain_oracle(v.prompt),self.ids,[4,5],'eos')
    ad=AliasAdapter();small=g.Dataset([pb[row['id']]],[],d.manifest,d.manifest_sha256)
    alias=g.run_candidate(small,ad,modelrun)
    add('encoder_list_alias_cannot_hide_prompt_id_change',alias[0]['status']=='error',{
        'status':alias[0]['status'],'preflight_ids':ad.preflight_copy,'recorded_ids':alias[0]['receipt']['input_token_ids']})
    class ReusedOutputBufferAdapter:
        identity=identity
        def __init__(self):self.buffer=[4,5];self.count=0;self.returned=[]
        def reset(self):pass
        def encode_prompt(self,v):return [7,8,9]
        def generate(self,v):
            self.count+=1;self.buffer[:]=[4,4+self.count];self.returned.append(list(self.buffer))
            return g.Completion(plain_oracle(v.prompt),[7,8,9],self.buffer,'eos')
    ad=ReusedOutputBufferAdapter();two=g.Dataset(d.public[:2],[],d.manifest,d.manifest_sha256)
    buffer_preds=g.run_candidate(two,ad,modelrun)
    recorded=[p['receipt']['output_token_ids'] for p in buffer_preds]
    add('prior_output_receipts_not_changed_by_adapter_buffer_reuse',recorded==ad.returned,{
        'output_ids_at_return':ad.returned,'output_ids_in_final_receipts':recorded})
    class BoundaryAdapter:
        identity=identity
        calls=0
        def __init__(self,n):self.n=n
        def reset(self):pass
        def encode_prompt(self,v):return list(range(self.n))
        def generate(self,v):self.calls+=1;return g.Completion(plain_oracle(v.prompt),list(range(self.n)),[5,6],'eos')
    boundaries=[]
    for n in (68,69):
        ad=BoundaryAdapter(n);p=g.run_candidate(small,ad,modelrun)[0]
        boundaries.append({'input_tokens':n,'max_new_tokens':32,'context_limit':100,'status':p['status'],'generate_calls':ad.calls})
    add('inclusive_context_boundary_and_no_generation_above_it',boundaries[0]['status']=='generated' and boundaries[1]['status']=='context_ineligible' and boundaries[1]['generate_calls']==0,boundaries)
    parsers={repr(x):g.normalized_answer(x) for x in ('UNKNOWN',' unknown',' UNKNOWN\r\n','UNKNOWN\u00a0','UNKNOWN\x0b','value_0123456789ab.','value_0123456789ab','value_0123456789ab\nUNKNOWN')}
    add('strict_parser_non_ASCII_and_extra_text',parsers[repr('UNKNOWN\u00a0')] is None and parsers[repr('UNKNOWN\x0b')] is None and parsers[repr(' UNKNOWN\r\n')]=='UNKNOWN',parsers)
    byte_lengths={}
    for family in g.FAMILIES:
        xs=[x['prompt_utf8_bytes'] for x in d.oracle if x['family']==family]
        byte_lengths[family]={'min':min(xs),'max':max(xs),'token_eligibility':'NOT_CHECKED_NO_ACTUAL_TOKENIZER'}

result={'classification':'INDEPENDENT_SYNTHETIC_INSTRUMENT_REVIEW_NOT_MODEL_RESULTS',
        'source_sha256':g.source_sha(),'protocol_sha256':g.FROZEN_PROTOCOL_SHA256,
        'checks':checks,'check_count':len(checks),'failure_count':sum(not x['passed'] for x in checks),
        'prompt_byte_lengths_not_token_counts':byte_lengths}
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(out),'source_sha256':g.source_sha(),'checks':len(checks),'failures':[x for x in checks if not x['passed']]},indent=2))
