"""Construct invented source dialogues with offsets derived during rendering, before any inference."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json, random, re
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
F=('chief_complaint','duration','severity','medication','allergy')
CC=['burning in my throat','a throbbing headache','left shoulder stiffness','itchy eyes','a persistent dry cough','pressure in my left ear','ankle swelling','a scratchy throat','a tingling sensation in my thumb','right wrist pain','a runny nose','a sore heel','a dull stomach ache','ringing in my ears','a hoarse voice','an itchy patch on my arm','a painful right knee','hand tremors','lower back pain','watery eyes']
DUR=['two days','three weeks','six hours','four days','ten minutes','five mornings','two nights','eight days','one week','forty minutes','twelve hours','three days','a week','six days','four hours']
SEV=['4/10','6/10','three out of ten','seven out of ten','mild','moderate','5/10','2/10','severe','eight out of ten','four out of ten','a six out of ten']
MED=['aspirin','naproxen','ibuprofen','loratadine','warfarin','acetaminophen','cetirizine','levothyroxine','metformin','amoxicillin','albuterol','prednisone','omeprazole','fexofenadine','gabapentin','losartan']
ALG=['nickel','latex','penicillin','peanuts','shellfish','adhesive tape','sulfonamides','amoxicillin','tree nuts','sesame','dust mites','codeine','kiwi','bee stings']
VALS=dict(zip(F,[CC,DUR,SEV,MED,ALG]))
Q={
F[0]:['What is bothering you today?','What brings you in?','Tell me your main symptom.','Which symptom are you here for?'],
F[1]:['How long has your current symptom lasted?','How long have you had this problem?','How long has it been going on?','How long has this symptom bothered you?'],
F[2]:['How bad is it right now?','What is your pain score now?','How severe is your symptom today?','How would you rate this symptom right now?'],
F[3]:['What medication are you taking now?','What have you taken for this symptom?','Which medicine did you actually take?','Have you taken anything for it?'],
F[4]:['Are you allergic to anything?','What allergies do you have?','Do you have any known allergies?','Any allergies we should record?']}
LABEL=dict(zip(F,['symptom','duration','severity','medication','allergy']))
S=['ordinary','bounded_measure','explicit_replacement','historical_only','prospective_only','clinician_proposal','family_experiencer','unidentified_source','logistical_interruption','unrelated_clock','unresolved_alternatives','blanket_denials','withdrawn_without_replacement','mixed_experiencers','current_after_past','qualified_phrase']
G=lambda x:f'«G|{x}»'
def seen(reason,x): return f'«{reason}|{x}»'
def say(f,x,rng):
    return rng.choice({F[0]:['I have '+x+'.','I came in with '+x+'.','The main problem is '+x+'.'],F[1]:['It has lasted '+x+'.','For '+x+'.','It has been '+x+'.'],F[2]:['I would rate it '+x+'.','It is '+x+' right now.','My rating is '+x+'.'],F[3]:['I have taken '+x+'.','I took '+x+'.','I am taking '+x+'.'],F[4]:['I am allergic to '+x+'.','I have an allergy to '+x+'.','My allergy is to '+x+'.']}[f])
def bare(f,v): return v+'.'
def direct(f,v,rng):
    # Capitalize whole bare phrases before inserting the anchor; anchored text equals the source.
    if rng.random()<0.30:
        v=v[0].upper()+v[1:]
        return bare(f,G(v))
    return say(f,G(v),rng)
DEN={F[3]:['No medications.','I have taken no medicine.','I did not take any medication.'],F[4]:['No known allergies.','I do not have any allergies.','I have no allergies.']}
HES=['I do not know.','I am not sure.','I have not checked.','I cannot remember.']
def build(si,i):
    rng=random.Random(637901+si*811+i*97)
    id=f'prospective-{si:02d}-{i:02d}'
    choices={f:rng.sample(VALS[f],2) for f in F}
    # Reference semantics are assigned independently of all product selectors.
    modes={f:rng.choices(['direct','missing','uncertain','denial'] if f in DEN else ['direct','missing','uncertain'],[5,1,1,2] if f in DEN else [5,1,1])[0] for f in F}
    focus_pool={1:[F[1],F[2]],3:[F[2],F[3]],4:[F[3],F[4]],8:[F[0],F[1],F[3]],9:[F[1]],10:[F[3],F[4]],11:[F[3],F[4]],13:[F[3],F[4]],14:[F[2],F[3]],15:[F[0],F[1],F[2]]}.get(si,list(F))
    focus=focus_pool[i%len(focus_pool)]
    modes[focus]=S[si]
    if si==0: modes={f:'direct' for f in F}
    if si==11: modes[F[3]]=modes[F[4]]='blanket_denials'
    order=list(F)
    if i%2: order=[F[0],F[3],F[4],F[1],F[2]]
    if i%3==0: order[1:]=order[1:][::-1]
    turns=[]; states={}; remarks={}
    def turn(w,t,f=None): turns.append((w,t,f))
    place=rng.choice(['east entrance','information desk','waiting area','blue door','side hallway','front desk'])
    if i%2==0: turn('Unknown',rng.choice(['The check-in slip is at the ','I left the folded map at the '])+place+'.')
    for f in order:
        mode=modes[f]; x,y=choices[f]
        q=rng.choice(Q[f])
        record=lambda role,line:turn(role,line,f)
        states[f]='supported'; remarks[f]=mode
        if mode=='missing': states[f]='missing'; continue
        if mode=='clinician_proposal':
            record('Doctor',f'Could your {LABEL[f]} be '+seen('clinician',x)+'?')
            record('Patient',rng.choice(['I cannot confirm that.','I have not established that.','I cannot say.']))
            states[f]='uncertain';continue
        if mode=='unrelated_clock':
            record('Doctor','When is the paperwork due?' if i%2 else 'How long has the medication refill been delayed?')
            record('Patient',('In ' if i%2 else 'The refill has been delayed for ')+seen('logistical' if i%2 else 'medication_interval',x)+'.')
            states[f]='missing'; continue
        record('Doctor',q)
        if mode in ('direct','ordinary'): record('Patient',direct(f,x,rng))
        elif mode in ('bounded_measure','qualified_phrase'):
            phrase=(rng.choice(['at least ','about ','nearly '])+x if f==F[1] else rng.choice(['at most ','about ','at least '])+rng.choice(['four out of ten','6/10','3/10','seven out of ten']) if f==F[2] else rng.choice(['intermittent ','constant '])+rng.choice(['right knee pain','left ear ringing','right shoulder pain']))
            record('Patient',say(f,G(phrase),rng))
        elif mode in ('denial','blanket_denials'):
            record('Patient',G(rng.choice(DEN[f])));states[f]='absent'
        elif mode=='uncertain':
            record('Patient',rng.choice(HES));states[f]='uncertain'
        elif mode=='explicit_replacement':
            record('Patient',say(f,seen('retracted',y),rng))
            record('Patient','I misspoke about my '+LABEL[f]+'. '+say(f,G(x),rng))
        elif mode=='historical_only':
            prefix='Last winter my pain was ' if f==F[2] else 'Last winter I took '
            record('Patient',prefix+seen('historical',x)+'. That was a different episode. I cannot give you a current answer.')
            states[f]='uncertain'
        elif mode=='prospective_only':
            text='I might take '+seen('prospective',x)+' next week. I have not decided.' if f==F[3] else 'I am scheduled to be tested for '+seen('prospective',x)+' next month. I do not know if I am allergic.'
            record('Patient',text); states[f]='uncertain'
        elif mode=='family_experiencer':
            prefix={F[0]:'My sister has ',F[1]:'My sister had symptoms for ',F[2]:'My sister rated her pain ',F[3]:'My sister takes ',F[4]:'My sister is allergic to '}[f]
            record('Patient',prefix+seen('family',x)+'. I do not know about myself.');states[f]='uncertain'
        elif mode=='unidentified_source':
            record(rng.choice(['Other','Unknown']),say(f,seen('other_speaker',x),rng))
            record('Patient','That was someone else speaking. I do not have an answer.')
            states[f]='uncertain'
        elif mode=='logistical_interruption':
            record('Patient',direct(f,x,rng))
            record('Unknown',rng.choice(['My receipt was left near the ','I had the wrong map for the ','That coat belongs at the '])+place+'.')
        elif mode=='unresolved_alternatives':
            p='I may have taken ' if f==F[3] else 'I might be allergic to '
            record('Patient',p+seen('alternative',x)+' or '+seen('alternative',y)+'. I cannot tell which.')
            states[f]='uncertain'
        elif mode=='withdrawn_without_replacement':
            record('Patient',say(f,seen('retracted',x),rng))
            record('Patient','I was mistaken about my '+LABEL[f]+'. I cannot give the right answer.');states[f]='uncertain'
        elif mode=='mixed_experiencers':
            who=rng.choice(['My partner','My brother','My mother','My neighbor'])
            record('Patient',who+(' takes ' if f==F[3] else ' is allergic to ')+seen('family',y)+'. '+say(f,G(x),rng))
        elif mode=='current_after_past':
            if f==F[2]: record('Patient','Yesterday the pain was '+seen('historical',y)+'. Right now my pain is '+G(x)+'.')
            else: record('Patient','I used to take '+seen('historical',y)+'. Right now I am taking '+G(x)+'.')
        else: raise ValueError(mode)
    if i%3==1: turn('Other','I have the visitor sticker for the '+place+'.')
    lines=[]; pos=0; gold={}; decoys=[]
    for role,marked,f in turns:
        if lines: pos+=1
        line=role+': '; cursor=0
        for m in re.finditer(r'«([^|]+)\|([^»]+)»',marked):
            line+=marked[cursor:m.start()]
            kind,value=m.groups(); start=pos+len(line);end=start+len(value)
            if kind=='G':
                assert role=='Patient' and f not in gold
                article=re.match(r'(?i)^(?:an? )',value) if f in (F[0],F[2]) and states[f]=='supported' else None
                trim=article.end() if article else 0
                gold[f]=dict(state=states[f],quote=value[trim:],start=start+trim,end=end)
            else: decoys.append(dict(field=f,kind=kind,role=role,quote=value,start=start,end=end))
            line+=value;cursor=m.end()
        line+=marked[cursor:]; assert '«' not in line and '»' not in line
        lines.append(line);pos+=len(line)
    transcript='\n'.join(lines)
    for f in F:
        if f not in gold:
            assert states[f] in ('uncertain','missing'),(id,f,states[f])
            gold[f]=dict(state=states[f])
    for a in [*gold.values(),*decoys]:
        if 'start' in a: assert transcript[a['start']:a['end']]==a['quote']
    return dict(id=id,stratum=S[si],instance=i,focus=focus,field_modes=remarks,transcript=transcript,gold={f:gold[f] for f in F},decoys=decoys)
if __name__=='__main__':
    rows=[build(s,i) for i in range(10) for s in range(16)]
    assert len({r['transcript'] for r in rows})==160
    counts=Counter(k['state'] for row in rows for k in row['gold'].values())
    p=ROOT/'fixture.json'
    with p.open('x') as w:json.dump({'seed':637901,'fields':F,'strata':S,'cases':rows},w,ensure_ascii=False,indent=2);w.write('\n')
    for r in rows:
        if r['id'] in ('prospective-12-00','prospective-13-00'): print(r['id'],r['transcript'],r['gold'])
    print(counts)
