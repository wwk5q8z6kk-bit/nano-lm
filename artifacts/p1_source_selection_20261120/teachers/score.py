"""Replay sealed synthetic inputs and saved, label-blind predictions through NanoScribe offline."""
import argparse
import collections
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCALAR = {'supported', 'absent'}


def sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass
class Claim:
    state: str
    spans: list
    malformed: bool = False


@dataclass
class Note:
    claims: dict
    usable: bool = True


def raw_note(case, text):
    from nanoscribe.grounded import parse_grounded_summary
    try:
        _, proposals, errors, _ = parse_grounded_summary(text, case['transcript'])
    except ValueError:
        return Note({field: Claim('malformed', [], True) for field in case['gold']})
    fields = {}
    for field, p, error in zip(case['gold'], proposals, errors, strict=True):
        if p:
            assert p.field.value == field
        fields[field] = Claim(p.state.value if p else 'malformed',
                              [(s.start, s.end, s.text) for s in p.spans] if p else [], bool(error))
    return Note(fields)


def checked_note(case, text):
    from nanoscribe.compact import compact_solver
    from nanoscribe.scribe import scribe
    result = scribe(case['transcript'], encounter_id=case['id'],
                    solver=compact_solver(lambda _: text, binder='grounded'), context_policy='grounded')
    claims = {r['field']: r for r in (result.get('note') or {}).get('claims', [])}
    evidence = {r['evidence_id']: r for r in (result.get('encounter') or {}).get('evidence', [])}
    fields = {}
    for field in case['gold']:
        slot = claims.get(field, {'state': 'withheld', 'evidence_ids': []})
        refs = [evidence[key] for key in slot['evidence_ids']]
        for ref in refs:
            assert case['transcript'][ref['start']:ref['end']] == ref['text']
            if slot['state'] in SCALAR:
                assert ref['speaker'] == 'patient'
        fields[field] = Claim(slot['state'], [(r['start'], r['end'], r['text']) for r in refs])
    usable = result['status'] == 'draft' and (result.get('verification') or {}).get('passed') is True
    return Note(fields, usable)


def category(claim, gold):
    if claim.malformed:
        return 'malformed'
    if gold['state'] not in SCALAR:
        return 'unsupported_assertion' if claim.state in SCALAR else 'correct_withholding'
    if claim.state not in SCALAR:
        return 'missed_explicit_assertion'
    if claim.state != gold['state']:
        return 'wrong_polarity'
    if len(claim.spans) != 1:
        return 'non_scalar'
    left, right, _ = claim.spans[0]
    start, end = gold['start'], gold['end']
    if (left, right) == (start, end):
        return 'exact_assertion'
    if left <= start and right >= end:
        return 'overlong_containing_gold'
    if start <= left and end >= right:
        return 'shorter_within_gold'
    if left < end and start < right:
        return 'partial_overlap'
    return 'different_source_interval'


def measure(case, note):
    fields = {}
    for field, gold in case['gold'].items():
        claim = note.claims[field]
        status = category(claim, gold) if note.usable else 'unusable_note'
        emitted = claim.state in SCALAR and not claim.malformed
        explicit = gold['state'] in SCALAR
        decoys = {d['kind'] for d in case['decoys'] if d['field'] == field and any(
            a < d['end'] and d['start'] < b for a, b, _ in claim.spans)} if emitted else set()
        counts = collections.Counter(fields=1, exact=status in ('correct_withholding', 'exact_assertion'),
                                     exact_state=note.usable and claim.state == gold['state'], asserted=emitted,
                                     gold_explicit=explicit, licensing_errors=emitted and
                                     (not explicit or claim.state != gold['state'] or bool(decoys)))
        counts[status] += 1
        counts.update('tagged_' + d for d in decoys)
        fields[field] = (counts, status)
    return fields


def make_slots(case, source, strategy, teacher=None, control=None):
    from nanoscribe.grounded import CandidateSpan, FIELD_ORDER, build_grounded_english_menu, serialize_grounded_slot
    used = set()
    slots = []
    for field in FIELD_ORDER:
        name = field.value
        code, spans = 'M', ()
        if strategy == 'control':
            opts = list(build_grounded_english_menu(source, field, used).values())
            choices = {(c, s[0].start, s[0].end): (c, s) for c, s in opts if c in ('S', 'A') and len(s) == 1}
            if len(choices) == 1 and not any(c in ('C', 'U') and s for c, s in opts):
                code, spans = next(iter(choices.values()))
        else:
            if strategy == 'oracle':
                g = case['gold'][name]
                winner = Claim(g['state'], [(g['start'], g['end'], g['quote'])]) if g['state'] in SCALAR else None
            else:
                t, c = teacher.claims[name], control.claims[name]
                t_scalar = t if t.state in SCALAR and len(t.spans) == 1 else None
                c_scalar = c if c.state in SCALAR and len(c.spans) == 1 else None
                if strategy == 'agreement':
                    winner = t_scalar if t_scalar and t_scalar == c_scalar else None
                else:
                    winner = t_scalar or (c_scalar if t.state in ('missing', 'withheld') else None)
            if winner:
                a, b, text = winner.spans[0]
                code = 'A' if winner.state == 'absent' else 'S'
                spans = (CandidateSpan(start=a, end=b, text=text),)
        used.update((s.start, s.end) for s in spans)
        slots.append(serialize_grounded_slot(source, field, code, spans))
    return '|'.join(slots)


def rates(counts):
    summary = dict(counts)
    summary['exact_usable_yield'] = round(counts['exact_assertion'] / counts['gold_explicit'], 4) if counts['gold_explicit'] else None
    summary['exact_assertion_precision'] = round(counts['exact_assertion'] / counts['asserted'], 4) if counts['asserted'] else None
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True, help='Nano-LM source checkout')
    parser.add_argument('--output', type=Path, help='write the computed JSON to a new file')
    parser.add_argument('--candidate', action='store_true', help='measure another checkout without enforcing pinned source or expected scores')
    args = parser.parse_args()
    repo = args.repo.resolve()
    seal = json.loads((HERE / 'seal.json').read_text())
    for name, expected in seal['inputs'].items():
        if name.startswith('../master-source/'):
            if args.candidate:
                continue
            path = repo / name.removeprefix('../master-source/')
        else:
            path = HERE / name
        if sha(path.read_bytes()) != expected:
            raise ValueError(f'sealed file hash mismatch for {name}')
    sys.path.insert(0, str(repo))
    import nanoscribe.grounded as grounded
    assert Path(grounded.__file__).resolve() == repo / 'nanoscribe/grounded.py'
    from nanoscribe.compact import SYSTEM_PROMPT
    runtime = json.loads((HERE / 'generation.json').read_text())
    if sha(SYSTEM_PROMPT.encode()) != runtime['system_prompt_sha256']:
        raise ValueError('system prompt differs from the recorded generation prompt')
    predictions = [json.loads(line) for line in (HERE / 'predictions.jsonl').read_text().splitlines()]
    lookup = {(r['arm'], r['id']): r for r in predictions}
    cases = json.loads((HERE / 'fixture.json').read_text())['cases']
    assert len(cases) == 160 and len(lookup) == len(predictions) == 2 * len(cases)
    totals = collections.defaultdict(collections.Counter)
    subsets = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    stage1 = collections.Counter(); paired = collections.Counter()
    for c in cases:
        doc = grounded.transcript_source(grounded.NanoInput(c['id'], c['transcript']))
        control = checked_note(c, make_slots(c, doc, 'control'))
        scores = {'control': measure(c, control), 'oracle': measure(c, checked_note(c, make_slots(c, doc, 'oracle')))}
        for arm in ('flash', 'pro'):
            row = lookup[(arm, c['id'])]
            assert sha(c['transcript'].encode()) == row['source_sha256']
            request = {'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
                       'contents': [{'role':'user', 'parts':[{'text':c['transcript']}]}],
                       'generationConfig': runtime['models'][arm]['generationConfig']}
            assert sha(json.dumps(request, ensure_ascii=False).encode()) == row['request_sha256']
            wire = row['raw'].strip()
            notes = {'raw': raw_note(c, wire), 'checked': checked_note(c, wire)}
            scores[arm + '_raw'] = measure(c, notes['raw'])
            scores[arm] = measure(c, notes['checked'])
            for mode in ('agreement', 'recovery'):
                scores[arm + '_' + mode] = measure(c, checked_note(c, make_slots(c, doc, mode, notes['checked'], control)))
        for arm, fields in scores.items():
            for field, (counts, _) in fields.items():
                totals[arm].update(counts)
                subsets['field/' + field][arm].update(counts)
                subsets['stratum/' + c['stratum']][arm].update(counts)
        for field in c['gold']:
            a, sa = scores['flash'][field]; b, sb = scores['pro'][field]
            p = dict(fields=1, different_exact=a['exact'] != b['exact'], flash_only_exact=a['exact'] and not b['exact'],
                     pro_only_exact=b['exact'] and not a['exact'], different_licensing=a['licensing_errors'] != b['licensing_errors'],
                     pro_net_exact_assertions=int(sb == 'exact_assertion') - int(sa == 'exact_assertion'))
            paired.update(p)
            if c['instance'] < 2:
                stage1.update(p)
    output = {'summary': {k: rates(v) for k,v in totals.items()}, 'paired': dict(paired), 'stage1_paired': dict(stage1),
              'subsets': {group: {arm:rates(rows) for arm, rows in arms.items()} for group, arms in subsets.items()}}
    if not args.candidate and output != json.loads((HERE / 'expected.json').read_text()):
        raise RuntimeError('replay disagrees with expected aggregate, field, stratum or pairing outputs')
    if args.output:
        with args.output.open('x') as out:
            json.dump(output, out, indent=2); out.write('\n')
    for arm, stats in output['summary'].items():
        print(f"{arm:18} {stats['exact']:3}/{stats['fields']} exact; {stats.get('exact_assertion',0):3} exact assertions; {stats['licensing_errors']} licensing errors")


if __name__ == '__main__':
    main()
