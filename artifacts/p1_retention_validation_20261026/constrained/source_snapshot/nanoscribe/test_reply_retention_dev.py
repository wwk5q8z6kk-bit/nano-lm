"""New implementation cases, authored before reading the independent cohort."""
import pytest

from nanoscribe.compact import LocalCompactPredictor, compact_solver
from nanoscribe.scribe import scribe


def checked(source, **slots):
    raw = '|'.join(f'{label}:{slots.get(label, "M")}'
                   for label in ('CC', 'DUR', 'SEV', 'MED', 'ALG'))
    result = scribe(source, context_policy='context',
                    solver=compact_solver(lambda _: raw, binder='context'))
    assert result['verification']['passed'], result
    for span in result['encounter']['evidence']:
        assert source[span['start']:span['end']] == span['text']
    return result


@pytest.mark.parametrize('question,reply,label,value,index', [
    ('When did it start?', 'It started six days ago. I remember the date.', 'DUR', 'six days', 1),
    ('When did it start?', 'It started six days ago. I cannot identify the tablet I tried.', 'DUR', 'six days', 1),
    ('How intense?', "It is moderate. That's all.", 'SEV', 'moderate', 2),
    ('Which medicine did you try?', 'I took naproxen today. Thanks.', 'MED', 'naproxen', 3),
    ('Which medicine did you try?', 'I used ibuprofen this morning. I cannot rate the pain.', 'MED', 'ibuprofen', 3),
    ('Any allergies?', 'Latex gives me hives. Thank you.', 'ALG', 'Latex', 4),
    ('What symptoms?', 'I have wrist stiffness. I am not sure how long it has lasted.', 'CC', 'wrist stiffness', 0),
])
def test_supported_clause_survives_separate_uncertainty_and_review_text(question, reply, label, value, index):
    source = f'Doctor: {question}\r\nPatient: {reply}'
    result = checked(source, **{label: f'S[{value}]'})
    assert result['note']['claims'][index]['state'] == 'supported'
    assert result['note']['claims'][index]['value'] == value.casefold()
    assert result['disposition'] == 'review'
    assert any(e['text'] == reply for e in result['encounter']['evidence'])


@pytest.mark.parametrize('tail', [
    'Actually, I meant ten days.', 'That was last year.', 'No, I did not.',
    "That was my sister's answer.", 'I was mistaken.',
])
def test_correction_tail_cannot_leave_stale_scalar_supported(tail):
    source = 'Doctor: When did it start?\nPatient: It started six days ago. ' + tail
    result = checked(source, DUR='S[six days]')
    assert result['note']['claims'][1]['state'] == 'uncertain'


def test_two_concrete_clauses_retain_two_fields_with_original_offsets():
    source = ('Doctor: Which medicine did you try?\r\n'
              'Patient: I took aspirin this evening. I am allergic to nickel.')
    result = checked(source, MED='S[aspirin]', ALG='S[nickel]')
    assert [c['state'] for c in result['note']['claims'][3:]] == ['supported', 'supported']


@pytest.mark.parametrize('reply', [
    'I took no aspirin today.', 'I might have taken aspirin today.',
    'I took aspirin last year.', 'I took aspirin and zinc today.',
    'My sister took aspirin today.', 'I considered taking aspirin today.',
])
def test_contextual_medication_clause_never_becomes_actual_use(reply):
    result = checked('Doctor: Which medicine did you try?\nPatient: ' + reply, MED='S[aspirin]')
    assert result['note']['claims'][3]['state'] == 'uncertain'


def test_decoder_offers_shared_exact_clauses_without_arbitrary_fragments(tmp_path, monkeypatch):
    source = ('Doctor: When did it start?\nPatient: It began seven days ago. Thanks.\n'
              'Doctor: Which medicine did you try?\nPatient: I took ibuprofen today. I cannot rate the pain.')
    menus = {}
    predictor = LocalCompactPredictor(str(tmp_path), decoding='constrained')
    def choose(messages, choices):
        label = messages[-1]['content'].split('|')[-1]
        menus[label] = choices
        return ':M', {}
    monkeypatch.setattr(predictor, '_choose', choose)
    predictor(source)
    assert ':S[seven days]' in menus['DUR']
    assert ':S[ibuprofen]' in menus['MED']
    assert ':S[ibuprofen today]' not in menus['MED']
    assert not any(c.startswith(':S[') for c in menus['CC'])


def test_other_field_answer_does_not_turn_unanswered_request_into_missing():
    source = 'Doctor: Which medicine did you try?\nPatient: I have nausea.'
    result = checked(source, CC='S[nausea]')
    assert result['note']['claims'][0]['state'] == 'supported'
    assert result['note']['claims'][3]['state'] == 'uncertain'


@pytest.mark.parametrize('reply', ['It helped.', "It didn't help.", 'I could finally sleep.'])
def test_effect_only_followup_preserves_actual_medication(reply):
    source = ('Doctor: Which medicine did you try?\nPatient: I took fexofenadine.\n'
              'Doctor: Has the medication helped?\nPatient: ' + reply)
    result = checked(source, MED='S[fexofenadine]')
    assert result['note']['claims'][3]['state'] == 'supported'
    assert any(e['text'] == reply for e in result['encounter']['evidence'])


@pytest.mark.parametrize('reply', ["I didn't take it.", 'It was last year.', 'I could finally sleep. I made that up.'])
def test_effect_question_cannot_hide_retraction_or_scope_change(reply):
    source = ('Doctor: Which medicine did you try?\nPatient: I took fexofenadine.\n'
              'Doctor: Has the medication helped?\nPatient: ' + reply)
    result = checked(source, MED='S[fexofenadine]')
    assert result['note']['claims'][3]['state'] == 'uncertain'


@pytest.mark.parametrize('tail', [
    'That was last year.', 'That was before this episode.',
    'It was hypothetical.', 'That belonged to my brother.',
])
def test_consecutive_patient_turn_keeps_local_correction_scope(tail):
    source = ('Doctor: What symptoms?\nPatient: I have shoulder soreness.\n'
              'Doctor: Which medicine did you try?\nPatient: I took ketoprofen.\nPatient: ' + tail)
    result = checked(source, CC='S[shoulder soreness]', MED='S[ketoprofen]')
    assert result['note']['claims'][0]['state'] == 'supported'
    assert result['note']['claims'][3]['state'] == 'uncertain'


def test_nonmedication_relation_does_not_pass_administration_wrapper():
    source = 'Doctor: Which medicine did you try?\nPatient: I took an allergy test.'
    result = checked(source, MED='S[an allergy test]')
    assert result['note']['claims'][3]['state'] == 'uncertain'


@pytest.mark.parametrize('value', ['The medication', 'An allergy test'])
def test_reaction_relation_requires_named_allergen(value):
    source = 'Doctor: Any allergies?\nPatient: ' + value + ' gives me hives.'
    result = checked(source, ALG='S[' + value + ']')
    assert result['note']['claims'][4]['state'] == 'uncertain'
