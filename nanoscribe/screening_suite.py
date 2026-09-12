"""Deterministic P1 screening suite generator (no PHI)."""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import AtomSpec
from nanoscribe.encounter import (
    AssertionState,
    AtomType,
    Certainty,
    ClinicalAtom,
    EncounterRecord,
    Experiencer,
    Speaker,
    TemporalState,
    Temporality,
    assemble_source,
)
from nanoscribe.harness import HarnessCase, P1TestSet
from nanoscribe.select import relocate, _surface_hits, _span_in_turn

SCREENING_REVISION = "p1_screening_eval_v1"
SCREENING_SUITE_NAME = "p1_screening_eval_v1"
CORE_COUNT = 64
ADVERSARIAL_COUNT = 64

_SYMPTOMS = (
    "neck",
    "back",
    "chest",
    "headache",
    "cough",
    "fever",
    "nausea",
    "dizziness",
    "fatigue",
    "rash",
    "swelling",
    "pain",
    "pressure",
    "tingling",
    "weakness",
    "stiffness",
)

_ASSESSMENTS = (
    "cervical strain",
    "viral syndrome",
    "muscle strain",
    "tension headache",
    "allergic rhinitis",
    "gastroenteritis",
    "anxiety",
    "hypertension",
    "migraine",
    "sinusitis",
)

_MEDS = (
    "ibuprofen",
    "acetaminophen",
    "lisinopril",
    "metformin",
    "atorvastatin",
    "omeprazole",
    "amlodipine",
    "levothyroxine",
)


def _symptom_core(index: int) -> HarnessCase:
    symptom = _SYMPTOMS[index % len(_SYMPTOMS)]
    enc_id = f"core-sym-{index:02d}"
    patient_line = f"My {symptom} has been bothering me."
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.SYMPTOM,
        raw_value=symptom,
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "What brings you in?"),
            (Speaker.PATIENT, patient_line),
        ),
    )
    span = relocate(source, symptom, evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.SYMPTOM,
        raw_value=symptom,
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _denial_core(index: int) -> HarnessCase:
    enc_id = f"core-deny-{index:02d}"
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.ALLERGY,
        raw_value="allergies",
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "Any allergies?"),
            (Speaker.PATIENT, "No allergies."),
        ),
    )
    span = relocate(source, "No allergies.", evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.ALLERGY,
        raw_value="allergies",
        assertion_state=AssertionState.DENIED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _uncertainty_core(index: int) -> HarnessCase:
    symptom = _SYMPTOMS[(index + 3) % len(_SYMPTOMS)]
    enc_id = f"core-unc-{index:02d}"
    ev_id = f"ev-{enc_id}"
    line = f"Maybe some {symptom} sometimes."
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.SYMPTOM,
        raw_value=symptom,
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "Any symptoms?"),
            (Speaker.PATIENT, line),
        ),
    )
    span = relocate(source, symptom, evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.SYMPTOM,
        raw_value=symptom,
        assertion_state=AssertionState.UNCERTAIN,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.UNCERTAIN,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _assessment_core(index: int) -> HarnessCase:
    assess = _ASSESSMENTS[index % len(_ASSESSMENTS)]
    enc_id = f"core-assess-{index:02d}"
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.ASSESSMENT,
        raw_value=assess,
        speaker=Speaker.CLINICIAN,
        experiencer=Experiencer.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.PATIENT, "It hurts when I turn."),
            (Speaker.CLINICIAN, f"I think this is {assess}."),
        ),
    )
    span = relocate(source, assess, evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.ASSESSMENT,
        raw_value=assess,
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.CLINICIAN,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _medication_core(index: int) -> HarnessCase:
    med = _MEDS[index % len(_MEDS)]
    enc_id = f"core-med-{index:02d}"
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.MEDICATION,
        raw_value=med,
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "What do you take?"),
            (Speaker.PATIENT, f"I take {med} daily."),
        ),
    )
    span = relocate(source, med, evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.MEDICATION,
        raw_value=med,
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _family_history_adversarial(index: int) -> HarnessCase:
    condition = _ASSESSMENTS[index % len(_ASSESSMENTS)]
    enc_id = f"adv-fh-{index:02d}"
    ev_fh = f"ev-fh-{enc_id}"
    ev_sym = f"ev-sym-{enc_id}"
    spec_fh = AtomSpec(
        atom_id=f"atom-fh-{enc_id}",
        atom_type=AtomType.HISTORY,
        raw_value=condition,
        speaker=Speaker.PATIENT,
    )
    spec_sym = AtomSpec(
        atom_id=f"atom-sym-{enc_id}",
        atom_type=AtomType.SYMPTOM,
        raw_value="tired",
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "Family history?"),
            (Speaker.PATIENT, f"My mother had {condition}."),
            (Speaker.PATIENT, "I've been tired this week."),
        ),
    )
    fh_span = relocate(source, condition, evidence_id=ev_fh)
    tired_span = relocate(source, "tired", evidence_id=ev_sym)
    assert fh_span and tired_span
    atoms = (
        ClinicalAtom(
            atom_id=spec_fh.atom_id,
            atom_type=AtomType.HISTORY,
            raw_value=condition,
            assertion_state=AssertionState.ASSERTED,
            speaker=Speaker.PATIENT,
            experiencer=Experiencer.OTHER,
            temporality=TemporalState(kind=Temporality.HISTORICAL),
            certainty=Certainty.STATED,
            evidence_ids=(ev_fh,),
        ),
        ClinicalAtom(
            atom_id=spec_sym.atom_id,
            atom_type=AtomType.SYMPTOM,
            raw_value="tired",
            assertion_state=AssertionState.ASSERTED,
            speaker=Speaker.PATIENT,
            experiencer=Experiencer.PATIENT,
            temporality=TemporalState(kind=Temporality.CURRENT),
            certainty=Certainty.STATED,
            evidence_ids=(ev_sym,),
        ),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(fh_span, tired_span),
        atoms=atoms,
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec_fh, spec_sym),
    )


def _duplicate_mention_adversarial(index: int) -> HarnessCase:
    word = _SYMPTOMS[index % len(_SYMPTOMS)]
    enc_id = f"adv-dup-{index:02d}"
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.SYMPTOM,
        raw_value=word,
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.PATIENT, f"My {word} was fine last month."),
            (Speaker.PATIENT, f"Now my {word} is worse."),
        ),
    )
    # gold: second mention (current worsening)
    hits = _surface_hits(source, word)
    assert len(hits) >= 2
    start, end = hits[1]
    span = _span_in_turn(source, start, end, word, ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.SYMPTOM,
        raw_value=word,
        assertion_state=AssertionState.ASSERTED,
        speaker=Speaker.PATIENT,
        experiencer=Experiencer.PATIENT,
        temporality=TemporalState(kind=Temporality.CURRENT),
        certainty=Certainty.STATED,
        evidence_ids=(ev_id,),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(span,),
        atoms=(atom,),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def _abstention_adversarial(index: int) -> HarnessCase:
    enc_id = f"adv-abs-{index:02d}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.MEDICATION,
        raw_value="medication",
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "Any chest pain?"),
            (Speaker.PATIENT, "No, just tired."),
        ),
    )
    gold = EncounterRecord(
        encounter_id=enc_id,
        sources=(source,),
        evidence=(),
        atoms=(),
        unresolved=(),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=enc_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=enc_id),
        atom_specs=(spec,),
    )


def generate_core_case(index: int) -> HarnessCase:
    kind = index % 5
    if kind == 0:
        return _symptom_core(index)
    if kind == 1:
        return _denial_core(index)
    if kind == 2:
        return _uncertainty_core(index)
    if kind == 3:
        return _assessment_core(index)
    return _medication_core(index)


def generate_adversarial_case(index: int) -> HarnessCase:
    kind = index % 3
    if kind == 0:
        return _family_history_adversarial(index)
    if kind == 1:
        return _duplicate_mention_adversarial(index)
    return _abstention_adversarial(index)


def screening_core_cases() -> list[HarnessCase]:
    return [generate_core_case(i) for i in range(CORE_COUNT)]


def screening_adversarial_cases() -> list[HarnessCase]:
    return [generate_adversarial_case(i) for i in range(ADVERSARIAL_COUNT)]


def screening_manifest() -> dict[str, object]:
    core = screening_core_cases()
    adv = screening_adversarial_cases()
    return {
        "schema": "nano.screening.suite.v1",
        "suite": SCREENING_SUITE_NAME,
        "revision": SCREENING_REVISION,
        "partition": "FROZEN_SCREENING_EVAL",
        "eval_only": True,
        "P1_CORE": [c.encounter_id for c in core],
        "P1_ADVERSARIAL": [c.encounter_id for c in adv],
        "held_out_eval": True,
        "no_phi": True,
    }
