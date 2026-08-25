"""Disjoint P1 distillation training generator — NOT eval-held-out cases."""

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
from nanoscribe.select import relocate

DISTILL_TRAIN_REVISION = "p1_distill_train_v1"
DISTILL_TRAIN_COUNT = 96
SEED_NAMESPACE = "distill_train_v1"

_SYMPTOMS = (
    "sore throat",
    "joint ache",
    "shortness of breath",
    "blurred vision",
    "heartburn",
    "chills",
    "numbness",
    "palpitations",
    "constipation",
    "insomnia",
    "wheezing",
    "bruising",
)

_ASSESSMENTS = (
    "reactive airway",
    "tension myalgia",
    "contact dermatitis",
    "orthostatic hypotension",
    "bacterial pharyngitis",
    "cluster headache",
    "GERD flare",
    "plantar fasciitis",
)

_MEDS = (
    "sertraline",
    "losartan",
    "pantoprazole",
    "rosuvastatin",
    "albuterol",
    "gabapentin",
    "hydrochlorothiazide",
    "montelukast",
)


def _symptom_case(index: int) -> HarnessCase:
    symptom = _SYMPTOMS[index % len(_SYMPTOMS)]
    enc_id = f"train-sym-{index:02d}"
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
            (Speaker.CLINICIAN, "What symptoms are you having?"),
            (Speaker.PATIENT, f"I have {symptom}."),
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


def _denial_case(index: int) -> HarnessCase:
    enc_id = f"train-deny-{index:02d}"
    ev_id = f"ev-{enc_id}"
    spec = AtomSpec(
        atom_id=f"atom-{enc_id}",
        atom_type=AtomType.ALLERGY,
        raw_value="penicillin",
        speaker=Speaker.PATIENT,
    )
    source = assemble_source(
        f"src-{enc_id}",
        (
            (Speaker.CLINICIAN, "Any medication allergies?"),
            (Speaker.PATIENT, "No penicillin allergy."),
        ),
    )
    span = relocate(source, "No penicillin allergy.", evidence_id=ev_id)
    assert span is not None
    atom = ClinicalAtom(
        atom_id=spec.atom_id,
        atom_type=AtomType.ALLERGY,
        raw_value="penicillin",
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


def _assessment_case(index: int) -> HarnessCase:
    assess = _ASSESSMENTS[index % len(_ASSESSMENTS)]
    enc_id = f"train-assess-{index:02d}"
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
            (Speaker.PATIENT, "The pain comes and goes."),
            (Speaker.CLINICIAN, f"Likely {assess}."),
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


def _medication_case(index: int) -> HarnessCase:
    med = _MEDS[index % len(_MEDS)]
    enc_id = f"train-med-{index:02d}"
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
            (Speaker.CLINICIAN, "Current medications?"),
            (Speaker.PATIENT, f"I use {med} at night."),
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


def generate_train_case(index: int) -> HarnessCase:
    kind = index % 4
    if kind == 0:
        return _symptom_case(index)
    if kind == 1:
        return _denial_case(index)
    if kind == 2:
        return _assessment_case(index)
    return _medication_case(index)


def distill_train_cases() -> list[HarnessCase]:
    return [generate_train_case(i) for i in range(DISTILL_TRAIN_COUNT)]


def distill_train_manifest() -> dict[str, object]:
    cases = distill_train_cases()
    return {
        "schema": "nano.distill.train.v1",
        "revision": DISTILL_TRAIN_REVISION,
        "seed_namespace": SEED_NAMESPACE,
        "partition": "TRAIN",
        "n_cases": len(cases),
        "encounter_ids": [case.encounter_id for case in cases],
        "held_out_eval": False,
        "no_phi": True,
    }
