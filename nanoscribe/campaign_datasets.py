"""Versioned P1 campaign evaluation suites (no PHI).

Encounter *structure* is defined once per case and the *surface values* come
from `campaign_instances.INSTANCES`, so campaign_v2 is a multi-instance
instrument: 5 draws of the same 16 slots, reported as mean +- across-instance
SD rather than as one point estimate. Instance ``i0`` reproduces the original
hand-authored values exactly, so campaign_v1 and every claim scored on it stay
comparable.
"""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import AtomSpec
from nanoscribe.campaign_instances import (
    INSTANCE_IDS,
    INSTANCES,
    InstanceValues,
    instance,
    split_encounter_id,
)
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
    UnresolvedItem,
    assemble_source,
)
from nanoscribe.harness import HarnessCase, P1TestSet
from nanoscribe.select import relocate

CAMPAIGN_DATASET_REVISION = "campaign_v1_20260823"
CAMPAIGN_V2_DATASET_REVISION = "campaign_v2_multi_20260825"

CAMPAIGN_V1_ENCOUNTERS = ("enc-1", "enc-2", "enc-3")
CAMPAIGN_V2_ADDED_ENCOUNTERS = ("enc-4", "enc-5")
CAMPAIGN_V2_BASE_ENCOUNTERS = CAMPAIGN_V1_ENCOUNTERS + CAMPAIGN_V2_ADDED_ENCOUNTERS

_CURRENT = TemporalState(kind=Temporality.CURRENT)
_HISTORICAL = TemporalState(kind=Temporality.HISTORICAL)


def _eid(base: str, values: InstanceValues) -> str:
    return base if values.instance_id == "i0" else f"{base}@{values.instance_id}"


def _sid(base: str, values: InstanceValues) -> str:
    return base if values.instance_id == "i0" else f"{base}-{values.instance_id}"


# --------------------------------------------------------------------------
# enc-1 — mixed symptom / allergy-denial / history / clinician assessment
# --------------------------------------------------------------------------


def enc1_case(values: InstanceValues) -> HarnessCase:
    encounter_id = _eid("enc-1", values)
    source = assemble_source(
        _sid("src-1", values),
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, f"My {values.e1_site} has been hurting."),
            (Speaker.CLINICIAN, f"I think this is {values.e1_assessment}."),
            (Speaker.PATIENT, values.e1_denial),
            (Speaker.PATIENT, f"I used to have {values.e1_history} years ago."),
        ),
    )
    site = relocate(source, values.e1_site, evidence_id="ev-neck")
    deny = relocate(source, values.e1_denial, evidence_id="ev-deny")
    hist = relocate(source, values.e1_history, evidence_id="ev-hist")
    assess = relocate(source, values.e1_assessment, evidence_id="ev-assess")
    assert site and deny and hist and assess
    gold = EncounterRecord(
        encounter_id=encounter_id,
        sources=(source,),
        evidence=(site, deny, hist, assess),
        atoms=(
            ClinicalAtom(
                atom_id="atom-neck",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e1_site,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-neck",),
            ),
            ClinicalAtom(
                atom_id="atom-alg",
                atom_type=AtomType.ALLERGY,
                raw_value="allergies",
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-deny",),
            ),
            ClinicalAtom(
                atom_id="atom-hist",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e1_history,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_HISTORICAL,
                certainty=Certainty.STATED,
                evidence_ids=("ev-hist",),
            ),
            ClinicalAtom(
                atom_id="atom-assess",
                atom_type=AtomType.ASSESSMENT,
                raw_value=values.e1_assessment,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.CLINICIAN,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-assess",),
            ),
        ),
        unresolved=(
            UnresolvedItem(
                unresolved_id="medication",
                topic="medication",
                reason="no medication occurs anywhere in the source",
            ),
        ),
    )
    specs = (
        AtomSpec("atom-neck", AtomType.SYMPTOM, values.e1_site),
        AtomSpec("atom-alg", AtomType.ALLERGY, "allergies"),
        AtomSpec("atom-hist", AtomType.SYMPTOM, values.e1_history),
        AtomSpec(
            "atom-assess",
            AtomType.ASSESSMENT,
            values.e1_assessment,
            speaker=Speaker.CLINICIAN,
            experiencer=Experiencer.PATIENT,
        ),
        AtomSpec("medication", AtomType.MEDICATION, "medication"),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=encounter_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=encounter_id),
        atom_specs=specs,
    )


# --------------------------------------------------------------------------
# enc-2 — patient uncertainty, not denial
# --------------------------------------------------------------------------


def enc2_uncertainty_case(values: InstanceValues) -> HarnessCase:
    encounter_id = _eid("enc-2", values)
    source = assemble_source(
        _sid("src-2", values),
        (
            (Speaker.CLINICIAN, f"Any {values.e2_probe}?"),
            (Speaker.PATIENT, f"Maybe a little {values.e2_value} sometimes."),
            (Speaker.CLINICIAN, "We'll monitor it."),
        ),
    )
    span = relocate(source, values.e2_value, evidence_id="ev-pressure")
    assert span is not None
    gold = EncounterRecord(
        encounter_id=encounter_id,
        sources=(source,),
        evidence=(span,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-chest",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e2_value,
                assertion_state=AssertionState.UNCERTAIN,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.UNCERTAIN,
                evidence_ids=("ev-pressure",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-chest",
            atom_type=AtomType.SYMPTOM,
            raw_value=values.e2_value,
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=encounter_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=encounter_id),
        atom_specs=specs,
    )


# --------------------------------------------------------------------------
# enc-3 — family history vs patient symptom
# --------------------------------------------------------------------------


def enc3_family_history_case(values: InstanceValues) -> HarnessCase:
    encounter_id = _eid("enc-3", values)
    source = assemble_source(
        _sid("src-3", values),
        (
            (Speaker.CLINICIAN, "Family history?"),
            (Speaker.PATIENT, f"My mother had {values.e3_family}."),
            (Speaker.PATIENT, f"I've been {values.e3_symptom} this week."),
        ),
    )
    family = relocate(source, values.e3_family, evidence_id="ev-fh")
    symptom = relocate(source, values.e3_symptom, evidence_id="ev-tired")
    assert family and symptom
    gold = EncounterRecord(
        encounter_id=encounter_id,
        sources=(source,),
        evidence=(family, symptom),
        atoms=(
            ClinicalAtom(
                atom_id="atom-fh",
                atom_type=AtomType.HISTORY,
                raw_value=values.e3_family,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.OTHER,
                temporality=_HISTORICAL,
                certainty=Certainty.STATED,
                evidence_ids=("ev-fh",),
            ),
            ClinicalAtom(
                atom_id="atom-tired",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e3_symptom,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-tired",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-fh",
            atom_type=AtomType.HISTORY,
            raw_value=values.e3_family,
            speaker=Speaker.PATIENT,
            experiencer=Experiencer.OTHER,
            temporality=_HISTORICAL,
        ),
        AtomSpec(
            atom_id="atom-tired",
            atom_type=AtomType.SYMPTOM,
            raw_value=values.e3_symptom,
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=encounter_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=encounter_id),
        atom_specs=specs,
    )


# --------------------------------------------------------------------------
# enc-4 — absent-atom probe (the false-positive case campaign_v1 lacks)
# --------------------------------------------------------------------------


def enc4_absent_atom_case(values: InstanceValues) -> HarnessCase:
    """One present slot plus five whose value occurs nowhere in the source.

    A model that echoes the value named in its prompt produces a quote that
    cannot bind (counted as ``unbound_assertion``); a model that reads the
    transcript abstains (counted as ``correct_abstention``). campaign_v1 has no
    case like this, so echoing was previously invisible.
    """
    encounter_id = _eid("enc-4", values)
    source = assemble_source(
        _sid("src-4", values),
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (
                Speaker.PATIENT,
                f"My {values.e4_site} has been {values.e4_present} since Friday.",
            ),
            (Speaker.CLINICIAN, "Let's take a look."),
        ),
    )
    present = relocate(source, values.e4_present, evidence_id="ev-sore")
    assert present is not None
    gold = EncounterRecord(
        encounter_id=encounter_id,
        sources=(source,),
        evidence=(present,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-throat",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e4_present,
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-sore",),
            ),
        ),
        unresolved=tuple(
            UnresolvedItem(
                unresolved_id=atom_id,
                topic=value,
                reason=f"{value!r} occurs nowhere in the source",
            )
            for atom_id, _atom_type, value in values.e4_absent
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-throat",
            atom_type=AtomType.SYMPTOM,
            raw_value=values.e4_present,
            speaker=Speaker.PATIENT,
        ),
    ) + tuple(
        AtomSpec(
            atom_id=atom_id,
            atom_type=atom_type,
            raw_value=value,
            speaker=Speaker.PATIENT,
        )
        for atom_id, atom_type, value in values.e4_absent
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=encounter_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=encounter_id),
        atom_specs=specs,
    )


# --------------------------------------------------------------------------
# enc-5 — explicit denial under a STATED-biased answer template
# --------------------------------------------------------------------------


def enc5_denial_case(values: InstanceValues) -> HarnessCase:
    encounter_id = _eid("enc-5", values)
    source = assemble_source(
        _sid("src-5", values),
        (
            (Speaker.CLINICIAN, "Do you have any habits we should note?"),
            (Speaker.PATIENT, f"No, I have never {values.e5_denied_history}."),
            (Speaker.CLINICIAN, "Any breathing trouble?"),
            (Speaker.PATIENT, f"No {values.e5_denied_symptom} at all."),
        ),
    )
    habit = relocate(source, values.e5_denied_history, evidence_id="ev-smoked")
    breath = relocate(source, values.e5_denied_symptom, evidence_id="ev-wheezing")
    assert habit and breath
    gold = EncounterRecord(
        encounter_id=encounter_id,
        sources=(source,),
        evidence=(habit, breath),
        atoms=(
            ClinicalAtom(
                atom_id="atom-smoke",
                atom_type=AtomType.HISTORY,
                raw_value=values.e5_denied_history,
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_HISTORICAL,
                certainty=Certainty.STATED,
                evidence_ids=("ev-smoked",),
            ),
            ClinicalAtom(
                atom_id="atom-wheeze",
                atom_type=AtomType.SYMPTOM,
                raw_value=values.e5_denied_symptom,
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=_CURRENT,
                certainty=Certainty.STATED,
                evidence_ids=("ev-wheezing",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-smoke",
            atom_type=AtomType.HISTORY,
            raw_value=values.e5_denied_history,
            speaker=Speaker.PATIENT,
            temporality=_HISTORICAL,
        ),
        AtomSpec(
            atom_id="atom-wheeze",
            atom_type=AtomType.SYMPTOM,
            raw_value=values.e5_denied_symptom,
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id=encounter_id,
        gold=gold,
        model_input=ModelInput(source=source, encounter_id=encounter_id),
        atom_specs=specs,
    )


_BUILDERS = {
    "enc-1": enc1_case,
    "enc-2": enc2_uncertainty_case,
    "enc-3": enc3_family_history_case,
    "enc-4": enc4_absent_atom_case,
    "enc-5": enc5_denial_case,
}


def case_for(base_encounter_id: str, instance_id: str = "i0") -> HarnessCase:
    return _BUILDERS[base_encounter_id](instance(instance_id))


def instance_cases(instance_id: str) -> list[HarnessCase]:
    """All five encounters for one instance draw."""
    values = instance(instance_id)
    return [_BUILDERS[base](values) for base in CAMPAIGN_V2_BASE_ENCOUNTERS]


def tiny_fixture_case() -> HarnessCase:
    case = enc1_case(instance("i0"))
    return HarnessCase(
        test_set=P1TestSet.TINY_FIXTURE,
        encounter_id=case.encounter_id,
        gold=case.gold,
        model_input=case.model_input,
        atom_specs=case.atom_specs,
    )


def fixture_lines_for_encounter(encounter_id: str) -> dict[str, str]:
    """Deterministic span-port lines for campaign fixture evaluation (no weights)."""
    base, instance_id = split_encounter_id(encounter_id)
    values = instance(instance_id)
    if base == "enc-1":
        return {
            "atom-neck": f'STATED: "{values.e1_site}"',
            "atom-alg": f'DENIED: "{values.e1_denial}"',
            "atom-hist": f'STATED: "{values.e1_history}"',
            "atom-assess": f'STATED: "{values.e1_assessment}"',
            "medication": "NOT_MENTIONED",
        }
    if base == "enc-2":
        return {"atom-chest": f'UNCERTAIN: "{values.e2_value}"'}
    if base == "enc-3":
        return {
            "atom-fh": f'STATED: "{values.e3_family}"',
            "atom-tired": f'STATED: "{values.e3_symptom}"',
        }
    if base == "enc-4":
        return {
            "atom-throat": f'STATED: "{values.e4_present}"',
            **{atom_id: "NOT_MENTIONED" for atom_id, _t, _v in values.e4_absent},
        }
    if base == "enc-5":
        return {
            "atom-smoke": f'DENIED: "{values.e5_denied_history}"',
            "atom-wheeze": f'DENIED: "{values.e5_denied_symptom}"',
        }
    raise KeyError(f"no fixture lines for encounter_id={encounter_id}")


DEFAULT_BASELINE_LINES = fixture_lines_for_encounter("enc-1")


def campaign_cases(suite: str) -> list[HarnessCase]:
    """Return harness cases for a named campaign partition."""
    key = suite.strip().lower()
    if key == "tiny_fixture":
        return [tiny_fixture_case()]
    if key == "p1_core":
        return [case_for("enc-1"), case_for("enc-2")]
    if key == "p1_adversarial":
        return [case_for("enc-3")]
    if key == "campaign_v1":
        # Single instance (i0) — the historical partition, left untouched.
        return [case_for(base) for base in CAMPAIGN_V1_ENCOUNTERS]
    if key == "campaign_v2":
        # Multi-instance: 5 draws of the same structure, 16 slots each.
        return [case for iid in INSTANCE_IDS for case in instance_cases(iid)]
    raise ValueError(f"unknown campaign suite: {suite}")


def dataset_revision_for(suite: str) -> str:
    """Revision string for a named suite."""
    if suite.strip().lower() == "campaign_v2":
        return CAMPAIGN_V2_DATASET_REVISION
    return CAMPAIGN_DATASET_REVISION


def suite_manifest() -> dict[str, object]:
    return {
        "schema": "nano.campaign.dataset.v1",
        "revision": CAMPAIGN_DATASET_REVISION,
        "partitions": {
            "P1_CORE": ["enc-1", "enc-2"],
            "P1_ADVERSARIAL": ["enc-3"],
            "tiny_fixture": ["enc-1"],
            "campaign_v1": list(CAMPAIGN_V1_ENCOUNTERS),
            "campaign_v2": [
                f"{base}@{iid}"
                for iid in INSTANCE_IDS
                for base in CAMPAIGN_V2_BASE_ENCOUNTERS
            ],
        },
        "campaign_v2_revision": CAMPAIGN_V2_DATASET_REVISION,
        "campaign_v2_instances": list(INSTANCE_IDS),
        "no_phi": True,
    }


__all__ = [
    "CAMPAIGN_DATASET_REVISION",
    "CAMPAIGN_V1_ENCOUNTERS",
    "CAMPAIGN_V2_ADDED_ENCOUNTERS",
    "CAMPAIGN_V2_BASE_ENCOUNTERS",
    "CAMPAIGN_V2_DATASET_REVISION",
    "DEFAULT_BASELINE_LINES",
    "INSTANCES",
    "INSTANCE_IDS",
    "campaign_cases",
    "case_for",
    "dataset_revision_for",
    "fixture_lines_for_encounter",
    "instance_cases",
    "split_encounter_id",
    "suite_manifest",
    "tiny_fixture_case",
]
