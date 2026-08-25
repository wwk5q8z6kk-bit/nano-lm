"""Versioned P1 campaign evaluation suites (no PHI)."""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import AtomSpec, default_baseline_specs
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
from nanoscribe.tracks import tiny_fixture_case

CAMPAIGN_DATASET_REVISION = "campaign_v1_20260823"
CAMPAIGN_V2_DATASET_REVISION = "campaign_v2_20260825"


def _specs_for_enc1() -> tuple[AtomSpec, ...]:
    return default_baseline_specs()


def enc2_uncertainty_case() -> HarnessCase:
    """Patient expresses uncertainty about symptom — not denial."""
    source = assemble_source(
        "src-2",
        (
            (Speaker.CLINICIAN, "Any chest pain?"),
            (Speaker.PATIENT, "Maybe a little pressure sometimes."),
            (Speaker.CLINICIAN, "We'll monitor it."),
        ),
    )
    pressure = relocate(source, "pressure", evidence_id="ev-pressure")
    assert pressure is not None
    gold = EncounterRecord(
        encounter_id="enc-2",
        sources=(source,),
        evidence=(pressure,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-chest",
                atom_type=AtomType.SYMPTOM,
                raw_value="pressure",
                assertion_state=AssertionState.UNCERTAIN,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.UNCERTAIN,
                evidence_ids=("ev-pressure",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-chest",
            atom_type=AtomType.SYMPTOM,
            raw_value="pressure",
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id="enc-2",
        gold=gold,
        model_input=ModelInput(source=source, encounter_id="enc-2"),
        atom_specs=specs,
    )


def enc3_family_history_case() -> HarnessCase:
    """Family history attribution — patient symptom vs family history distractor."""
    source = assemble_source(
        "src-3",
        (
            (Speaker.CLINICIAN, "Family history?"),
            (Speaker.PATIENT, "My mother had diabetes."),
            (Speaker.PATIENT, "I've been tired this week."),
        ),
    )
    diabetes = relocate(source, "diabetes", evidence_id="ev-fh")
    tired = relocate(source, "tired", evidence_id="ev-tired")
    assert diabetes and tired
    gold = EncounterRecord(
        encounter_id="enc-3",
        sources=(source,),
        evidence=(diabetes, tired),
        atoms=(
            ClinicalAtom(
                atom_id="atom-fh",
                atom_type=AtomType.HISTORY,
                raw_value="diabetes",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.OTHER,
                temporality=TemporalState(kind=Temporality.HISTORICAL),
                certainty=Certainty.STATED,
                evidence_ids=("ev-fh",),
            ),
            ClinicalAtom(
                atom_id="atom-tired",
                atom_type=AtomType.SYMPTOM,
                raw_value="tired",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-tired",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-fh",
            atom_type=AtomType.HISTORY,
            raw_value="diabetes",
            speaker=Speaker.PATIENT,
            experiencer=Experiencer.OTHER,
            temporality=TemporalState(kind=Temporality.HISTORICAL),
        ),
        AtomSpec(
            atom_id="atom-tired",
            atom_type=AtomType.SYMPTOM,
            raw_value="tired",
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id="enc-3",
        gold=gold,
        model_input=ModelInput(source=source, encounter_id="enc-3"),
        atom_specs=specs,
    )


# Slots whose value occurs nowhere in enc-4's source. Five of them, spread over
# distinct atom types, so the false-positive probe is not resting on n=2 and so
# the slots stay distinguishable even under the Q-off diagnostic cell.
_ENC4_ABSENT: tuple[tuple[str, AtomType, str], ...] = (
    ("atom-absent-med", AtomType.MEDICATION, "lisinopril"),
    ("atom-absent-fever", AtomType.SYMPTOM, "fever"),
    ("atom-absent-allergy", AtomType.ALLERGY, "penicillin"),
    ("atom-absent-hist", AtomType.HISTORY, "asthma"),
    ("atom-absent-vital", AtomType.MEASUREMENT, "blood pressure"),
)


def enc4_absent_atom_case() -> HarnessCase:
    """Slots whose value never occurs in the source — the false-positive probe.

    campaign_v1 has no case like this: every slot's gold value is present, so a
    model that simply echoes the value named in its prompt scores full marks and
    the leakage is invisible. Here two of the three slots are unresolved, so
    echoing produces a quote that cannot bind to the source and is counted
    (spurious / malformed / critical), while reading the transcript produces
    NOT_MENTIONED and counts as a correct abstention.
    """
    source = assemble_source(
        "src-4",
        (
            (Speaker.CLINICIAN, "What brings you in today?"),
            (Speaker.PATIENT, "My throat has been sore since Friday."),
            (Speaker.CLINICIAN, "Let's take a look."),
        ),
    )
    throat = relocate(source, "sore", evidence_id="ev-sore")
    assert throat is not None
    gold = EncounterRecord(
        encounter_id="enc-4",
        sources=(source,),
        evidence=(throat,),
        atoms=(
            ClinicalAtom(
                atom_id="atom-throat",
                atom_type=AtomType.SYMPTOM,
                raw_value="sore",
                assertion_state=AssertionState.ASSERTED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
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
            for atom_id, _atom_type, value in _ENC4_ABSENT
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-throat",
            atom_type=AtomType.SYMPTOM,
            raw_value="sore",
            speaker=Speaker.PATIENT,
        ),
    ) + tuple(
        AtomSpec(
            atom_id=atom_id,
            atom_type=atom_type,
            raw_value=value,
            speaker=Speaker.PATIENT,
        )
        for atom_id, atom_type, value in _ENC4_ABSENT
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id="enc-4",
        gold=gold,
        model_input=ModelInput(source=source, encounter_id="enc-4"),
        atom_specs=specs,
    )


def enc5_denial_case() -> HarnessCase:
    """Values present but explicitly denied — assertion state under a STATED-biased hint."""
    source = assemble_source(
        "src-5",
        (
            (Speaker.CLINICIAN, "Do you smoke?"),
            (Speaker.PATIENT, "No, I have never smoked."),
            (Speaker.CLINICIAN, "Any breathing trouble?"),
            (Speaker.PATIENT, "No wheezing at all."),
        ),
    )
    smoked = relocate(source, "smoked", evidence_id="ev-smoked")
    wheezing = relocate(source, "wheezing", evidence_id="ev-wheezing")
    assert smoked and wheezing
    gold = EncounterRecord(
        encounter_id="enc-5",
        sources=(source,),
        evidence=(smoked, wheezing),
        atoms=(
            ClinicalAtom(
                atom_id="atom-smoke",
                atom_type=AtomType.HISTORY,
                raw_value="smoked",
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.HISTORICAL),
                certainty=Certainty.STATED,
                evidence_ids=("ev-smoked",),
            ),
            ClinicalAtom(
                atom_id="atom-wheeze",
                atom_type=AtomType.SYMPTOM,
                raw_value="wheezing",
                assertion_state=AssertionState.DENIED,
                speaker=Speaker.PATIENT,
                experiencer=Experiencer.PATIENT,
                temporality=TemporalState(kind=Temporality.CURRENT),
                certainty=Certainty.STATED,
                evidence_ids=("ev-wheezing",),
            ),
        ),
    )
    specs = (
        AtomSpec(
            atom_id="atom-smoke",
            atom_type=AtomType.HISTORY,
            raw_value="smoked",
            speaker=Speaker.PATIENT,
            temporality=TemporalState(kind=Temporality.HISTORICAL),
        ),
        AtomSpec(
            atom_id="atom-wheeze",
            atom_type=AtomType.SYMPTOM,
            raw_value="wheezing",
            speaker=Speaker.PATIENT,
        ),
    )
    return HarnessCase(
        test_set=P1TestSet.P1_ADVERSARIAL,
        encounter_id="enc-5",
        gold=gold,
        model_input=ModelInput(source=source, encounter_id="enc-5"),
        atom_specs=specs,
    )


def enc1_as_core() -> HarnessCase:
    case = tiny_fixture_case()
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=case.encounter_id,
        gold=case.gold,
        model_input=case.model_input,
        atom_specs=case.atom_specs,
    )


def fixture_lines_for_encounter(encounter_id: str) -> dict[str, str]:
    """Deterministic span-port lines for campaign fixture evaluation (no weights)."""
    from nanoscribe.adapters import DEFAULT_BASELINE_LINES

    if encounter_id == "enc-1":
        return dict(DEFAULT_BASELINE_LINES)
    if encounter_id == "enc-2":
        return {"atom-chest": 'UNCERTAIN: "pressure"'}
    if encounter_id == "enc-3":
        return {
            "atom-fh": 'STATED: "diabetes"',
            "atom-tired": 'STATED: "tired"',
        }
    if encounter_id == "enc-4":
        return {
            "atom-throat": 'STATED: "sore"',
            **{atom_id: "NOT_MENTIONED" for atom_id, _t, _v in _ENC4_ABSENT},
        }
    if encounter_id == "enc-5":
        return {
            "atom-smoke": 'DENIED: "smoked"',
            "atom-wheeze": 'DENIED: "wheezing"',
        }
    raise KeyError(f"no fixture lines for encounter_id={encounter_id}")


def campaign_cases(suite: str) -> list[HarnessCase]:
    """Return harness cases for a named campaign partition."""
    key = suite.strip().lower()
    if key == "tiny_fixture":
        return [tiny_fixture_case()]
    if key == "p1_core":
        return [enc1_as_core(), enc2_uncertainty_case()]
    if key == "p1_adversarial":
        return [enc3_family_history_case()]
    if key == "campaign_v1":
        return [enc1_as_core(), enc2_uncertainty_case(), enc3_family_history_case()]
    if key == "campaign_v2":
        # Superset of campaign_v1 — one run yields both the v1 subset (which
        # reproduces the prior coverage claim) and the false-positive cases.
        return [
            enc1_as_core(),
            enc2_uncertainty_case(),
            enc3_family_history_case(),
            enc4_absent_atom_case(),
            enc5_denial_case(),
        ]
    raise ValueError(f"unknown campaign suite: {suite}")


CAMPAIGN_V1_ENCOUNTERS = ("enc-1", "enc-2", "enc-3")
CAMPAIGN_V2_ADDED_ENCOUNTERS = ("enc-4", "enc-5")


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
            "campaign_v2": list(CAMPAIGN_V1_ENCOUNTERS + CAMPAIGN_V2_ADDED_ENCOUNTERS),
        },
        "no_phi": True,
    }
