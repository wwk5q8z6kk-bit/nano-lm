"""Versioned P1 campaign evaluation suites (no PHI)."""

from __future__ import annotations

from nanoscribe.adapt import ModelInput
from nanoscribe.adapters import AtomSpec, default_baseline_specs
from nanoscribe.dataset_leakage import DatasetPartition, assert_no_leakage
from nanoscribe.distill_train_suite import (
    DISTILL_TRAIN_REVISION,
    distill_train_cases,
    distill_train_manifest,
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
    assemble_source,
)
from nanoscribe.harness import HarnessCase, P1TestSet
from nanoscribe.select import relocate
from nanoscribe.tracks import tiny_fixture_case

from nanoscribe.screening_suite import (
    SCREENING_REVISION,
    SCREENING_SUITE_NAME,
    screening_adversarial_cases,
    screening_core_cases,
    screening_manifest,
)

CAMPAIGN_DATASET_REVISION = "campaign_v2_20260823"
SMOKE_SUITE_REVISION = "p1_contract_smoke_v1"


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


def enc1_as_core() -> HarnessCase:
    case = tiny_fixture_case()
    return HarnessCase(
        test_set=P1TestSet.P1_CORE,
        encounter_id=case.encounter_id,
        gold=case.gold,
        model_input=case.model_input,
        atom_specs=case.atom_specs,
    )


def smoke_contract_cases() -> list[HarnessCase]:
    """3-case P1 smoke contract — enc-1 core + uncertainty + adversarial."""
    enc1 = enc1_as_core()
    enc2 = enc2_uncertainty_case()
    enc3 = enc3_family_history_case()
    return [
        HarnessCase(
            test_set=P1TestSet.P1_SMOKE_CONTRACT_SUITE,
            encounter_id=enc1.encounter_id,
            gold=enc1.gold,
            model_input=enc1.model_input,
            atom_specs=enc1.atom_specs,
        ),
        HarnessCase(
            test_set=P1TestSet.P1_SMOKE_CONTRACT_SUITE,
            encounter_id=enc2.encounter_id,
            gold=enc2.gold,
            model_input=enc2.model_input,
            atom_specs=enc2.atom_specs,
        ),
        HarnessCase(
            test_set=P1TestSet.P1_SMOKE_CONTRACT_SUITE,
            encounter_id=enc3.encounter_id,
            gold=enc3.gold,
            model_input=enc3.model_input,
            atom_specs=enc3.atom_specs,
        ),
    ]


def campaign_cases(suite: str) -> list[HarnessCase]:
    """Return harness cases for a named campaign partition."""
    key = suite.strip().lower()
    if key == "tiny_fixture":
        return [tiny_fixture_case()]
    if key == "p1_core":
        return [enc1_as_core(), enc2_uncertainty_case()]
    if key == "p1_adversarial":
        return [enc3_family_history_case()]
    if key in {"campaign_v1", "p1_contract_smoke_v1", "campaign_smoke", "p1_smoke_contract_suite"}:
        return smoke_contract_cases()
    if key in {"p1_screening_eval_v1"}:
        return screening_core_cases() + screening_adversarial_cases()
    if key == "screening_p1_core":
        return screening_core_cases()
    if key == "screening_p1_adversarial":
        return screening_adversarial_cases()
    if key in {"screening_p1", "c2_screening", "c2", "campaign_c2"}:
        return screening_core_cases() + screening_adversarial_cases()
    if key in {"p1_distill_train_v1", "distill_train"}:
        return distill_train_cases()
    if key in {"c1_canary", "c1", "campaign_c1"}:
        return (screening_core_cases() + screening_adversarial_cases())[:32]
    raise ValueError(f"unknown campaign suite: {suite}")


def partition_cases(partition: DatasetPartition) -> list[HarnessCase]:
    if partition == DatasetPartition.FROZEN_SCREENING_EVAL:
        return campaign_cases("p1_screening_eval_v1")
    if partition == DatasetPartition.TRAIN:
        return campaign_cases("p1_distill_train_v1")
    if partition == DatasetPartition.DEV:
        train = distill_train_cases()
        return [case for index, case in enumerate(train) if index % 8 == 0]
    if partition == DatasetPartition.INTERNAL_TEST:
        train = distill_train_cases()
        return [case for index, case in enumerate(train) if index % 8 == 4]
    raise ValueError(f"unknown partition: {partition}")


def validate_campaign_partitions() -> None:
    partitions = {
        DatasetPartition.TRAIN: partition_cases(DatasetPartition.TRAIN),
        DatasetPartition.DEV: partition_cases(DatasetPartition.DEV),
        DatasetPartition.INTERNAL_TEST: partition_cases(DatasetPartition.INTERNAL_TEST),
        DatasetPartition.FROZEN_SCREENING_EVAL: partition_cases(DatasetPartition.FROZEN_SCREENING_EVAL),
    }
    assert_no_leakage(partitions)


def suite_manifest() -> dict[str, object]:
    return {
        "schema": "nano.campaign.dataset.v1",
        "revision": CAMPAIGN_DATASET_REVISION,
        "partitions": {
            "TRAIN": distill_train_manifest()["encounter_ids"][:8],
            "DEV": "every 8th distill train case",
            "INTERNAL_TEST": "every 8th distill train case offset 4",
            "FROZEN_SCREENING_EVAL": screening_manifest()["P1_CORE"][:8],
            "P1_CORE": ["enc-1", "enc-2"],
            "P1_ADVERSARIAL": ["enc-3"],
            "tiny_fixture": ["enc-1"],
            "p1_contract_smoke_v1": ["enc-1", "enc-2", "enc-3"],
            "campaign_v1": ["enc-1", "enc-2", "enc-3"],
            "p1_screening_eval_v1": screening_manifest()["P1_CORE"],
            "p1_distill_train_v1": distill_train_manifest()["encounter_ids"][:8],
            "c1_canary": "first 32 of p1_screening_eval_v1",
            "c2_screening": "full p1_screening_eval_v1 (128 cases)",
        },
        "smoke_suite": SMOKE_SUITE_REVISION,
        "screening_suite": SCREENING_SUITE_NAME,
        "screening_revision": SCREENING_REVISION,
        "distill_train_revision": DISTILL_TRAIN_REVISION,
        "no_phi": True,
    }
