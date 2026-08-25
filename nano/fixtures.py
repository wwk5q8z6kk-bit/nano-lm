"""Synthetic, non-PHI fixtures for NANO-CLIN-001.

Every fixture is generated, contains no real patient material, and carries its
own ground truth so the benchmark is checkable. Required coverage across the
three fixtures (D-NANO-2026-08-25 §9): patient-reported material, clinician
assertions, direct measurements, speaker attribution, negation, uncertain dates,
duplicate information, conflicting diagnosis dates, medication start/stop, a
laboratory trajectory, an explicitly absent item, and an item merely not found.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    patient_id: str
    transcript: str
    prior_chart: str = ""
    # ground truth
    gold_assertions: tuple = ()      # (subject, predicate, obj, epistemic_status)
    gold_conflicts: tuple = ()       # (description,)
    gold_gaps: tuple = ()            # (expected_information, kind)
    critical_facts: tuple = ()       # facts whose omission is a critical failure
    must_not_assert: tuple = ()      # strings that must never appear as documented


BASIC = Fixture(
    fixture_id="fx_basic",
    patient_id="synth-001",
    transcript=(
        "clinician: What brings you in today?\n"
        "patient: I have had a sore throat for about three days.\n"
        "clinician: Any fever?\n"
        "patient: No fever.\n"
        "clinician: Your temperature is 37.1 degrees celsius.\n"
        "clinician: I think this is a viral pharyngitis.\n"
        "clinician: Start ibuprofen 400 mg as needed.\n"
    ),
    prior_chart="",
    gold_assertions=(
        ("patient", "reports_symptom", "sore throat", "patient_reported"),
        ("patient", "denies_symptom", "fever", "patient_reported"),
        ("patient", "has_measurement", "37.1 celsius", "direct_measurement"),
        ("clinician", "assesses", "viral pharyngitis", "clinician_asserted"),
        ("clinician", "starts_medication", "ibuprofen 400 mg", "clinician_asserted"),
    ),
    critical_facts=("sore throat", "ibuprofen"),
    # "fever" is DENIED by the patient. Rendering it as a finding is a safety bug.
    must_not_assert=("patient has fever",),
)


CONFLICTING = Fixture(
    fixture_id="fx_conflicting",
    patient_id="synth-002",
    transcript=(
        "clinician: Your notes say diabetes was diagnosed in 2016.\n"
        "patient: I think it was 2019, but I am not certain.\n"
        "clinician: Your HbA1c today is 8.2 percent.\n"
        "clinician: The previous HbA1c was 7.4 percent.\n"
    ),
    prior_chart=(
        "2018-05-02 note: No history of diabetes.\n"
        "2020-11-14 note: Type 2 diabetes, diagnosed 2016.\n"
    ),
    gold_assertions=(
        ("patient", "has_measurement", "8.2 percent", "direct_measurement"),
        ("patient", "has_measurement", "7.4 percent", "direct_measurement"),
    ),
    gold_conflicts=(
        "diabetes onset year disagrees across sources (2016 vs 2019 vs denied 2018)",
    ),
    critical_facts=("8.2 percent",),
    # The system must not silently pick one onset date.
    must_not_assert=("diabetes diagnosed 2016", "diabetes diagnosed 2019"),
)


UNCERTAIN = Fixture(
    fixture_id="fx_uncertain",
    patient_id="synth-003",
    transcript=(
        "patient: I stopped the metoprolol a while back, maybe around 2021.\n"
        "clinician: Do you recall why?\n"
        "patient: I do not remember.\n"
        "clinician: I do not see a colonoscopy in your record.\n"
        "clinician: You had no adverse reaction documented to penicillin.\n"
    ),
    prior_chart="2019-02-03 note: metoprolol 25 mg twice daily started.\n",
    gold_assertions=(
        ("patient", "stops_medication", "metoprolol", "patient_reported"),
    ),
    gold_gaps=(
        ("reason for metoprolol discontinuation", "not_found"),
        ("colonoscopy", "not_found"),
    ),
    critical_facts=("metoprolol",),
    # "around 2021" must not become an exact date; not-found must not become absent.
    must_not_assert=("stopped metoprolol on 2021-01-01",
                     "patient has never had a colonoscopy"),
)


ALL_FIXTURES = (BASIC, CONFLICTING, UNCERTAIN)


def by_id(fixture_id: str) -> Fixture:
    for f in ALL_FIXTURES:
        if f.fixture_id == fixture_id:
            return f
    raise KeyError(fixture_id)
