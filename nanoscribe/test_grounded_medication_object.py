"""A dose or other neighboring text cannot stand in for the named medication."""
import pytest

from nano_ai.contract import FieldState
from nanoscribe.test_grounded_substance_ownership import MED, observed


@pytest.mark.parametrize("statement,value", [
    ("I took acetaminophen 500 mg.", "mg"),
    ("I took acetaminophen 500 mg.", "500 mg"),
    ("I took acetaminophen 500 mg in water.", "water"),
    ("I took 500 mg of acetaminophen.", "mg"),
    ("I took 500 mg of acetaminophen.", "500 mg"),
    ("I took a 25-mg tablet of co-codamol.", "tablet"),
    ("I took 20 mg of sodium chloride.", "chloride"),
    ("I took 500 mg from the cabinet.", "cabinet"),
    ("I took 500 mg from the cabinet.", "from the cabinet"),
    ("I took 500 mg at bedtime.", "500 mg"),
    ("I took 500 mg orally.", "orally"),
    ("I took 500 mg each night.", "each night"),
    ("I took 500 mg total.", "total"),
    ("I took 500 mg from the bottle.", "500 mg from the bottle"),
    ("I took 500 mg of it.", "it"),
    ("I took 2 tablets.", "tablets"),
    ("I took 500 mg.", "mg"),
    ("I took 500 mg.", "500 mg"),
    ("I took two tablets of co-codamol.", "codamol"),
    ("I have taken 500 mg of acetaminophen for my pain.", "mg"),
])
def test_dosage_and_adjuncts_cannot_name_a_medication(statement, value):
    claim, direct, compact, english = observed("Patient: " + statement, MED, value)
    assert claim["value"] is None and direct is None
    assert f":S[{value}]" not in compact and f" REPORTED[{value}]" not in english


@pytest.mark.parametrize("statement,value", [
    ("I took acetaminophen 500 mg.", "acetaminophen"),
    ("I took acetaminophen 500 mg.", "acetaminophen 500 mg"),
    ("I took 500 mg of acetaminophen.", "acetaminophen"),
    ("I took 500mg acetaminophen.", "acetaminophen"),
    ("I took a 25-mg tablet of co-codamol.", "co-codamol"),
    ("I took 20 mg of sodium chloride.", "sodium chloride"),
    ("I took 0.5 mg of co‑trimoxazole.", "co‑trimoxazole"),
    ("I took two tablets of co-codamol.", "co-codamol"),
    ("I have taken 500 mg of acetaminophen for my pain.", "acetaminophen"),
    ("My daily medication is 25 micrograms of Zylomex.", "Zylomex"),
    ("I took paracetamol 500 mg.", "paracetamol"),
    ("I took Xyzalorfen-420mg.", "Xyzalorfen-420mg"),
])
def test_source_name_is_available_and_independently_verifies(statement, value):
    claim, direct, compact, english = observed("Patient: " + statement, MED, value)
    assert claim["value"] == value and direct is FieldState.SUPPORTED
    assert f":S[{value}]" in compact and f" REPORTED[{value}]" in english
