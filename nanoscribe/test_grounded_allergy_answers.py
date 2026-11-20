"""Ground bare allergen answers in the question's polarity and the complete source list."""
import pytest

from nano_ai.contract import FieldState
from nanoscribe.grounded import parse_grounded_summary
from nanoscribe.test_grounded_substance_ownership import ALG, observed


@pytest.mark.parametrize("question,answer,names", [
    ("Any allergies?", "Penicillin and latex.", ("Penicillin", "latex")),
    ("What substances are you allergic to?", "Co-trimoxazole and red cedar pollen.", ("Co-trimoxazole", "red cedar pollen")),
    ("Any allergies?", "Latex plus bee venom plus nickel.", ("Latex", "bee venom", "nickel")),
    ("Do you have any allergies?", "Penicillin, latex and bee venom.", ("Penicillin", "latex", "bee venom")),
    ("Any allergies?", "Penicillin (skin rash and itching) and latex.", ("Penicillin", "latex")),
    ("Which drugs are you not allergic to?", "I am allergic to penicillin and latex.", ("penicillin", "latex")),
])
def test_coordinated_objects_preserve_every_allergen_in_scalar_uncertainty(question, answer, names):
    report = f"Doctor: {question}\nPatient: {answer}"
    for name in names:
        claim, direct, compact, english = observed(report, ALG, name)
        assert claim["state"] == "uncertain" and claim["value"] is None
        assert len(claim["evidence_ids"]) == len(names)
        assert direct is None
        assert not any(value.startswith(":S[") for value in compact)
        assert not any(code == "S" for code, _ in english.values())
        assert names in [tuple(s.text for s in spans) for code, spans in english.values() if code == "U"]
        for option in compact:
            if option.startswith(":U["):
                _, props, _, error = parse_grounded_summary("CC:M|DUR:M|SEV:M|MED:M|ALG" + option, report)
                assert error is None
                if tuple(s.text for s in props[-1].spans) == names:
                    break
        else:
            pytest.fail("the compact menu omitted the complete coordinated allergy evidence")


@pytest.mark.parametrize("question,answer,value,supported", [
    ("Which drugs are you not allergic to?", "Penicillin.", "Penicillin", False),
    ("Which drugs aren’t you allergic to?", "Penicillin and latex.", "Penicillin", False),
    ("Which substances do not cause you any allergies?", "Latex.", "Latex", False),
    ("Which medicines cause no allergic reaction?", "Penicillin.", "Penicillin", False),
    ("Which allergies do you not have?", "Latex.", "Latex", False),
    ("Your chart lists several allergies. Which do you not have?", "Latex.", "Latex", False),
    ("Do you have any allergies? If so, which ones?", "Latex.", "Latex", True),
    ("Which drugs are you not allergic to?", "I am allergic to latex.", "latex", True),
    ("Which drugs are you allergic to?", "Penicillin.", "Penicillin", True),
    ("I don't mean your medicines. Any allergies?", "Latex.", "Latex", True),
    ("Which supplies did you buy?", "Penicillin and latex.", "latex", False),
    ("Any allergies?", "My brother is allergic to penicillin and latex.", "latex", False),
    ("Any allergies?", "Penicillin and then I went home.", "Penicillin", True),
    ("Any allergies?", "Penicillin (itching and skin rash).", "Penicillin", True),
    ("Any allergies?", "Penicillin (itching and skin rash).", "skin", False),
    ("Any allergies?", "Penicillin with a rash.", "Penicillin", True),
])
def test_bare_allergen_requires_positive_question_or_its_own_positive_relation(question, answer, value, supported):
    report = f"Doctor: {question}\nPatient: {answer}"
    claim, direct, compact, english = observed(report, ALG, value)
    assert (claim["state"] == "supported") == supported
    assert (direct is FieldState.SUPPORTED) == supported
    assert (f":S[{value}]" in compact) == supported
    assert (f" REPORTED[{value}]" in english) == supported
    assert not any(option.startswith(":A[") for option in compact)
    assert not any(code == "A" for code, _ in english.values())
    assert not any(code == "U" and len(spans) > 1 for code, spans in english.values())
