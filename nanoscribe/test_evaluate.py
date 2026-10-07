import pytest
from nanoscribe.encounter import EncounterError
from nanoscribe.evaluate import _classify_construction, _INVALID_SPAN_CODES

@pytest.mark.parametrize("code", ["unknown_source"] + list(_INVALID_SPAN_CODES))
def test_classify_construction_malformed_and_critical(code):
    error = EncounterError(code=code, message="test")
    assert _classify_construction(error) == (True, True)

@pytest.mark.parametrize("code", ["unknown_evidence", "duplicate_id"])
def test_classify_construction_malformed_and_critical_explicit(code):
    error = EncounterError(code=code, message="test")
    assert _classify_construction(error) == (True, True)

@pytest.mark.parametrize("code", ["type_error", "missing_field"])
def test_classify_construction_malformed_and_critical_type_error(code):
    error = EncounterError(code=code, message="test")
    assert _classify_construction(error) == (True, True)

@pytest.mark.parametrize("code", ["some_other_code", "another_unknown_code"])
def test_classify_construction_malformed_not_critical(code):
    error = EncounterError(code=code, message="test")
    assert _classify_construction(error) == (True, False)
