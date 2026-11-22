import pytest
from unittest.mock import Mock, patch
from nanoscribe.evaluate import _probe_construction, PredictedAtom
from nanoscribe.encounter import EncounterError

@pytest.fixture
def base_probe_mocks():
    """Returns (gold, pred, spans) initialized with dummy data to pass _prediction_complete."""
    gold = Mock()
    gold.sources = []

    pred = Mock(spec=PredictedAtom)
    pred.atom_type = "invalid_type"
    pred.raw_value = "raw"
    pred.assertion_state = "invalid_state"
    pred.speaker = "invalid_speaker"
    pred.experiencer = "invalid_experiencer"
    pred.temporality = "invalid_temporality"
    pred.certainty = "invalid_certainty"
    pred.evidence_ids = ("ev1",)
    pred.normalized_value = None
    pred.normalization_transform = None
    pred.review_required = False
    pred.atom_id = "valid_string"

    spans = []
    return gold, pred, spans

def test_probe_construction_catches_value_error(base_probe_mocks):
    gold, pred, spans = base_probe_mocks

    with patch("nanoscribe.evaluate.ClinicalAtom") as mock_clinical_atom:
        mock_clinical_atom.side_effect = ValueError("Mocked ValueError")
        error = _probe_construction(gold, pred, spans)
        assert isinstance(error, EncounterError)
        assert error.code == "type_error"
        assert "Mocked ValueError" in str(error)

def test_probe_construction_catches_type_error(base_probe_mocks):
    gold, pred, spans = base_probe_mocks

    with patch("nanoscribe.evaluate.ClinicalAtom") as mock_clinical_atom:
        mock_clinical_atom.side_effect = TypeError("Mocked TypeError")
        error = _probe_construction(gold, pred, spans)
        assert isinstance(error, EncounterError)
        assert error.code == "type_error"
        assert "Mocked TypeError" in str(error)

def test_probe_construction_catches_attribute_error(base_probe_mocks):
    gold, pred, spans = base_probe_mocks

    with patch("nanoscribe.evaluate.ClinicalAtom") as mock_clinical_atom:
        mock_clinical_atom.side_effect = AttributeError("Mocked AttributeError")
        error = _probe_construction(gold, pred, spans)
        assert isinstance(error, EncounterError)
        assert error.code == "type_error"
        assert "Mocked AttributeError" in str(error)

def test_probe_construction_catches_encounter_error(base_probe_mocks):
    gold, pred, spans = base_probe_mocks

    with patch("nanoscribe.evaluate.ClinicalAtom") as mock_clinical_atom:
        mock_clinical_atom.side_effect = EncounterError("custom_error_code", "Mocked EncounterError")
        error = _probe_construction(gold, pred, spans)
        assert isinstance(error, EncounterError)
        assert error.code == "custom_error_code"
        assert "Mocked EncounterError" in str(error)

def test_probe_construction_passes_when_no_exception(base_probe_mocks):
    gold, pred, spans = base_probe_mocks

    with patch("nanoscribe.evaluate.ClinicalAtom"), patch("nanoscribe.evaluate.EncounterRecord"):
        error = _probe_construction(gold, pred, spans)
        assert error is None
