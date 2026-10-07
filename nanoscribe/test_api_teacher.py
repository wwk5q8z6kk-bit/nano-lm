import sys
import pytest

from nanoscribe.api_teacher import _openai_client

def test_openai_client_import_error(monkeypatch):
    # Force an import error by masking 'openai' module
    monkeypatch.setitem(sys.modules, "openai", None)

    with pytest.raises(RuntimeError, match="(?i).*pip install openai.*"):
        _openai_client()
