"""Identity regression checks for the internal Wedge v1 command surface."""

from __future__ import annotations

import pytest

from wedge_v1.cli import main


def test_cli_help_identifies_internal_wedge_not_nano_ai(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        main(["--help"])

    assert raised.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Wedge v1 internal evidence and validation pipeline" in help_text
    assert "not the Nano AI or its inference entry point" in help_text
    assert "Nano Runtime" not in help_text
