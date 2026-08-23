"""Artifact validation errors — machine-classifiable like AdaptError."""

from __future__ import annotations


class ArtifactError(ValueError):
    """A machine-classifiable artifact contract violation."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        self.code = code
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message} [{code}]")


def artifact_fail(code: str, message: str, path: str) -> None:
    raise ArtifactError(code, message, path=path)
