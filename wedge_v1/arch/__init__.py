"""Architecture registry, failure codes, and structured traces."""
from wedge_v1.arch.failure_codes import FailureCode
from wedge_v1.arch.registry import registry_snapshot
from wedge_v1.arch.trace import AskTrace

__all__ = ["AskTrace", "FailureCode", "registry_snapshot"]
