"""Capability-oriented structured generation for Nano Agent."""

from nanoscribe.capabilities.parser import CapabilityToolParser, ToolResult
from nanoscribe.capabilities.registry import (
    CapabilityId,
    CapabilitySpec,
    CapabilityStatus,
    CAPABILITY_REGISTRY,
    capability_for_tool,
    get_capability,
    list_capabilities,
)

__all__ = [
    "CapabilityId",
    "CapabilitySpec",
    "CapabilityStatus",
    "CAPABILITY_REGISTRY",
    "CapabilityToolParser",
    "ToolResult",
    "capability_for_tool",
    "get_capability",
    "list_capabilities",
]
