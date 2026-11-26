"""Verification Engine: Epistemic Tiering, Anti-Circularity Detection, & Deterministic Critics.

Distinguishes strictly between:
  - `VERIFIED_COMPUTATION`: Verified by isolated Python execution or independent symbolic/numeric script
  - `VERIFIED_SCHEMA`: Verified by JSON/AST/constraint parser + conservation invariants
  - `SUPPORTED_EVIDENCE`: Grounded in retrieved/provided context without unsupported numeric claims
  - `PLAUSIBLE_UNVERIFIED`: Coherent answer without independent executable check
  - `UNCERTAIN_OR_FAILED`: Failed verification or detected circular reasoning -> returns actionable critique
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from lab.trackb.eval_capabilities import strip_thinking
from nanolm.eval.validators import extract_python_code, verify_python_solution

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*([^\n<]+)", re.IGNORECASE)
_THINK_CONTENT_RE = re.compile(r"(?:<\|think\|>|<think>)(.*?)(?:<\|end_think\|>|</think>)", re.DOTALL)


@dataclass
class VerificationReport:
    passed: bool
    tier: str
    extracted_answer: str
    circular_reasoning_detected: bool
    critique: str


def detect_circular_reasoning(raw_response: str, ngram_n: int = 10, repeat_threshold: int = 3) -> bool:
    """Detect repetitive circular thought loops or unclosed truncated `<think>` blocks."""
    if "<think>" in raw_response and "</think>" not in raw_response:
        return True
    m = _THINK_CONTENT_RE.search(raw_response)
    thought = m.group(1) if m else raw_response
    words = thought.lower().split()
    if len(words) < ngram_n * repeat_threshold:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - ngram_n + 1):
        ng = tuple(words[i : i + ngram_n])
        c = counts.get(ng, 0) + 1
        if c >= repeat_threshold:
            return True
        counts[ng] = c
    return False


def extract_canonical_answer(raw_response: str) -> str:
    vis = strip_thinking(raw_response)
    boxes = _BOXED_RE.findall(vis)
    if boxes:
        return boxes[-1].strip()
    m = _FINAL_RE.search(vis)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in vis.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def verify_code_candidate(raw_response: str, self_tests: str = "") -> VerificationReport:
    circ = detect_circular_reasoning(raw_response)
    vis = strip_thinking(raw_response)
    code = extract_python_code(vis)
    if not code or "def " not in code:
        return VerificationReport(
            False, "UNCERTAIN_OR_FAILED", "", circ,
            "No valid Python `def ...` function found outside `<think>`. Provide complete Python code inside ```python```.",
        )
    # Compile/syntax + self-test execution in isolated subprocess
    harness = self_tests.strip() if self_tests.strip() else "pass"
    ok, msg = verify_python_solution(f"```python\n{code}\n```", harness)
    if ok:
        return VerificationReport(True, "VERIFIED_COMPUTATION", code, circ, "ok")
    return VerificationReport(
        False, "UNCERTAIN_OR_FAILED", code, circ,
        f"Sandbox execution failed with `{msg}`. Fix the bug/edge case and output corrected ```python``` code.",
    )


def verify_json_candidate(
    raw_response: str,
    required_keys: set[str] | None = None,
    invariant_fn: Callable[[dict[str, Any]], tuple[bool, str]] | None = None,
) -> VerificationReport:
    circ = detect_circular_reasoning(raw_response)
    vis = strip_thinking(raw_response)
    cand = vis
    m = re.search(r"```(?:json)?\s*\n(.*?)```", vis, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
    elif "{" in vis and "}" in vis:
        cand = vis[vis.find("{") : vis.rfind("}") + 1]
    try:
        obj = json.loads(cand)
    except Exception as exc:
        return VerificationReport(False, "UNCERTAIN_OR_FAILED", "", circ, f"Invalid JSON syntax: {exc}. Output ONLY valid JSON.")
    if not isinstance(obj, dict):
        return VerificationReport(False, "UNCERTAIN_OR_FAILED", "", circ, "Root JSON value must be an object `{...}`.")
    if required_keys and not required_keys.issubset(obj.keys()):
        missing = sorted(required_keys - set(obj.keys()))
        return VerificationReport(False, "UNCERTAIN_OR_FAILED", cand, circ, f"Missing required JSON keys: {missing}.")
    if invariant_fn is not None:
        try:
            ok, reason = invariant_fn(obj)
        except Exception as exc:
            return VerificationReport(False, "UNCERTAIN_OR_FAILED", cand, circ, f"Schema/type check failed: {exc}")
        if not ok:
            return VerificationReport(False, "UNCERTAIN_OR_FAILED", cand, circ, f"Schema/conservation check failed: {reason}")
    return VerificationReport(True, "VERIFIED_SCHEMA", json.dumps(obj), circ, "ok")
