"""Programmatic validators for Code Execution, Math/Logic, Summarization, JSON, Charts, and Mermaid Diagrams."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_MERMAID_FENCE_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FINAL_ANS_RE = re.compile(r"FINAL_ANSWER\s*:\s*([^\n<]+)", re.IGNORECASE)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MERMAID_EDGE_RE = re.compile(
    r'([A-Za-z0-9_]+)(?:\["([^"]+)"\]|\[([^\]]+)\])?\s*-->\s*(?:\|"([^"]+)"\||\|([^|]+)\|)?\s*([A-Za-z0-9_]+)(?:\["([^"]+)"\]|\[([^\]]+)\])?'
)


def extract_python_code(response: str) -> str:
    stripped = re.sub(r"<\|think\|>.*?<\|end_think\|>", "", response, flags=re.DOTALL).strip()
    m = _CODE_FENCE_RE.search(stripped)
    if m:
        return m.group(1).strip()
    return stripped


def verify_python_solution(response: str, test_harness: str, timeout_sec: float = 3.0) -> tuple[bool, str]:
    """Execute candidate Python code against held-out unit tests in an isolated subprocess."""
    code = extract_python_code(response)
    if not code or "def " not in code:
        return False, "no_function_definition"
    script = f"{code}\n\n{test_harness}\nprint('__ALL_TESTS_PASSED__')\n"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "timeout"
    if proc.returncode == 0 and "__ALL_TESTS_PASSED__" in proc.stdout:
        return True, "passed"
    err = (proc.stderr or proc.stdout).strip().splitlines()
    return False, (err[-1][:120] if err else f"exit_{proc.returncode}")


def verify_math_or_logic(response: str, expected_answer: str) -> bool:
    """Verify exact numeric or symbolic answer from FINAL_ANSWER marker or final line."""
    stripped = re.sub(r"<\|think\|>.*?<\|end_think\|>", "", response, flags=re.DOTALL).strip()
    m = _FINAL_ANS_RE.search(stripped)
    candidate = m.group(1).strip() if m else (stripped.splitlines()[-1].strip() if stripped else "")
    exp_clean = expected_answer.strip().lower().rstrip(".")
    cand_clean = candidate.strip().lower().rstrip(".")
    if cand_clean == exp_clean:
        return True
    try:
        exp_val = float(exp_clean.replace(",", ""))
        nums = _NUM_RE.findall(cand_clean.replace(",", ""))
        if nums and abs(float(nums[-1]) - exp_val) < 1e-6:
            return True
    except ValueError:
        pass
    return False


def verify_summary_faithfulness(
    response: str,
    required_facts: list[str],
    required_qualifications: list[str],
    forbidden_claims: list[str],
    max_words: int = 55,
) -> dict[str, Any]:
    """Score information retention, qualification preservation, and absence of unsupported claims."""
    stripped = re.sub(r"<\|think\|>.*?<\|end_think\|>", "", response, flags=re.DOTALL).strip()
    lower = stripped.lower()
    words = stripped.split()
    if not words:
        return {"passed": False, "fact_recall": 0.0, "qualification_preserved": False, "hallucinated": False}

    fact_hits = sum(1 for f in required_facts if f.lower() in lower)
    fact_recall = fact_hits / max(1, len(required_facts))
    qual_ok = all(q.lower() in lower for q in required_qualifications)
    hallucinated = any(bad.lower() in lower for bad in forbidden_claims)
    concise = len(words) <= max_words
    passed = (fact_recall == 1.0) and qual_ok and (not hallucinated) and concise
    return {
        "passed": passed,
        "fact_recall": round(fact_recall, 4),
        "qualification_preserved": qual_ok,
        "hallucinated": hallucinated,
        "concise": concise,
    }


def verify_structured_json(
    response: str,
    required_fields: dict[str, Any],
    semantic_predicate: Any,
) -> tuple[bool, bool, str]:
    """Return (valid_json, schema_and_semantics_ok, reason)."""
    stripped = re.sub(r"<\|think\|>.*?<\|end_think\|>", "", response, flags=re.DOTALL)
    stripped = stripped.replace("<|json|>", "").strip()
    m = _JSON_FENCE_RE.search(stripped)
    raw_json = m.group(1).strip() if m else stripped
    if not raw_json.startswith("{") and "{" in raw_json and "}" in raw_json:
        raw_json = raw_json[raw_json.find("{") : raw_json.rfind("}") + 1]
    try:
        obj = json.loads(raw_json)
    except Exception:
        return False, False, "invalid_json"
    if not isinstance(obj, dict):
        return True, False, "not_json_object"
    for key, typ in required_fields.items():
        if key not in obj or not isinstance(obj[key], typ):
            return True, False, f"field_mismatch:{key}"
    try:
        sem_ok = bool(semantic_predicate(obj))
    except Exception:
        sem_ok = False
    return True, sem_ok, ("passed" if sem_ok else "semantic_check_failed")


def verify_mermaid_flowchart(
    response: str,
    expected_edges: list[tuple[str, str, str | None]],
) -> tuple[bool, bool, str]:
    """Validate Mermaid flowchart syntax AND semantic edge directionality & labels."""
    stripped = re.sub(r"<\|think\|>.*?<\|end_think\|>", "", response, flags=re.DOTALL)
    stripped = stripped.replace("<|diagram|>", "").strip()
    m = _MERMAID_FENCE_RE.search(stripped)
    body = m.group(1).strip() if m else stripped

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines or not (lines[0].startswith("flowchart ") or lines[0].startswith("graph ")):
        return False, False, "missing_flowchart_header"

    node_labels: dict[str, str] = {}
    parsed_edges: list[tuple[str, str, str]] = []
    for ln in lines[1:]:
        for match in _MERMAID_EDGE_RE.finditer(ln):
            src_id, s_q, s_u, e_q, e_u, dst_id, d_q, d_u = match.groups()
            if s_q or s_u:
                node_labels[src_id] = (s_q or s_u).strip()
            if d_q or d_u:
                node_labels[dst_id] = (d_q or d_u).strip()
            src_name = node_labels.get(src_id, src_id).lower()
            dst_name = node_labels.get(dst_id, dst_id).lower()
            edge_lbl = ((e_q or e_u) or "").strip().lower()
            parsed_edges.append((src_name, dst_name, edge_lbl))

    if not parsed_edges:
        return False, False, "no_valid_edges"

    for exp_src, exp_dst, exp_lbl in expected_edges:
        es = exp_src.lower()
        ed = exp_dst.lower()
        el = (exp_lbl or "").lower()
        found = any(
            (es in ps and ed in pd and (not el or el in pl))
            for ps, pd, pl in parsed_edges
        )
        if not found:
            return True, False, f"missing_edge:{exp_src}->{exp_dst}"
    return True, True, "passed"


def verify_chart_spec(
    response: str,
    allowed_chart_types: set[str],
    expected_x: list[Any],
    expected_y: list[float],
) -> tuple[bool, bool, str]:
    """Validate structured chart specification: chart selection, axis naming, and exact series values."""
    def _pred(obj: dict[str, Any]) -> bool:
        ctype = str(obj.get("chart_type", "")).lower()
        if ctype not in allowed_chart_types:
            return False
        if not str(obj.get("x_label", "")).strip() or not str(obj.get("y_label", "")).strip():
            return False
        xs = obj.get("x_values", [])
        ys = obj.get("y_values", [])
        if list(xs) != list(expected_x) or len(ys) != len(expected_y):
            return False
        return all(abs(float(a) - float(b)) < 1e-4 for a, b in zip(ys, expected_y))

    return verify_structured_json(
        response,
        required_fields={
            "chart_type": str,
            "x_label": str,
            "y_label": str,
            "x_values": list,
            "y_values": list,
        },
        semantic_predicate=_pred,
    )
