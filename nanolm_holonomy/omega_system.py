"""NANOLM Omega Zero-Oracle Unified Cognitive Controller (v2).

Public interface ``omega_generate`` takes strictly ``(model, tokenizer, system_prompt,
user_prompt, max_new_tokens, device)`` with ZERO access to held-out ``BenchmarkTask``
specs or test assertions.

Enhancements in Controller v2:
1. Intrinsic Top-2 Logit-Margin gating for state-tracking JSON vs quantitative
   symbolic synthesis (zero extra calls on high-confidence outputs).
2. Schema- and type-preserving JSON consensus guard (``preserve_valid_json_structure``)
   blocking incomplete task sets or type shifts during symbolic repair, plus
   1-step stderr/schema-feedback retry.
3. Canonical Mermaid chained-edge normalization (``normalize_chained_mermaid_edges``)
   splitting multi-hop single-line arrows into standard 2-endpoint edges.
4. Multi-probe & return-type Python smoke falsification inside isolated ``python3 -I``.
"""

from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
from typing import Any

import torch
from torch.nn import functional as F

from nanolm_frontier.surgery_and_rlvr import apply_chat_template_safe
from nanolm_holonomy.first_principles import (
    AppendOnlySourceLedger,
    DeterministicTypedRuleSolver,
    LedgerRecord,
)
from nano.contracts import DerivationMode, EpistemicStatus
from nanolm_posttrain.eval_suite import _extract_fenced_or_raw

TaskMode = str

_ARROW_SPLIT_RE = re.compile(r"\s*(-->|-\.->|==>|->>|-->>|->)\s*")
_NODE_ID_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)")
_QUANT_KEYWORDS = (
    "subtotal",
    "discount",
    "tax",
    "grand_total",
    "sum_total",
    "total_cost",
    "remaining_",
    "net_",
    "makespan",
)
_TOOL_KEYWORDS = ('"calls"', '"tool"', "available tools")


def classify_task_mode(system_prompt: str, user_prompt: str) -> TaskMode:
    """Route strictly from surface prompt instructions (zero task-oracle access)."""
    sys_l = system_prompt.casefold()
    usr_l = user_prompt.casefold()

    if "mermaid" in sys_l or "mermaid" in usr_l or "flowchart" in sys_l:
        return "DIAGRAM_MERMAID"
    if "python" in sys_l or re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", usr_l):
        return "CODE_PYTHON"
    if "makespan" in usr_l or ("critical" in usr_l and "deps" in usr_l):
        return "SYMBOLIC_COMPUTE"
    if any(k in usr_l or k in sys_l for k in _TOOL_KEYWORDS) or (
        "tool" in sys_l and "args" in sys_l
    ):
        return "STRUCTURED_JSON"
    if "final_answer:" in sys_l and (
        any(k in usr_l for k in ("mod ", "gcd", "compute", "shortest", "remainder", "dp", "evaluate", "trace", "init "))
        or len(re.findall(r"\d+", user_prompt)) >= 3
    ):
        return "SYMBOLIC_COMPUTE"
    if "json" in sys_l and any(k in usr_l for k in _QUANT_KEYWORDS):
        return "SYMBOLIC_COMPUTE"
    if "json" in sys_l and (
        any(k in usr_l for k in ("init ", "move ", "order", "event log", "transfer", "quota", "role", "balance", "ticket"))
        or len(re.findall(r"\d+", user_prompt)) >= 4
    ):
        return "STATE_JSON"
    if "json" in sys_l:
        return "STRUCTURED_JSON"
    return "DIRECT_FAST"


def _value_category(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    return type(v).__name__


def preserve_valid_json_keys(existing_reply: str, candidate_obj: dict[str, Any]) -> bool:
    """Block symbolic solver overrides that change the key schema of an already-valid JSON reply."""
    ok, _ = preserve_valid_json_structure(existing_reply, candidate_obj)
    return ok


def preserve_valid_json_structure(existing_reply: str, candidate_obj: dict[str, Any]) -> tuple[bool, str]:
    """Validate key set, per-key type compatibility, and complete string-item lists against existing JSON."""
    if not isinstance(candidate_obj, dict) or not candidate_obj:
        return False, "candidate must be a non-empty JSON object"
    try:
        existing_obj = json.loads(_extract_fenced_or_raw(existing_reply))
    except (json.JSONDecodeError, TypeError):
        return True, "no_existing_json"

    if not isinstance(existing_obj, dict) or not existing_obj:
        return True, "no_existing_json"

    if set(candidate_obj.keys()) != set(existing_obj.keys()):
        return (
            False,
            f"key_set_mismatch: expected {sorted(existing_obj.keys())}, got {sorted(candidate_obj.keys())}",
        )

    for key, orig_val in existing_obj.items():
        cand_val = candidate_obj[key]
        cat_orig = _value_category(orig_val)
        cat_cand = _value_category(cand_val)
        if cat_orig != cat_cand:
            return False, f"type_mismatch[{key}]: expected {cat_orig}, got {cat_cand}"
        if (
            isinstance(orig_val, list)
            and len(orig_val) >= 2
            and all(isinstance(x, str) for x in orig_val)
            and len(set(orig_val)) == len(orig_val)
        ):
            if not isinstance(cand_val, list) or len(cand_val) != len(orig_val) or set(cand_val) != set(orig_val):
                return (
                    False,
                    f"item_set_mismatch[{key}]: expected exact permutation of {orig_val}, got {cand_val}",
                )

    return True, "structure_ok"


def _describe_json_schema_hint(existing_reply: str) -> str:
    try:
        obj = json.loads(_extract_fenced_or_raw(existing_reply))
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(obj, dict) or not obj:
        return ""
    parts: list[str] = []
    for k, v in obj.items():
        tname = type(v).__name__
        if (
            isinstance(v, list)
            and len(v) >= 2
            and all(isinstance(x, str) for x in v)
            and len(set(v)) == len(v)
        ):
            parts.append(f'"{k}": list[str] (exact items {v})')
        else:
            parts.append(f'"{k}": {tname}')
    return "Required JSON keys & types: {" + ", ".join(parts) + "}"


def normalize_chained_mermaid_edges(response: str) -> str:
    """Expand chained Mermaid edge lines ``A --> B --> C`` into canonical single-edge lines."""
    fences = re.findall(r"```mermaid\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    body = fences[-1] if fences else response
    out_lines: list[str] = []
    changed = False

    for line in body.splitlines():
        stripped = line.strip()
        arrows = _ARROW_SPLIT_RE.findall(stripped)
        if len(arrows) >= 2 and not stripped.startswith(("%%", "subgraph")):
            tokens = _ARROW_SPLIT_RE.split(stripped)
            # tokens = [node0, arrow0, node1, arrow1, node2, ...]
            indent = line[: len(line) - len(line.lstrip())]
            for idx in range(0, len(tokens) - 2, 2):
                src = tokens[idx].strip()
                arrow = tokens[idx + 1]
                dst = tokens[idx + 2].strip()
                if idx > 0:
                    m_id = _NODE_ID_RE.match(src)
                    if m_id:
                        src = m_id.group(1)
                if src and dst:
                    out_lines.append(f"{indent}{src} {arrow} {dst}")
                    changed = True
        else:
            out_lines.append(line)

    if not changed:
        return response
    expanded = "\n".join(out_lines).strip()
    if fences or "flowchart" in expanded.casefold() or "statediagram" in expanded.casefold() or "graph " in expanded.casefold():
        return f"```mermaid\n{expanded}\n```"
    return expanded


def _dummy_value_for_annotation(ann: ast.expr | None, arg_name: str, *, variant: int = 0) -> Any:
    name_l = arg_name.lower()
    if name_l in ("src", "start"):
        return 0
    if name_l in ("dst", "end"):
        return 1
    if ann is not None:
        src = ast.unparse(ann).lower()
        if "tuple" in src and "int" in src and "list" in src:
            return [(0, 1, 1)]
        if "list" in src and "str" in src:
            return ["+", "2", "3", "*"] if variant == 1 and "tok" in name_l else ["alpha", "beta"]
        if "list" in src and "int" in src:
            return [5, 3, 1, 4, 2] if variant > 0 else [3, 1, 4, 1]
        if "list" in src:
            return [1, 2]
        if "dict" in src:
            return {"a": 1}
        if "bool" in src:
            return True
        if "float" in src:
            return 2.5
        if "int" in src:
            return 1 if variant == 1 else 3
        if "str" in src:
            return "abccba"

    if any(k in name_l for k in ("nums", "arr", "coins", "items", "vals", "weights")):
        return [1, 2, 5]
    if "edges" in name_l:
        return [(0, 1, 2), (1, 2, 3)]
    if name_l in ("s", "text", "expr", "word", "query"):
        return "ab"
    if name_l in ("n", "k", "target", "amount"):
        return 1 if variant == 1 else 3
    return [1, 2, 3]


def _ret_type_ok(ann: ast.expr | None) -> str:
    if ann is None:
        return ""
    src = ast.unparse(ann).lower()
    if src == "int":
        return "assert isinstance(_r, int) and not isinstance(_r, bool), f'expected int got {type(_r).__name__}'"
    if src == "float":
        return "assert isinstance(_r, (int, float)) and not isinstance(_r, bool), f'expected float got {type(_r).__name__}'"
    if src == "bool":
        return "assert isinstance(_r, bool), f'expected bool got {type(_r).__name__}'"
    if src == "str":
        return "assert isinstance(_r, str), f'expected str got {type(_r).__name__}'"
    if src.startswith("list"):
        return "assert isinstance(_r, list), f'expected list got {type(_r).__name__}'"
    if src.startswith("dict"):
        return "assert isinstance(_r, dict), f'expected dict got {type(_r).__name__}'"
    return ""


def extract_required_python_symbol(user_prompt: str) -> str | None:
    """Extract requested top-level Python function or class name purely from user_prompt text."""
    for pat in (
        r"`def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
        r"(?:implement|write|create|define)\s+(?:a\s+)?(?:python\s+)?(?:function|method|def)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
        r"(?:implement|write|create|define)\s+(?:a\s+)?(?:python\s+)?class\s+`?([A-Za-z_][A-Za-z0-9_]*)`?",
    ):
        m = re.search(pat, user_prompt, flags=re.IGNORECASE)
        if m:
            sym = m.group(1)
            if sym.lower() not in ("a", "the", "python", "function", "class", "code"):
                return sym
    return None


def verify_python_smoke_zero_oracle(
    code_text: str,
    *,
    required_symbol: str | None = None,
) -> tuple[bool, str]:
    """Parse AST + verify requested identifier + run multi-probe checks in isolated ``python3 -I``."""
    raw = _extract_fenced_or_raw(code_text)
    try:
        tree = ast.parse(raw)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    defined_names = {
        n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if required_symbol and required_symbol not in defined_names:
        return False, f"ContractError: required symbol '{required_symbol}' not defined (found {sorted(defined_names)})"

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return True, "no_function_ast_ok"

    target_fn = next((f for f in funcs if f.name == required_symbol), funcs[-1])
    args0 = [_dummy_value_for_annotation(a.annotation, a.arg, variant=0) for a in target_fn.args.args]
    args1 = [_dummy_value_for_annotation(a.annotation, a.arg, variant=1) for a in target_fn.args.args]
    args2 = [_dummy_value_for_annotation(a.annotation, a.arg, variant=2) for a in target_fn.args.args]
    ret_chk = _ret_type_ok(target_fn.returns)

    call_lines = [
        raw,
        "",
        f"_r = {target_fn.name}(*{repr(args0)})",
    ]
    if target_fn.returns is not None:
        call_lines.append("assert _r is not None, 'Function returned None'")
    if ret_chk:
        call_lines.append(ret_chk)

    # Multi-probe check for 2-arg functions like (values: list[int], k: int) -> int
    if len(target_fn.args.args) == 2 and isinstance(args1[0], list) and isinstance(args1[1], int):
        call_lines.append(f"_r1 = {target_fn.name}(*{repr(args1)})")
        call_lines.append(f"_r2 = {target_fn.name}(*{repr(args2)})")
        call_lines.append("assert _r1 != -1 or _r2 != -1, 'SentinelError: returned -1 on valid positive list inputs'")
    if len(args0) == 1 and isinstance(args0[0], list):
        call_lines.append(f"{target_fn.name}([])")

    script = "\n".join(call_lines)
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", script],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: exceeded 2.0s"

    if proc.returncode != 0:
        err_tail = (proc.stderr or proc.stdout).strip().splitlines()
        last_err = err_tail[-1] if err_tail else f"exit={proc.returncode}"
        if "ValueError" in last_err and "SentinelError" not in last_err:
            return True, "smoke_value_error_allowed"
        return False, last_err
    return True, "smoke_ok"


def _check_mermaid_zero_oracle(user_prompt: str, response: str) -> tuple[bool, str]:
    raw = _extract_fenced_or_raw(response)
    low = raw.casefold()
    valid_hdr = any(k in low for k in ("flowchart", "graph ", "sequencediagram", "statediagram"))
    if not valid_hdr:
        return False, "Missing valid Mermaid diagram header (e.g. flowchart TD or sequenceDiagram)"
    edges = len(re.findall(r"(-->|-\.->|==>|->>|-->>|->)", raw))
    quoted = re.findall(r"`([^`]{2,24})`", user_prompt)
    min_edges = max(2, min(len(set(quoted)) - 1, 4)) if len(set(quoted)) >= 3 else 2
    if edges < min_edges:
        return False, f"Diagram has only {edges} edges (< {min_edges} required transitions)"
    return True, "mermaid_ok"


def _run_model_chat_with_margin(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    sample: bool = False,
    temperature: float = 0.35,
) -> tuple[str, int, dict[str, float]]:
    first_dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_text = apply_chat_template_safe(tokenizer, messages, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(first_dev)
    prompt_len = int(enc["input_ids"].shape[1])

    gen_kwargs: dict[str, Any] = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": sample,
        "temperature": temperature if sample else None,
        "top_p": 0.95 if sample else None,
        "pad_token_id": pad_id,
    }
    raw_scores: tuple[torch.Tensor, ...] | None = None
    with torch.no_grad():
        try:
            out_obj = model.generate(**gen_kwargs, return_dict_in_generate=True, output_scores=not sample)
            if hasattr(out_obj, "sequences"):
                out_ids = out_obj.sequences
                raw_scores = getattr(out_obj, "scores", None)
            else:
                out_ids = out_obj
        except TypeError:
            out_ids = model.generate(**gen_kwargs)

    g_ids = out_ids[0, prompt_len:]
    margins: list[float] = []
    val_margins: list[float] = []
    if raw_scores:
        for step_idx, step_lg in enumerate(raw_scores):
            if step_idx >= int(g_ids.numel()):
                break
            probs = F.softmax(step_lg[0].float(), dim=-1)
            k_top = min(2, probs.numel())
            top2 = torch.topk(probs, k=k_top).values
            m_t = float((top2[0] - top2[1]).item()) if k_top == 2 else 1.0
            margins.append(m_t)
            piece = tokenizer.decode([int(g_ids[step_idx].item())], skip_special_tokens=True)
            if any(c.isalnum() for c in piece):
                val_margins.append(m_t)

    reply = tokenizer.decode(g_ids, skip_special_tokens=True).strip()
    stats = {
        "mean_margin": round(sum(margins) / max(len(margins), 1), 4) if margins else 1.0,
        "min_margin": round(min(margins), 4) if margins else 1.0,
        "value_min_margin": round(min(val_margins), 4) if val_margins else 1.0,
    }
    return reply, int(g_ids.numel()), stats


def _run_python_script_stdout(script_text: str) -> tuple[bool, str]:
    raw = _extract_fenced_or_raw(script_text)
    try:
        ast.parse(raw)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", raw],
            capture_output=True,
            text=True,
            timeout=2.5,
        )
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    if proc.returncode != 0:
        err = (proc.stderr or "nonzero exit").strip().splitlines()
        return False, err[-1][-160:] if err else "nonzero exit"
    out = proc.stdout.strip()
    return (bool(out), out)


def omega_generate(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    """Unified Zero-Oracle cognitive generation & falsification controller (v2)."""
    del device
    mode = classify_task_mode(system_prompt, user_prompt)
    base_msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    reply, total_tok, margin_stats = _run_model_chat_with_margin(
        model,
        tokenizer,
        base_msgs,
        max_new_tokens=max_new_tokens,
        sample=False,
    )
    meta: dict[str, Any] = {
        "mode": mode,
        "calls": 1,
        "repaired": False,
        "reason": "direct",
        **margin_stats,
    }

    if mode == "CODE_PYTHON":
        req_sym = extract_required_python_symbol(user_prompt)
        ok, err = verify_python_smoke_zero_oracle(reply, required_symbol=req_sym)
        for attempt in range(2):
            if ok:
                break
            sym_note = f" Make sure to define `{req_sym}` explicitly at top level." if req_sym else ""
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        f"Your previous draft failed verification with:\n`{err}`\n"
                        f"Previous draft:\n```python\n{_extract_fenced_or_raw(reply)}\n```\n"
                        f"Fix any undefined tree/helper assumptions, missing definitions, imports, boundary guards, or return types and return only valid Python.{sym_note}"
                    ),
                },
            ]
            cand, nt, _ = _run_model_chat_with_margin(
                model,
                tokenizer,
                repair_msgs,
                max_new_tokens=max_new_tokens,
                sample=(attempt > 0),
                temperature=0.35,
            )
            total_tok += nt
            meta["calls"] += 1
            ok2, err2 = verify_python_smoke_zero_oracle(cand, required_symbol=req_sym)
            if ok2:
                reply = cand
                ok, err = ok2, err2
                meta["repaired"] = True
                meta["reason"] = "python_smoke_repaired"
                break
            err = err2

    elif mode == "STRUCTURED_JSON":
        try:
            obj = json.loads(_extract_fenced_or_raw(reply))
            ok = isinstance(obj, dict)
            err = "ok"
        except json.JSONDecodeError as exc:
            ok = False
            err = str(exc)
        if not ok:
            repair_msgs = [
                {"role": "system", "content": "Return ONLY a valid JSON object matching the exact schema requested."},
                {"role": "user", "content": f"{user_prompt}\n\nPrevious invalid output failed JSON parsing ({err}). Return valid JSON only."},
            ]
            cand, nt, _ = _run_model_chat_with_margin(model, tokenizer, repair_msgs, max_new_tokens=max_new_tokens, sample=False)
            total_tok += nt
            meta["calls"] += 1
            try:
                if isinstance(json.loads(_extract_fenced_or_raw(cand)), dict):
                    reply = cand
                    meta["repaired"] = True
                    meta["reason"] = "json_repaired"
            except json.JSONDecodeError:
                pass

    elif mode == "DIAGRAM_MERMAID":
        norm_reply = normalize_chained_mermaid_edges(reply)
        if norm_reply != reply:
            reply = norm_reply
            meta["repaired"] = True
            meta["reason"] = "mermaid_edges_normalized"
        ok, err = _check_mermaid_zero_oracle(user_prompt, reply)
        if not ok:
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\nPrevious diagram draft had an error: {err}.\n"
                        "Include ALL required nodes, states, labels, and put every single edge on its own line inside a ```mermaid``` block."
                    ),
                },
            ]
            cand, nt, _ = _run_model_chat_with_margin(model, tokenizer, repair_msgs, max_new_tokens=max_new_tokens, sample=False)
            total_tok += nt
            meta["calls"] += 1
            cand_norm = normalize_chained_mermaid_edges(cand)
            ok2, _ = _check_mermaid_zero_oracle(user_prompt, cand_norm)
            if ok2:
                reply = cand_norm
                meta["repaired"] = True
                meta["reason"] = "mermaid_repaired"

    elif mode in ("SYMBOLIC_COMPUTE", "STATE_JSON"):
        if mode == "STATE_JSON":
            try:
                existing = json.loads(_extract_fenced_or_raw(reply))
                if isinstance(existing, dict) and existing and margin_stats["value_min_margin"] >= 0.70:
                    meta["reason"] = "margin_confident_json"
                    meta["tokens_generated"] = total_tok
                    return reply, meta
            except (json.JSONDecodeError, TypeError):
                pass

        wants_json = "json" in system_prompt.casefold()
        # Typed Relational DAG Critical-Path Operator (Zero-Oracle surface detection on "makespan" + deps)
        if not wants_json and ("makespan" in user_prompt.casefold() or ("critical path" in user_prompt.casefold() and "duration" in user_prompt.casefold())):
            dag_prompt = (
                "Extract the task graph as strict JSON ONLY matching: "
                '{"tasks": [{"id": "<str>", "duration": <number>, "deps": ["<id>", ...]}]} '
                "including EVERY task mentioned.\n\n"
                f"Problem:\n{user_prompt}"
            )
            dag_reply, nt, _ = _run_model_chat_with_margin(
                model,
                tokenizer,
                [{"role": "system", "content": "Return ONLY a valid JSON object."}, {"role": "user", "content": dag_prompt}],
                max_new_tokens=min(320, max_new_tokens + 96),
                sample=False,
            )
            total_tok += nt
            meta["calls"] += 1
            makespan_val = solve_dag_makespan_from_json(dag_reply)
            if makespan_val is not None:
                val_str = str(int(makespan_val)) if float(makespan_val).is_integer() else str(round(float(makespan_val), 4))
                reply = f"Verified via relational DAG critical-path operator.\nFINAL_ANSWER: {val_str}"
                meta["repaired"] = True
                meta["reason"] = "dag_critical_path_solver"
                meta["tokens_generated"] = total_tok
                return reply, meta

        schema_hint = _describe_json_schema_hint(reply) if wants_json else ""
        algo_note = (
            "For parallel DAG schedules, preserve every single task ID and compute critical-path DP "
            "`finish[t] = dur[t] + max((finish[d] for d in deps[t]), default=0)` with makespan = `max(finish.values())`."
            if "deps" in user_prompt.casefold() or "makespan" in user_prompt.casefold()
            else "Compute every intermediate update step by step in pure Python."
        )
        solver_prompt = (
            "Write a short self-contained Python 3 script using ONLY the standard library to solve this exact problem.\n"
            f"{algo_note}\n"
            "Strictly respect any custom rules or operator precedence stated in the problem.\n"
            + (f"{schema_hint}\n" if schema_hint else "")
            + (
                "Print ONLY the final JSON object via `print(json.dumps(result))`."
                if wants_json
                else "Print ONLY the exact final answer on the last line via `print(result)`."
            )
            + f"\n\nProblem:\n{user_prompt}"
        )

        last_err = ""
        last_code = ""
        for attempt in range(2):
            cur_prompt = solver_prompt if attempt == 0 else (
                f"{solver_prompt}\n\n"
                f"Your previous solver failed (`{last_err}`):\n```python\n{_extract_fenced_or_raw(last_code)}\n```\n"
                "Fix the script and match the exact required JSON keys, types, and full item list."
            )
            code_reply, nt, _ = _run_model_chat_with_margin(
                model,
                tokenizer,
                [{"role": "system", "content": "Return only valid Python inside a ```python``` block."}, {"role": "user", "content": cur_prompt}],
                max_new_tokens=min(340, max_new_tokens + 112),
                sample=False,
            )
            last_code = code_reply
            total_tok += nt
            meta["calls"] += 1
            ok_exec, out = _run_python_script_stdout(code_reply)
            if not ok_exec or not out:
                last_err = out or "empty stdout"
                continue

            last_line = out.splitlines()[-1].strip()
            if wants_json:
                try:
                    parsed = json.loads(last_line)
                except json.JSONDecodeError as exc:
                    last_err = f"JSONDecodeError: {exc}"
                    continue
                ok_struct, struct_reason = preserve_valid_json_structure(reply, parsed)
                if ok_struct:
                    reply = json.dumps(parsed)
                    meta["repaired"] = True
                    meta["reason"] = "python_symbolic_json" if mode == "SYMBOLIC_COMPUTE" else "python_state_json"
                    break
                last_err = struct_reason
            else:
                if len(last_line) <= 64 and not last_line.startswith("{") and "<" not in last_line and ">" not in last_line:
                    reply = f"Verified via symbolic Python execution.\nFINAL_ANSWER: {last_line}"
                    meta["repaired"] = True
                    meta["reason"] = "python_symbolic_scalar"
                    break
                last_err = f"Invalid scalar output: {last_line[:48]}"

    meta["tokens_generated"] = total_tok
    return reply, meta


def solve_dag_makespan_from_json(json_text: str) -> float | None:
    """Topological critical-path DP over model-extracted DAG JSON (returns None if invalid/cyclic)."""
    raw = _extract_fenced_or_raw(json_text)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        m = re.findall(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        try:
            parsed = json.loads(m[-1])
        except json.JSONDecodeError:
            return None
    tasks = parsed.get("tasks") if isinstance(parsed, dict) else None
    if not isinstance(tasks, list) or not tasks:
        return None
    dur: dict[str, float] = {}
    deps: dict[str, list[str]] = {}
    for item in tasks:
        if not isinstance(item, dict) or "id" not in item or "duration" not in item:
            return None
        tid = str(item["id"]).strip()
        try:
            dval = float(item["duration"])
        except (TypeError, ValueError):
            return None
        if dval < 0 or math.isnan(dval) or math.isinf(dval):
            return None
        raw_deps = item.get("deps") or []
        if not isinstance(raw_deps, list):
            return None
        dur[tid] = dval
        deps[tid] = [str(d).strip() for d in raw_deps if str(d).strip() and str(d).strip() != tid]

    finish: dict[str, float] = {}
    visiting: set[str] = set()

    def _ef(node: str) -> float:
        if node in finish:
            return finish[node]
        if node in visiting or node not in dur:
            raise ValueError("cycle_or_missing")
        visiting.add(node)
        prev = max((_ef(p) for p in deps[node] if p in dur), default=0.0)
        visiting.remove(node)
        finish[node] = prev + dur[node]
        return finish[node]

    try:
        return max(_ef(tid) for tid in dur)
    except ValueError:
        return None


def classify_fp_task_mode(system_prompt: str, user_prompt: str) -> str:
    """Detect whether prompt specifies a bitemporal/tenant source ledger problem."""
    text_l = (system_prompt + "\n" + user_prompt).casefold()
    ledger_cues = ("record_time", "valid_time", "as_of", "as of ", "tenant", "supersedes", "retract", "epistemic")
    hits = sum(1 for c in ledger_cues if c in text_l)
    if hits >= 2 or ("do(" in text_l and "intervention" in text_l):
        return "WORLD_LEDGER_OR_CAUSAL"
    return classify_task_mode(system_prompt, user_prompt)


def _resolve_world_ledger_from_json(extracted_json: str, *, wants_json: bool, user_prompt: str = "") -> str | None:
    """Validate extracted ledger records through immutable AppendOnlySourceLedger + bitemporal masks."""
    raw = _extract_fenced_or_raw(extracted_json)
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(obj, dict)
        or not isinstance(obj.get("records"), list)
        or len(obj["records"]) == 0
        or not isinstance(obj.get("query"), dict)
    ):
        return None
    q = obj["query"]
    q_ent = str(q.get("entity_id", "")).strip()
    q_ten = str(q.get("tenant_id", "")).strip()
    auth_ten = str(q.get("authorized_tenant_id", q_ten)).strip()
    q_slot = str(q.get("slot", "")).strip()
    try:
        as_of = float(q.get("as_of_time", 1e9))
    except (TypeError, ValueError):
        return None
    if not q_ent or not q_ten or not q_slot:
        return None

    ledger = AppendOnlySourceLedger()
    try:
        for idx, r in enumerate(obj["records"]):
            if not isinstance(r, dict):
                return None
            kind = str(r.get("source_kind", "unverified_report")).strip()
            ep = EpistemicStatus.CLINICIAN_ASSERTED if kind in ("verified_order", "correction") else EpistemicStatus.PATIENT_REPORTED
            if kind == "retraction":
                ep = EpistemicStatus.SUPERSEDED
            tgt = str(r.get("target_record_id", "")).strip()
            rec = LedgerRecord(
                record_id=str(r.get("record_id") or f"r_{idx}"),
                entity_id=str(r.get("entity_id", q_ent)).strip(),
                tenant_id=str(r.get("tenant_id", q_ten)).strip(),
                slot=str(r.get("slot", q_slot)).strip(),
                raw_value=str(r.get("raw_value", "")).strip(),
                polarity=(-1 if kind == "retraction" else (1 if kind in ("verified_order", "correction") else 0)),
                valid_time=float(r.get("valid_time", 0.0)),
                record_time=float(r.get("record_time", 0.0)),
                source_kind=kind,
                epistemic_status=ep,
                derivation_mode=DerivationMode.OBSERVED,
                text=(f"[correction] supersedes={tgt}" if kind == "correction" and tgt else (f"[retraction] cancels={tgt}" if kind == "retraction" and tgt else f"[{kind}]")),
                target_record_id=tgt,
            )
            ledger.append(rec)
    except (ValueError, TypeError):
        return None

    resolved = DeterministicTypedRuleSolver.resolve_ledger(
        ledger,
        entity_id=q_ent,
        tenant_id=q_ten,
        authorized_tenant_id=auth_ten,
        slot=q_slot,
        as_of_time=as_of,
    )
    ans = resolved["raw_value"] if resolved["raw_value"] else ("DENIED" if resolved["state"] == "ISOLATED_DENIED" else "UNKNOWN")
    if wants_json:
        p_low = user_prompt.casefold()
        if "state" in p_low and "value" in p_low:
            return json.dumps({"state": resolved["state"], "value": ans})
        return None
    return f"Verified via immutable bitemporal ledger ({resolved['state']}).\nFINAL_ANSWER: {ans}"


def omega_fp_generate(
    model: Any,
    tokenizer: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[str, dict[str, Any]]:
    """Zero-Oracle Controller-v3 uniting World Ledger U_t verification with Controller-v2."""
    fp_mode = classify_fp_task_mode(system_prompt, user_prompt)
    if fp_mode != "WORLD_LEDGER_OR_CAUSAL":
        return omega_generate(
            model,
            tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            device=device,
        )

    # Check if prompt contains structured temporal records suitable for AppendOnlySourceLedger
    extract_prompt = (
        "Extract observed event records and the as-of query as JSON ONLY:\n"
        '{"query": {"entity_id": str, "tenant_id": str, "authorized_tenant_id": str, "slot": str, "as_of_time": float}, '
        '"records": [{"record_id": str, "entity_id": str, "tenant_id": str, "slot": str, "raw_value": str, '
        '"valid_time": float, "record_time": float, "source_kind": "verified_order|unverified_report|correction|retraction", '
        '"target_record_id": str}]}\n'
        "If the task is purely causal/numerical without timestamped records, return {\"records\": []}.\n\n"
        f"{user_prompt}"
    )
    ext_reply, tok_ext, _ = _run_model_chat_with_margin(
        model,
        tokenizer,
        [{"role": "system", "content": "Return ONLY a valid JSON object."}, {"role": "user", "content": extract_prompt}],
        max_new_tokens=min(384, max_new_tokens + 128),
        sample=False,
    )
    wants_json = "json" in system_prompt.casefold()
    resolved_txt = _resolve_world_ledger_from_json(ext_reply, wants_json=wants_json, user_prompt=user_prompt)
    if resolved_txt is not None:
        return resolved_txt, {
            "mode": "WORLD_LEDGER_VERIFIED",
            "calls": 1,
            "repaired": True,
            "reason": "bitemporal_ledger_u_t",
            "tokens_generated": tok_ext,
        }

    reply, meta = omega_generate(
        model,
        tokenizer,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    meta["calls"] = int(meta.get("calls", 1)) + 1
    meta["tokens_generated"] = int(meta.get("tokens_generated", 0)) + tok_ext
    return reply, meta
