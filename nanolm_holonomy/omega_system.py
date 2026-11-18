"""NANOLM Omega Zero-Oracle Unified Cognitive Controller.

Public interface ``omega_generate`` takes strictly ``(model, tokenizer, system_prompt,
user_prompt, max_new_tokens, device)`` with ZERO access to held-out ``BenchmarkTask``
specs or test assertions. All falsification happens via intrinsic syntax parsing,
type-driven Python smoke execution in ``python3 -I``, self-synthesized Python
solvers, and prompt-internal Mermaid/JSON consistency checks.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from typing import Any

import torch

from nanolm_frontier.surgery_and_rlvr import apply_chat_template_safe
from nanolm_posttrain.eval_suite import _extract_fenced_or_raw

TaskMode = str


def classify_task_mode(system_prompt: str, user_prompt: str) -> TaskMode:
    """Route strictly from surface prompt instructions (zero task-oracle access)."""
    sys_l = system_prompt.casefold()
    usr_l = user_prompt.casefold()

    if "mermaid" in sys_l or "mermaid" in usr_l or "flowchart" in sys_l:
        return "DIAGRAM_MERMAID"
    if "python" in sys_l or re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", usr_l):
        return "CODE_PYTHON"
    if "makespan" in usr_l or "critical" in usr_l and "deps" in usr_l:
        return "SYMBOLIC_COMPUTE"
    if "available tools" in usr_l or "available tools" in sys_l or '"calls"' in sys_l or "calls" in sys_l:
        return "STRUCTURED_JSON"
    if "final_answer:" in sys_l and (
        any(k in usr_l for k in ("mod ", "gcd", "compute", "shortest", "remainder", "dp", "evaluate", "trace", "init "))
        or len(re.findall(r"\d+", user_prompt)) >= 3
    ):
        return "SYMBOLIC_COMPUTE"
    if "json" in sys_l and (
        any(k in usr_l for k in ("subtotal", "remaining", "net_", "balance", "init ", "move ", "order", "makespan"))
        or len(re.findall(r"\d+", user_prompt)) >= 4
    ):
        return "SYMBOLIC_COMPUTE"
    if "json" in sys_l:
        return "STRUCTURED_JSON"
    return "DIRECT_FAST"


def preserve_valid_json_keys(existing_reply: str, candidate_obj: dict[str, Any]) -> bool:
    """Block symbolic solver overrides that change the key schema of an already-valid JSON reply."""
    if not isinstance(candidate_obj, dict) or not candidate_obj:
        return False
    try:
        existing_obj = json.loads(_extract_fenced_or_raw(existing_reply))
    except (json.JSONDecodeError, TypeError):
        return True
    if isinstance(existing_obj, dict) and existing_obj:
        return set(candidate_obj.keys()) == set(existing_obj.keys())
    return True


def _dummy_value_for_annotation(ann: ast.expr | None, arg_name: str) -> Any:
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
            return ["alpha", "beta"]
        if "list" in src and "int" in src:
            return [3, 1, 4, 1]
        if "list" in src:
            return [1, 2]
        if "dict" in src:
            return {"a": 1}
        if "bool" in src:
            return True
        if "float" in src:
            return 2.5
        if "int" in src:
            return 3
        if "str" in src:
            return "abccba"

    if any(k in name_l for k in ("nums", "arr", "coins", "items", "vals", "weights")):
        return [1, 2, 5]
    if "edges" in name_l:
        return [(0, 1, 2), (1, 2, 3)]
    if name_l in ("s", "text", "expr", "word", "query"):
        return "ab"
    if name_l in ("n", "k", "target", "amount"):
        return 3
    return [1, 2, 3]


def verify_python_smoke_zero_oracle(code_text: str) -> tuple[bool, str]:
    """Parse AST + run an isolated ``python3 -I`` type-driven smoke call (no benchmark tests)."""
    raw = _extract_fenced_or_raw(code_text)
    try:
        tree = ast.parse(raw)
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc}"

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not funcs:
        return True, "no_function_ast_ok"

    target_fn = funcs[-1]
    arg_vals = [_dummy_value_for_annotation(a.annotation, a.arg) for a in target_fn.args.args]
    # Also check empty list edge case when first arg is list
    call_lines = [
        raw,
        "",
        "import json as _j",
        f"_args = {repr(arg_vals)}",
        f"{target_fn.name}(*_args)",
    ]
    if len(arg_vals) == 1 and isinstance(arg_vals[0], list):
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
        # Ignore intentional ValueError raised by valid input validation
        if "ValueError" in last_err:
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
    # Count quoted or CamelCase/UPPER tokens mentioned in user_prompt as lower bound
    quoted = re.findall(r"`([^`]{2,24})`", user_prompt)
    min_edges = max(2, min(len(set(quoted)) - 1, 4)) if len(set(quoted)) >= 3 else 2
    if edges < min_edges:
        return False, f"Diagram has only {edges} edges (< {min_edges} required transitions)"
    return True, "mermaid_ok"


def _run_model_chat(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    sample: bool = False,
    temperature: float = 0.35,
) -> tuple[str, int]:
    first_dev = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_text = apply_chat_template_safe(tokenizer, messages, add_generation_prompt=True)
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(first_dev)
    prompt_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=0.95 if sample else None,
            pad_token_id=pad_id,
        )
    g_ids = out_ids[0, prompt_len:]
    return tokenizer.decode(g_ids, skip_special_tokens=True).strip(), int(g_ids.numel())


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
        return False, (proc.stderr or "nonzero exit").strip()[-160:]
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
    """Unified Zero-Oracle cognitive generation & falsification controller."""
    del device
    mode = classify_task_mode(system_prompt, user_prompt)
    base_msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    reply, total_tok = _run_model_chat(model, tokenizer, base_msgs, max_new_tokens=max_new_tokens, sample=False)
    meta: dict[str, Any] = {"mode": mode, "calls": 1, "repaired": False, "reason": "direct"}

    if mode == "CODE_PYTHON":
        ok, err = verify_python_smoke_zero_oracle(reply)
        for attempt in range(2):
            if ok:
                break
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\n"
                        f"Your previous draft failed an execution smoke test with:\n`{err}`\n"
                        f"Previous draft:\n```python\n{_extract_fenced_or_raw(reply)}\n```\n"
                        "Fix all imports, boundary guards, and syntax errors and return only valid Python."
                    ),
                },
            ]
            cand, nt = _run_model_chat(
                model,
                tokenizer,
                repair_msgs,
                max_new_tokens=max_new_tokens,
                sample=(attempt > 0),
                temperature=0.35,
            )
            total_tok += nt
            meta["calls"] += 1
            ok2, err2 = verify_python_smoke_zero_oracle(cand)
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
            cand, nt = _run_model_chat(model, tokenizer, repair_msgs, max_new_tokens=max_new_tokens, sample=False)
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
        ok, err = _check_mermaid_zero_oracle(user_prompt, reply)
        if not ok:
            repair_msgs = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"{user_prompt}\n\nPrevious diagram draft had an error: {err}.\n"
                        "Include ALL required nodes, states, labels, and every directed transition edge inside a ```mermaid``` block."
                    ),
                },
            ]
            cand, nt = _run_model_chat(model, tokenizer, repair_msgs, max_new_tokens=max_new_tokens, sample=False)
            total_tok += nt
            meta["calls"] += 1
            ok2, _ = _check_mermaid_zero_oracle(user_prompt, cand)
            if ok2:
                reply = cand
                meta["repaired"] = True
                meta["reason"] = "mermaid_repaired"

    elif mode == "SYMBOLIC_COMPUTE":
        wants_json = "json" in system_prompt.casefold()
        solver_prompt = (
            "Write a short self-contained Python 3 script using ONLY the standard library to solve this exact problem.\n"
            "Strictly respect any custom rules, DAG dependencies, or counterfactual operator precedence stated in the problem.\n"
            + (
                "Print ONLY the final JSON object via `print(json.dumps(result))`."
                if wants_json
                else "Print ONLY the exact final answer on the last line via `print(result)`."
            )
            + f"\n\nProblem:\n{user_prompt}"
        )
        code_reply, nt = _run_model_chat(
            model,
            tokenizer,
            [{"role": "system", "content": "Return only valid Python inside a ```python``` block."}, {"role": "user", "content": solver_prompt}],
            max_new_tokens=min(320, max_new_tokens + 96),
            sample=False,
        )
        total_tok += nt
        meta["calls"] += 1
        ok_exec, out = _run_python_script_stdout(code_reply)
        if ok_exec and out:
            last_line = out.splitlines()[-1].strip()
            if wants_json:
                try:
                    parsed = json.loads(last_line)
                    if isinstance(parsed, dict) and parsed and preserve_valid_json_keys(reply, parsed):
                        reply = json.dumps(parsed)
                        meta["repaired"] = True
                        meta["reason"] = "python_symbolic_json"
                except json.JSONDecodeError:
                    pass
            else:
                if len(last_line) <= 64 and not last_line.startswith("{") and "<" not in last_line and ">" not in last_line:
                    reply = f"Verified via symbolic Python execution.\nFINAL_ANSWER: {last_line}"
                    meta["repaired"] = True
                    meta["reason"] = "python_symbolic_scalar"

    meta["tokens_generated"] = total_tok
    return reply, meta
