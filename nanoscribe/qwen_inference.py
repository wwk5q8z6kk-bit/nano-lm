"""Lazy Qwen2.5 span-port inference — optional torch/transformers dependency."""

from __future__ import annotations

import os
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoscribe.adapters import AtomSpec

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Pinned commit for DEFAULT_QWEN_MODEL. A bare tag is not content-addressed:
# the hub can move it and a rerun would silently not be a rerun. The pin lives
# in committed code rather than an env var so it travels with the experiment
# branch and shows up in the diff. Resolved 2026-08-25 from the HF API
# (lastModified 2024-09-25T12:32:50Z). Hub ids only; a local directory is used
# as-is.
DEFAULT_QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"

_PINNED_REVISIONS = {DEFAULT_QWEN_MODEL: DEFAULT_QWEN_REVISION}


def revision_for(weights_path: str) -> str | None:
    """Pinned revision for a hub id, or None for a local directory."""
    if os.path.isdir(weights_path):
        return None
    return _PINNED_REVISIONS.get(weights_path)


@dataclass
class _LoadedQwen:
    model: object
    tokenizer: object
    device: str


_CACHE: dict[str, _LoadedQwen] = {}


def resolve_weights_path(weights_path: str | None) -> str | None:
    if weights_path:
        return weights_path
    return os.environ.get("NANOSCIBE_QWEN_WEIGHTS") or None


def _rss_bytes() -> int:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports bytes; Linux reports kilobytes.
    if os.uname().sysname == "Darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss) * 1024


def _load_qwen(weights_path: str) -> _LoadedQwen:
    cached = _CACHE.get(weights_path)
    if cached is not None:
        return cached

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    revision = revision_for(weights_path)
    kwargs = {"trust_remote_code": True}
    if revision is not None:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(weights_path, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=dtype,
        **kwargs,
    )
    model.to(device)
    model.eval()
    loaded = _LoadedQwen(model=model, tokenizer=tokenizer, device=device)
    _CACHE[weights_path] = loaded
    return loaded


def _atom_user_prompt(spec: AtomSpec, model_input) -> str:
    from nanoscribe.prompt import build_span_port_prompt

    return build_span_port_prompt(model_input.source, spec)


def _generate_line(loaded: _LoadedQwen, user_prompt: str, *, max_new_tokens: int) -> str:
    import torch
    from nanoscribe.adapt import extract_span_port_line
    from nanoscribe.prompt import span_port_system_prompt

    tokenizer = loaded.tokenizer
    model = loaded.model
    messages = [
        {"role": "system", "content": span_port_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_span_port_line(text)


def generate_span_port_lines(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    weights_path: str,
    max_new_tokens: int = 48,
) -> tuple[dict[str, str], float, int]:
    """Run one generation per atom slot; return lines + latency + rss."""
    from nanoscribe.adapt import extract_span_port_line

    started = time.perf_counter()
    mem_before = _rss_bytes()
    loaded = _load_qwen(weights_path)
    lines: dict[str, str] = {}
    for spec in atom_specs:
        raw = _generate_line(
            loaded,
            _atom_user_prompt(spec, model_input),
            max_new_tokens=max_new_tokens,
        )
        lines[spec.atom_id] = raw
    latency_s = time.perf_counter() - started
    mem_after = _rss_bytes()
    return lines, latency_s, max(mem_after, mem_before)
