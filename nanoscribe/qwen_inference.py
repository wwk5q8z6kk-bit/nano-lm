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

    tokenizer = AutoTokenizer.from_pretrained(weights_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        weights_path,
        torch_dtype=dtype,
        trust_remote_code=True,
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

    tokenizer = loaded.tokenizer
    model = loaded.model
    messages = [{"role": "user", "content": user_prompt}]
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
    return text.strip().splitlines()[0].strip()


def generate_span_port_lines(
    model_input,
    atom_specs: Sequence[AtomSpec],
    *,
    weights_path: str,
    max_new_tokens: int = 48,
) -> tuple[dict[str, str], float, int]:
    """Run one generation per atom slot; return lines + latency + rss."""
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
