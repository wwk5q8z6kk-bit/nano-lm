"""Run the 8-pillar (132-item) NanoLM capability, tool-calling, RULER & reliability suite via vLLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from huggingface_hub import model_info
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from lab.trackb.eval_capabilities import evaluate_all_pillars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--think", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    args = ap.parse_args()

    if Path(args.model).is_dir():
        rev = None
        mfile = Path(args.model) / "nanolm_training_manifest.json"
        if mfile.exists():
            md = json.loads(mfile.read_text())
            total_params = md.get("total_params") or md.get("unique_params")
        else:
            total_params = None
    else:
        info = model_info(args.model)
        rev = info.sha
        total_params = info.safetensors.total if info.safetensors else None

    use_auto_dtype = any(k in args.model.lower() for k in ("gpt-oss", "awq", "fp8"))
    tok = AutoTokenizer.from_pretrained(args.model, revision=rev, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        revision=rev,
        dtype="auto" if use_auto_dtype else "bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        trust_remote_code=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    is_smollm3 = any(k in args.model.lower() for k in ("smollm3", "b6_3b", "b7_3b", "b8_3b")) or (
        "/no_think" in (getattr(tok, "chat_template", "") or "")
    )

    def _format_prompt(p: str) -> str:
        msgs = []
        if is_smollm3:
            msgs.append({"role": "system", "content": "/think" if args.think else "/no_think"})
        msgs.append({"role": "user", "content": p})
        if "gpt-oss" in args.model.lower():
            try:
                return tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="medium" if args.think else "low",
                )
            except TypeError:
                pass
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=bool(args.think)
            )
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate_batch(prompts: list[str]) -> list[str]:
        formatted = [_format_prompt(p) for p in prompts]
        t0 = time.perf_counter()
        outs = llm.generate(formatted, sp)
        dt = time.perf_counter() - t0
        out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        in_toks = sum(len(o.prompt_token_ids) for o in outs)
        generate_batch.telemetry = {  # type: ignore[attr-defined]
            "gen_seconds": round(dt, 2),
            "input_tokens": in_toks,
            "output_tokens": out_toks,
            "measured_output_tok_per_s": round(out_toks / max(0.01, dt), 1),
            "measured_input_tok_per_s": round(in_toks / max(0.01, dt), 1),
        }
        return [o.outputs[0].text for o in outs]

    rep = evaluate_all_pillars(
        generate_batch,
        model_id=f"{args.model}__think{args.think}",
        model_meta={
            "hf_repo": args.model,
            "revision": rev,
            "think": bool(args.think),
            "total_params": total_params,
            "active_params": total_params,
            "dtype": "bfloat16",
            "max_model_len": args.max_model_len,
        },
    )
    rep["leaderboards"]["C_efficiency"].update(getattr(generate_batch, "telemetry", {}))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rep, indent=2))
    print(json.dumps({
        "model": rep["model_id"],
        "overall": rep["overall_pass_rate"],
        "A_intelligence": rep["leaderboards"]["A_intelligence"],
        "B_reliability": rep["leaderboards"]["B_reliability"],
        "C_efficiency": rep["leaderboards"]["C_efficiency"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
