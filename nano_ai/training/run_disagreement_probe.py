"""Does model/rule disagreement predict error? (plan P3, subtask 15)

EXPLORATORY. Selects nothing, gates nothing, trains nothing.

`RESULT_DP1_AND_THE_VOCABULARY_CEILING.md` established that
`contract._is_field_denial` is high-precision and low-recall: it fired 4/4
correctly on calibration and flipped 0 of 3,833 correct `supported`, but
recognises only ~3% of denial phrasings from independent clinical lexicons.

That shape is wrong for a decision rule and possibly right for an *escalation
signal*. This probe tests the only claim that would justify the router: that
fields where the model and the rule DISAGREE have a materially higher error rate
than fields where they agree. If they do not, the router idea dies here and is
recorded as dead.

Partition, per field:
  rule_fires  = any evidence span the model returned is a recognised denial
  model_says  = the model's proposed state is ABSENT
  agreement   = (rule_fires == model_says)

Then compare joint-exact accuracy inside each partition. Reported with counts,
not just rates -- a disagreement set of 4 fields cannot support a router however
its rate looks (the DP-1 lesson).

    python3 -m nano_ai.training.run_disagreement_probe \
        --checkpoint seed05=artifacts/.../seed-20260805/epoch-2.pt \
        --checkpoint seed06=artifacts/.../seed-20260806/epoch-2.pt \
        --output artifacts/nano_h6/analysis/disagreement.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from nano_ai.adapters.state_span import parse_state_span_summary
from nano_ai.contract import FieldState, _is_field_denial

# Below this many disagreements, no routing claim may be made regardless of
# the observed rate. Same discipline as MIN_SEEDS_FOR_ARM_CLAIM: a ratio over a
# handful of observations is not a measurement.
MIN_DISAGREEMENTS_FOR_CLAIM = 30


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--development", type=Path, default=Path("artifacts/nano_h6/kaggle/dataset-dev")
    )
    parser.add_argument("--tokenizer", type=Path, default=Path("sft/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from nano_ai.training.evaluate_pointer import build_pointer_inference_inputs
    from nano_ai.training.evaluate_state_span import load_development_bundle
    from nano_ai.training.evidence_query_inference import (
        _proposal_exact,
        batched_evidence_query_inference,
    )
    from nano_ai.training.run_generalization_probe import _load_model
    from nano_ai.training.train_pointer import encode_pointer_partition, load_pointer_tokenizer

    tokenizer = load_pointer_tokenizer(args.tokenizer)
    manifest_sha = hashlib.sha256(
        (args.development / "manifest.json").read_bytes()
    ).hexdigest()
    examples = load_development_bundle(
        args.development, expected_manifest_sha256=manifest_sha
    ).examples
    records = encode_pointer_partition(tokenizer, examples, expected_split="dev")
    inputs = build_pointer_inference_inputs(examples, records)

    per_seed = {}
    for entry in args.checkpoint:
        name, _, path = entry.partition("=")
        model = _load_model(Path(path), args.device)
        with torch.inference_mode():
            inference = batched_evidence_query_inference(
                model, inputs, device=args.device, batch_size=args.batch_size
            )

        correct = Counter()
        total = Counter()
        gold_mix = {"agree": Counter(), "disagree": Counter()}
        for prediction, example in zip(inference.predictions, examples, strict=True):
            proposed = prediction.proposals if prediction.error is None else ()
            golds = parse_state_span_summary(example.target, example.transcript)
            for index, gold in enumerate(golds):
                if not proposed:
                    continue
                proposal = proposed[index]
                rule_fires = any(
                    _is_field_denial(proposal.field, span.text)
                    for span in proposal.spans
                )
                model_says = proposal.state is FieldState.ABSENT
                bucket = "agree" if rule_fires == model_says else "disagree"
                total[bucket] += 1
                correct[bucket] += _proposal_exact(proposal, gold)
                gold_mix[bucket][gold.state.value] += 1

        def rate(bucket):
            return correct[bucket] / total[bucket] if total[bucket] else None

        agree_err = 1 - rate("agree") if total["agree"] else None
        dis_err = 1 - rate("disagree") if total["disagree"] else None
        per_seed[name] = {
            "agree": {"n": total["agree"], "correct": correct["agree"],
                      "error_rate": round(agree_err, 4) if agree_err is not None else None,
                      "gold_states": dict(gold_mix["agree"])},
            "disagree": {"n": total["disagree"], "correct": correct["disagree"],
                         "error_rate": round(dis_err, 4) if dis_err is not None else None,
                         "gold_states": dict(gold_mix["disagree"])},
            "error_rate_lift": (
                round(dis_err / agree_err, 2)
                if agree_err and dis_err is not None and agree_err > 0
                else None
            ),
            "claim_supported": bool(
                total["disagree"] >= MIN_DISAGREEMENTS_FOR_CLAIM
                and dis_err is not None
                and agree_err is not None
                and dis_err > agree_err
            ),
        }
        d = per_seed[name]
        print(
            f"{name}: agree n={d['agree']['n']:5d} err={d['agree']['error_rate']:.1%}   "
            f"disagree n={d['disagree']['n']:5d} err="
            f"{d['disagree']['error_rate']:.1%}   "
            f"lift={d['error_rate_lift']}x   claim_supported={d['claim_supported']}"
        )

    payload = {
        "schema": "nano.model-rule-disagreement.v1",
        "status": "EXPLORATORY -- selects nothing, gates nothing",
        "question": "does model/rule disagreement predict error well enough to route on?",
        "min_disagreements_for_claim": MIN_DISAGREEMENTS_FOR_CLAIM,
        "per_seed": per_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
