# fabric/ — NanoScribe Phase 1 minimal vertical slice

Master Plan action #3: the smallest end-to-end verification-first path on the existing
scribe task — Intent → nano-lm generator → typed Claims → Verifier → Risk decision →
Evidence Ledger — measured on the frozen inst0 instrument.

## Results (2026-07-20, frozen anchors, inst0 = scribe_eval.json, 40 dialogues)

| model | verifier | raw gen. error | presented error | caught | lost correct | provenance |
|---|---|---|---|---|---|---|
| nano 3.15M | grounding.v1 | 18.4% (35/190) | **1.9%** (3/158) | 32/35 | 0 | 100% |
| nano 3.15M | grounding.v2 | 18.4% | **0.0%** (0/155) | 35/35 | 0 | 100% |
| scale 10M | grounding.v1 | 11.5% (23/200) | **1.1%** (2/179) | 21/23 | 0 | 100% |
| scale 10M | grounding.v2 | 11.5% | **0.0%** (0/177) | 23/23 | 0 | 100% |

Held/seen: scale's 23 errors are **100% held-side** (pure held-out-value copying failure);
nano 28 held / 7 seen. All caught. Phase 1 gate — equal-or-better failure rates on the
clean metric + full provenance per claim — **exceeded on every cell**.

## The v1→v2 delta is the program's thesis in miniature

- **grounding.v1** (generic, world-blind): a VALUE claim is verified iff the value string
  appears with word boundaries in a *patient* line (role-aware). Its entire residual
  (1.1–1.9%) is **binding failures literal grounding cannot see**: cross-slot capture
  ("moderate" presented as chief complaint), template-word capture ("troubling"), and
  partial copy ("throat" ⊂ "throat lozenges" — the known med truncation).
- **grounding.v2** (template-anchored, world-grammar-aware): the claim must equal the
  captured group of the slot reply's template match; mismatches become CONTRADICTED
  *with the actually-bound value as counter-evidence*. Removes the residual exactly.

Caveat, stated openly: v2 is a rules-perfect reference extractor for this closed
synthetic world — it could solve the task alone. The slice measures the **fabric**
(typed packets, hard gates, decision policy, ledger, measured deltas), and the v1/v2
pair measures the verifier-strength axis, not verifier novelty.

## Hard rules enforced in code (see `schemas.py`)

- VERIFIED requires ≥1 evidence span; a PRESENTed claim without spans raises.
- ¬Found(x) ⇏ ¬x: an ABSENT claim with no *positive* absence evidence (explicit denial
  span) is UNVERIFIABLE, never VERIFIED → QUALIFY, never PRESENT.
- CONTRADICTED is first-class and carries counter-evidence.
- All IDs are content-addressed (sha256) for immutable lineage.

## Files

- `schemas.py` — Claim / EvidenceSpan / VerificationResult / Decision (frozen
  dataclasses, invariant-enforcing validators, JSON round-trip). Self-test: `python3 fabric/schemas.py`.
- `slice.py` — the slice runner. `NANO_CKDIR=$PWD/checkpoints/anchors python3 fabric/slice.py nano scale`
- `test_fabric.py` — 8 model-free regression pins for every measured failure class.
- `results_slice_v1.json` — the 2×2 matrix above.
- `ledger_{model}_inst0_{verifier}.jsonl` — full lineage per claim (claim, result,
  decision, spans, eval-only truth).

## Lexicon/template provenance

Question anchors, reply templates, and denial templates are exec'd directly from
`scribe/build_scribe_data.py` (train+held union) — nothing hand-guessed; denial variants
were additionally verified empirically against all six frozen instruments (single
variant per slot in eval: "Nothing at all." / "None whatsoever.").

## Next (per owner Master Plan ordering)

C-1b lexical-interference experiment (PREREG_token_coverage.md AMENDMENT 1) runs after
this slice — the fabric's ledger (per-item outputs + spans) is the instrumentation
C-1b's mandatory per-item logging requires.
