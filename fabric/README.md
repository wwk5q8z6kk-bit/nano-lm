# fabric/ — closed-world verification regression harness

Fabric is a closed-world verification regression harness over the existing scribe
prototype: supplied synthetic clinic-dialogue transcripts and five structured
fields. It exercises the smallest end-to-end verification-first path — nano-lm
generator → typed Claims → Verifier → Risk decision → per-run ledger serialization
— measured on the frozen inst0 instrument.
It is not the complete Nano AI or a general Nano architecture.

**Boundary (2026-07-31 evidence freeze):** The repository contains a measured
verification vertical slice with typed claims, source-grounding rules, contradiction
states, abstention behavior, and content-addressed identifiers. It is **not** an
append-only transactional database or a complete cognitive operating system.
Ledger `*.jsonl` files are **per-run rewritten** artifacts (open mode `"w"`), not an
append-only store. Intent/control-kernel stages in older plan prose are **not
implemented** in this directory. It does not implement Nano's full scribe
intelligence; any broader scribe capability remains unimplemented unless separately
evidenced.

## Results (2026-07-20, frozen anchors, inst0 = scribe_eval.json, 40 dialogues)

| model | verifier | raw gen. error | presented error **@ coverage** | caught | lost correct | provenance |
|---|---|---|---|---|---|---|
| nano 3.15M | grounding.v1 | 18.4% (35/190) | **1.9%** (3/158) **@ 83.2%** | 32/35 | 0 | 100% |
| nano 3.15M | grounding.v2 | 18.4% | **0.0%** (0/155) **@ 81.6%** | 35/35 | 0 | 100% |
| scale 10M | grounding.v1 | 11.5% (23/200) | **1.1%** (2/179) **@ 89.5%** | 21/23 | 0 | 100% |
| scale 10M | grounding.v2 | 11.5% | **0.0%** (0/177) **@ 88.5%** | 23/23 | 0 | 100% |

> **Read every presented-error number with its coverage.** `presented error`
> divides by `presented`, a quantity the verifier controls: a verifier that
> presented nothing would score **0.0% and pass gate v1**, which is why
> `gate_pass_v2` adds a constant-free non-degeneracy guard denominated in
> `raw_pred` (the shape `scribe/gate_grounded.py:129-133` already uses).
> Pinned by `test_gate_v1_passes_a_mute_verifier_and_gate_v2_does_not`.
>
> On this evidence the verifier is **not** over-abstaining: across all 24 cells
> of `results_slice_v1.json`, withheld = 2,642 and `caught_err` = 2,642 with
> `lost_correct` = **0** — every withheld field was an error. Coverage sits at
> 81.5–92.2% because the *generator* was wrong that often, not because the
> verifier was timid. Canonical terms: `papers/SELECTIVE_VOCABULARY.md`.


Held/seen: scale's 23 errors are **100% held-side** (pure held-out-value copying failure);
nano 28 held / 7 seen. All caught. Phase 1 gate — equal-or-better failure rates on the
clean metric + full provenance per claim — **exceeded on every cell**.

## The v1→v2 delta (verifier-strength axis on this task)

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
- `ledger_{model}_inst0_{verifier}.jsonl` — per-run rewritten JSONL with
  content-addressed IDs per claim (claim, result, decision, spans, eval-only truth);
  gitignored; not an append-only DB.

## Lexicon/template provenance

Question anchors, reply templates, and denial templates are exec'd directly from
`scribe/build_scribe_data.py` (train+held union) — nothing hand-guessed; denial variants
were additionally verified empirically against all six frozen instruments (single
variant per slot in eval: "Nothing at all." / "None whatsoever.").

## Claim scope (SRC baseline 2026-07-30)

Presented-error → 0.0% under **grounding.v2** is an existence proof for
propose→verify→abstain on this synthetic task under a **rules-strong, decidable**
verifier relation. It does **not** license open-world zero-hallucination, Nano
AI generalization, or broader scribe-capability claims. v2 could solve the task
alone (documented above).

E1 and E4 do not support expanding this slice into a Nano architecture; E2
has no result. See `papers/EMPIRICAL_FOUNDATION.md`. This directory remains a
**regression harness** for measured failure classes (`test_fabric.py`, CI).

## Status of prior "next"

C-1b and C-3 are **closed** (interference REFUTED; T/B REFUTED; morphology
descriptive). Do not treat them as open fabric prerequisites.
