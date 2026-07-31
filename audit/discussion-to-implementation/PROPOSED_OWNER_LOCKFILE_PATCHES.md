# Proposed Owner Lockfile Patches

*Draft only. Do not apply without owner approval. No scientific meaning changes intended beyond status accuracy and definition fixes.*

## P1 — `trajectory/PREREG_E2_lora_universes.md` Status
Replace RUNNING U3 paragraph with:
> **GATED / STOP (2026-07-31).** No `results_e2_*.json`. Stray pods terminated / none active (`runpodctl` empty). U3 not completed. Ban geometry language unchanged.

## P2 — `trajectory/FINDINGS.md` E2/E3 footer
- E3 human: change BLOCKED → **EXECUTED as agent-rubric audit** (`results_e3_human.json`); IAA absent.
- E2: change RUNNING → **GATED/STOP; no RESULT**.

## P3 — `papers/RESEARCH_PROGRAM.md` measured foundation table
- E3 human: BLOCKED → **rubric audit EXACT_SURVIVES** (not dual clinician).
- E2: BLOCKED (GPU) → **GATED/STOP post-KILL (no RESULT)**.

## P4 — `papers/EMPIRICAL_FOUNDATION.md` operating state
- Remove “human faithfulness pending” / “human pending” where body already says Stage 1 DONE; replace with “dual-clinician IAA / synonym ontology still open”.

## P5 — `trajectory/DECISION_P1_program_lock.md` ρ row
- Change ρ meaning to **review/flagged load (fraction of fields flagged)** to match `e1/common.py` and PREREG_E1.
- Hallucination remains separate (`halluc` / liability proxy).

## P6 — `trajectory/REGIME_P1_where_classical_fails.md`
- Paper α / Stage 1 line: “skipped” → “Stage 1 rubric audit executed; Gate 1 PASS”.
- Next: Stage 3 DONE; next decision is Stage 4 authorize or Idle.

## P7 — `fabric/README.md` / `fabric/slice.py` header
- Remove implication that Intent→Control is implemented.
- State ledger files are **per-run rewrite JSONL with content-addressed IDs**, not an append-only database.

## P8 — Paper α limitation (optional submission polish)
- Replace “pending a bounded human study” with: bounded agent-rubric audit n=100 yielded faithful-rate 0.00; dual-clinician IAA and synonym ontology remain unvalidated.

## P9 — Git archival (process, not prose)
- Owner chooses: commit post-α artifact/lock set **or** add `WORKING_TREE_LOCKS.md` stating locks are intentionally uncommitted.

## Explicitly NOT patched here
- Utility weights, E1 verdict, R★ τ’s, E4 decision rule, Paper α scientific claims beyond limitation clarity.
