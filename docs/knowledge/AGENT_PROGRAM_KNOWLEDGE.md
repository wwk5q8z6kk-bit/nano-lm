# Nano agent program knowledge

This is the compact operating map. It summarizes navigation; measured evidence
and the typed authorities linked below still win.

## What Nano is

Nano is the AI research program itself. Its current proving capability is P1:
faithful scribing from messy transcripts into grounded structured state. The
research, harnesses, and compute infrastructure exist to improve that AI; they
are not a separate research product.

```text
Nano Core (`nano/`)            domain-general evidence, state, and runtime
Learned Nano AI (`nano_ai/`)   learned transcript-to-structure research
Medical P1 (`nanoscribe/`)     encounter proposals, binding, and evaluation
```

## Authority and live state

1. Empirical artifacts and terminal decisions
2. [`PROJECT_CHARTER.md`](../PROJECT_CHARTER.md)
3. [`SYSTEM_ARCHITECTURE.md`](../SYSTEM_ARCHITECTURE.md)
4. [`ACTIVE_NOW.json`](../ACTIVE_NOW.json) and [`ACTIVE_NOW.md`](../ACTIVE_NOW.md)
5. [`EXECUTION_PLAN.md`](../EXECUTION_PLAN.md)

Use `origin/master` as integration truth. Cross-branch work is not current
authority until selectively reviewed and merged.

## Current scientific boundary

H6 is complete and **REJECTED**. Its one-shot evaluation was consumed on Kaggle
and is preserved at
[`artifacts/nano_h6/H6_TERMINAL_DECISION.json`](../../artifacts/nano_h6/H6_TERMINAL_DECISION.json).
The checkpoint passed overall and held-value copying but failed absence,
conflict, and uncertainty. Therefore:

- do not rerun, reopen, or rename H6;
- do not treat copying as the current primary blocker;
- localize abstention-calibration failures before proposing another learned
  architecture;
- keep future training data disjoint from frozen evaluation data.

## Compute choice

Use local CPU/MPS for development and cheap mechanism tests. Use Kaggle first
when its free runtime fits. Use RunPod for exact GPU/runtime needs, persistent
storage, or managed inference. Before paid work, bind the Git SHA, data revision,
termination condition, budget, and artifact destination. Never use PHI or
private owner material.

## Working sequence

```text
inspect live master + evidence
→ name one falsifiable bottleneck
→ implement the smallest diagnostic or intervention
→ test locally
→ freeze manifest and gates when evidence-bearing
→ run on the cheapest valid backend
→ verify and pull artifacts before teardown
→ update active state from the result
```

## Required checks

```bash
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
python3 -m pytest -q nanoscribe
python3 nanoscribe/benchmark.py
python3 -m pytest -q nano
python3 -m pytest -q nano_ai/tests
python3 -m pytest -q benchmarks/
```
