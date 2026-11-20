# Paper α Correction Note (post-freeze claim sync)

**Date:** 2026-07-31  
**Status:** `PUBLIC_FROZEN_CORRECTED`  
**Does not:** reopen Paper α science; rewrite original prereg chronology; authorize E4.

## Corrections synchronized into public methods/claims

1. **Nano 3.15M pretraining budget** is **32.8M tokens** (4000 steps, ~3.1 epochs of a 10.96M-token shard; `pretrain/AUDIT.md`), not ~200M. The 10M anchor remains ~200M (`scale/AUDIT.md`).
2. **Scale language** is schedule-aware: unequal token/parameter ratios across cells; no parameter-only “flat across ~50×” causal law.
3. **E1 substrate wording:** M1 dominates official generative refs under frozen U; M2 is within δ=0.05 of official M0; does not prove no generative-value regime (R★ untested).
4. **E3:** automated normalize 0/486; Stage-1 audit is **agent-applied rubric** (`agent-rubric-pass-1`), human/clinician arm **NOT_RUN**.
5. **ρ** in E1 U is **review load**, not hallucination rate.

Original local/prereg texts remain recoverable via git history and
`wip/pre-freeze-snapshot-20260731(-dirty)`.


## Gate after this correction set

Working-tree application of H/H′/I/J does **not** clear archival remediation.
Gate remains `AUDIT_REMEDIATION_REQUIRED` until the E1/E3 primary bundle is
committed and raw JSONL is durably published with hashes. Do not create a
post-α evidence tag until then; do not force-move `paper-alpha-v1`.
