# Stage T — scaling trajectory of the held-out copying gap

File categories (project-management convention, adopted 2026-07-17; applies
program-wide from Stage T on):

**Immutable records** — change only via visible, dated amendments:
- `PREREG.md` — the pre-registered design (2 amendments, both pre-measurement)
- `scribe_eval_T.json` — eval instance T (seed 20260717)
- `scribe_dev.json` — Arm 2 dev split (seed 20260718; never scored)
- `arm2_prompt.txt` — frozen Arm 2 prompt (no edits after first measurement)
- `results_arm1_*.json` — raw measurements (land here after Kaggle runs)

**Infrastructure** — evolves with versioning via git history:
- `gen_eval_instance.py`, `gen_dev_split.py` — instrument generation (exec the
  v1 generator definitions verbatim; do not fork the distribution)
- `kaggle_arm1.py` — Arm 1 finetune + one-shot measurement (Kaggle T4)

**Analysis** — exploratory, expected to evolve; lives under `analysis/` when it
exists; never feeds back into immutable records except through amendments or
the paper.

Execution order (per PREREG): equivalence check (first Arm 1 rung) → remaining
Arm 1 rungs → Arm 2 (frozen prompt, one greedy run per model per instance) →
report against pre-registered interpretation bands (PERSISTS / THRESHOLD /
DIVERGENT). Stage M (mechanism) is separately pre-registered after Stage T's
results are frozen and written up.
