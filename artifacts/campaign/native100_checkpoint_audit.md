# native100_* checkpoint audit

Task 5 of the Native30 revalidation session. Recorded before anything in this
tree was touched. **Nothing here was modified.**

## Inventory

| dir | `latest.pt` | `step_*.pt` | `*.json` | size |
|---|---|---|---|---|
| `native100_evidence_bottleneck_s0` | **NO** | 0 | 1 (`step_000006.json`) | 4.0K |
| `native100_evidence_bottleneck_s1` | yes | 0 | 6 (`step_000006` … `step_000200`) | 1.1G |
| `native100_span_port_s0` | yes | 0 | 1 | 1.1G |
| `native100_span_port_s1` | yes | 0 | 1 | 1.1G |

Confirmed: the three surviving dirs hold **only** `latest.pt` with no `.pt` step
checkpoints. They are genuinely single-copy — there is no second copy of any of
them anywhere in the tree. Any operation on this directory is unrecoverable.

## `native100_evidence_bottleneck_s0` — LOST ARTIFACT, not an incomplete run

**Verdict: the checkpoints were written successfully and later deleted.**

The determination is decidable from write order rather than inference.
`save_checkpoint` (`nanoscribe/native/checkpoint.py:14-47`) writes in this
sequence:

1. `torch.save(payload, out)` → `step_000006.pt`
2. `torch.save(payload, latest)` → `latest.pt`
3. `meta.write_text(...)` → `step_000006.json`

**The JSON is written last.** For `step_000006.json` to exist, both `.pt` writes
must already have completed. The surviving file also carries exactly the five
keys `meta_payload` emits — `schema`, `timestamp`, `step`, `config`, `extra` —
so it was written by this path and not by some other producer.

An *incomplete run* would leave the opposite signature: a `.pt` with no `.json`,
or nothing at all. What is on disk is a `.json` with no `.pt`, which that
ordering cannot produce.

**Contents:** step 6 of `max_steps` 200, `loss` 61.734, timestamp
`2026-08-23T10:51:49Z`, `native_b_evidence_aware`, d_model 704 / 16 layers /
100M. Its seed-1 sibling from the same arm completed all 200 steps.

**Most likely cause: disk-pressure cleanup.** The volume is at 89% with 97 GiB
free, each of these dirs is 1.1G, and a 19GB untracked Kaggle download was
sitting beside them (now gitignored, commit `a62ad8b`). A sweep that deleted
`*.pt` while leaving small `.json` files would produce exactly this state, in
exactly one directory.

**Not recoverable.** `.pt` files are gitignored, so there is no object in git to
restore from; unlike `artifacts/measurement-integrity-audit.md`, which this
session recovered from WIP commit `1b241d6`.

## One anomaly, recorded rather than resolved

`step_000006.json` should not exist under the current cadence. Checkpoints save
when `step % max(1, max_steps // 5) == 0`, which for `max_steps=200` is every 40
steps — 40, 80, 120, 160, 200. Step 6 is not on that schedule, and the config
embedded in the file records `max_steps: 200`.

Step 6 *is* on the schedule for `max_steps=30` (`30 // 5 == 6`). The seed-1
sibling carries both a `step_000006.json` and the regular 40-step series, which
is consistent with a short smoke run at `max_steps=30` followed by a full run at
200 into the same directory.

This does not change the verdict — the write-order argument holds regardless of
which run produced the file — but it means `native100_*` may contain checkpoints
from **two different configurations sharing a directory**, and `latest.pt` is
whichever wrote last. Anything comparing across these dirs should verify the
step and config inside each payload rather than trusting the directory name.

## Recommendation

Leave this tree alone. It is single-copy, one member has already lost its
weights, and the revalidation wave writes to `reval30_*_fixed_*` and does not
touch it. If disk pressure needs relief, the 19GB
`artifacts/campaign/kaggle_native30_download/` is the target — it is a
re-downloadable copy containing a nested git repo, and reclaiming it frees ~5×
more space than all three surviving native100 checkpoints combined.
