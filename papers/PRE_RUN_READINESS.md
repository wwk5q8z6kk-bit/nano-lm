# Pre-run readiness — everything required before the next compute run

**Written 2026-08-05 after H6's rejection.** This is a gate, not a wish list:
no paid run starts until every **BLOCKER** below is closed. Items are ordered
by dependency, and each names its verification command.

---

## 0. First: what *is* the next run?

H6's evaluator wrote the answer into its own decision record:

> *"reject exact H6 coupling; return to diagnosis without automatically running
> another architecture"*

And the evidence says what to diagnose. Across three independent measurements
today — H6's sealed development gates, wedge's real-corpus study, and fabric's
degenerate gate — the failure is the same: **epistemic-state calibration**
(absence / conflict / uncertainty), not value copying, which passes.

So the next run is **not** H7-as-another-architecture, and **not** rung-1
pretraining. It is whichever experiment the diagnosis in §1 selects. Two
candidates are already on the table, both cheap, and §1 decides between them:

- **A — abstention/state calibration.** Directly targets the three gates that
  killed H5 *and* H6. Data- or objective-side, not a new head.
- **B — slot-diversity × the winning corner.** The one lever ever measured to
  move the residual copying slot (+66.7 pts, D5→D80) has never been crossed
  with the LoRA/Chinchilla cell that solved the other slots. Cause is one line:
  `scribe/build_scribe_data_v2.py:30`, a five-value allergy pool.

A is aimed at the wall we just hit twice. B is aimed at a gap that is already
passing its floor. **A is the presumptive choice**; §1 must confirm on evidence.

---

## 1. BLOCKER — Diagnosis before design (free, local, no GPU)

**D1. Where do the H5/H6 state failures come from?**
Both runs' per-item development diagnostics are already on disk
(`uncalibrated_raw.item_diagnostics`, `by_field`, `by_gold_state` in
`artifacts/nano_h6/kaggle/eval-20260805/results/development_evaluation.json`).
Classify every absence/conflict/uncertain error: is the model predicting the
wrong *state*, the right state with a wrong *span*, or failing to abstain?
H5's residual was 727 state-correct/span-wrong vs 270 span-correct/state-wrong —
recompute that split for H6 and compare.
*Verify:* a written breakdown with counts per failure mode, committed.

**D2. Is the threshold policy itself the problem?**
`minimal_zero_wrong_presented_inclusive_v1` takes seed-20260805/epoch-2 from
**0.952 uncalibrated to 0.281 calibrated** on the calibration partition. A
policy that destroys two thirds of correct output to reach zero wrong-presented
is the same over-abstention disease living in the *gate*. Quantify the tradeoff
curve from the recorded calibration data (no new run needed).
*Verify:* a table of (threshold → retained correct, wrong presented) committed.

**D3. Decide A vs B on the D1/D2 evidence, and preregister it.**
No experiment is designed before D1 and D2 are written down.

---

## 2. BLOCKER — Instruments before training

**I1. The state-calibration instrument must exist before an experiment targets
state calibration.** Today the H-cycle gates measure joint exactness; they do
not isolate "should have abstained" from "should not have abstained." Without
that separation, a run cannot tell success from a lucky trade — which is
exactly how H6 produced +11.2 uncertainty and −19.6 absence and looked flat.
*Verify:* pinned tests over the new metric, plus recomputation of H5 and H6
from stored artifacts so the instrument has two historical anchors.

**I2. Publish the denominator in fabric (backlog B3).** `fabric/slice.py:247`
computes `presented_err / max(1, presented)` — abstain on everything, score
0.0%, pass the gate. Add coverage and end-to-end yield beside every presented-
error number, give unparseable generations a typed REVIEW decision so they
enter the statistics, and add a constant-free degeneracy guard.
*Verify:* `python3 -m pytest fabric` plus the coverage column in
`fabric/README.md`; the 0.0% claim carries its 81.5–91% coverage everywhere.

**I3. Finish W-ABSTAIN-2** (product side). Scope-matched denominator, 2-token
case treated separately, thresholds frozen before measuring.
*Verify:* `python3 -m wedge_v1.eval.margin_sweep`; 355+ tests green.

---

## 3. BLOCKER — Correctness bugs that would corrupt a run

These are confirmed defects, each cheap, each capable of silently invalidating
a training run.

**C1. `prepare_dataset` publishes before verifying.** `os.replace(staging,
output)` runs *before* `verify_prepared_dataset`, and the `except` branch
`rmtree`s a path that no longer exists — a failed verification publishes a
broken dataset under its real name and deletes nothing. Verify into staging,
then publish.
*Verify:* a test that forces verification failure and asserts no output
directory exists afterward.

**C2. Tokenizer EOD injection.** `pretrain/tokenizer.json` registers
`<|endoftext|>` as an added token, so any web page containing that literal
tokenizes to id 0 — the same id `prepare.py:126` appends as a document
boundary. FineWeb-Edu is web text; phantom boundaries will occur. The test
double (`ord(c) % 251 + 1`) can never emit 0, so the suite structurally cannot
catch it.
*Verify:* a test asserting no body token equals the EOD id, with a documented
sanitation step.

**C3. Hardcoded vocabulary size.** `prepare.py:231-232` writes
`"vocabulary_size": 4096` as a literal; both tokenizers are 4097/4098 with
added tokens. Derive it from the tokenizer.

**C4. No weight initialization.** `nano_ai/training/model.py` has none — masked
today only because every path warm-starts from an anchor. Any from-scratch
rung-1 pretrain would start from PyTorch defaults, silently.
*Verify:* explicit init with the recorded scheme (0.02, depth-scaled, per
`pretrain/AUDIT.md`), plus a test.

**C5. Determinism is claimed but unproven.** Training reports write
`"deterministic_algorithms": True`, yet no `sdpa_kernel`/`SDPBackend`/
`cudnn.deterministic` call exists anywhere in `nano_ai/`, and all 13
trainer call sites in tests sit inside `pytest.raises` — **no test has ever
completed a training step.** Prove bit-identity on CPU (free) with a two-step
double run, then pin forward.
*Verify:* two runs, same seed, byte-identical checkpoint SHA-256.

---

## 4. BLOCKER — Data path (only if the chosen experiment needs new data)

**P1. Tokenizer freeze with reserved special tokens.** Register `<think>`,
`</think>`, and structural symbols now — costless today, prevents a retrain
later. `<think>` currently appears zero times repo-wide.
**P2. Contamination digests.** The smoke set was prepared with
`contamination_exclusion: {entries: 0}`. Compute the 13-gram digest set over
the sealed partitions inside the authorized run before any scale prep.
**P3. Split the authorization state.** `approved_for_bounded_local_smoke`
carries semantics ("bounded", "smoke") that no code enforces — a 3.2B-token
run is preparable today under the smoke label. Add a token cap or a distinct
`approved_for_full_pretraining` state.
**P4. Single-pass streaming.** The corpus is currently streamed twice (train
and validation generators each traverse all 14 parquet files); invisible at
1M tokens, a second 28.5 GB download at scale.

---

## 5. BLOCKER — Evidence hygiene (the project's own standard)

**E1.** `artifacts/nano_h6/` has no `MANIFEST.json`, is absent from
`artifacts/MANIFEST.json`'s children, and 2 of ~140 files are in
`SHA256SUMS`. The subsystem devoted to provable provenance is the least
protected evidence set in the repo.
**E2.** `.gitignore:4` (`*.jsonl`) excludes every hash-chained run ledger from
version control. Record their hashes in the manifest now; resolve tracking in
the scrub pass.
**E3.** `EVIDENCE_MANIFEST.json` is stamped `updated: 2026-08-03` and covers
none of the last two days — including H6's terminal decision.

---

## 6. BLOCKER — Repo safety before any push

The GitHub remote is **PUBLIC** and 39+ local-only commits carry: RunPod
account balance, a second email address, a real-name↔email binding, SSH key
paths, and pod IPs. No credentials and no oversized files — but this is
permanent once pushed.
*Action:* scrub `artifacts/nano_h6/runops/**` and `papers/SUBMISSION_PACKET.md`,
add a LICENSE (the repo is public with none), then push. The 4 local-only tags
are safe to push today.

---

## 7. Owner decisions (batched — nothing else waits on these)

1. **RunPod ticket** — send the prepared text, or say "drop RunPod" (Kaggle now
   demonstrably runs the whole H-cycle for $0).
2. **Paper α** — submit (packet is one click) or decline.
3. **Rung-1 $150** — the budget is approved, but today's evidence argues the
   remaining gap is a *finetune-data* problem, not a scale problem. Recommend
   holding the spend until §1's diagnosis says scale is what's needed.
4. **Public repo + LICENSE** — keep public (needs a license) or make private.

---

## 8. Readiness checklist — the gate itself

No paid run begins until all of these are true:

- [ ] D1, D2 written and committed; D3 preregistered
- [ ] I1 instrument exists, pinned by tests, anchored on H5 + H6
- [ ] I2 fabric denominator published; degeneracy guard in place
- [ ] I3 W-ABSTAIN-2 decided (accept or reject, honestly)
- [ ] C1–C5 fixed, each with a test
- [ ] P1–P4 done **if** the chosen experiment touches data
- [ ] E1–E3 evidence registry current
- [ ] Preregistration frozen: hypothesis, gates, thresholds, kill rule, budget
- [ ] Platform decided (Kaggle free tier is the default; paid only if the
      experiment provably needs it)
- [ ] Two-domain backup path confirmed before the run, not after

Everything in §§1–5 is free, local, and needs no provider. That is the point:
**the entire remaining pre-run critical path costs $0.**
