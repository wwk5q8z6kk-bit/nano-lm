# Training-data registry and acquisition plan

**Status:** Proposal-stage record. No new training run is authorized and none is
started by this document.
**Created:** 2026-08-04, by a review session at `/Users/mac` (Claude Code),
under the owner directive: *"Before provisioning hardware for any new training
run, identify and verify the complete training-data path… If no approved
training dataset exists, stop at the proposal stage."*
**Standing rule adopted:** Do not launch a paid training pod unless the
training script can access a complete, verified, authorized dataset. The frozen
H6 evaluation artifacts are **not** a training dataset.

---

## 1. Current verified training dataset (the only one that exists)

All Nano training data is synthetically generated in-repo by deterministic,
content-addressed generators. No external corpus, no scraped data, no PII, no
real clinical data. Verified on disk 2026-08-04 (hashes recomputed this day).

| Checklist item | Value |
|---|---|
| Dataset name / version | `byte_exact_h5_balanced_replay_v1` (generator `nano.replay-mixture-dataset.v1`) — the dataset H5 and H6 trained on. Upstream families: `nano.surface-transfer-dataset.v1` (H4) and the H3 evidence-query family. |
| Locality | **Local**, generated in-repo. Also packaged inside `artifacts/nano_h6/runpod-input/nano-h6-runpod-input.tar.gz`. Not persisted on any RunPod volume as authoritative copy; local files are canonical. |
| Approved local path | `artifacts/nano_h5/data/fit.jsonl`, `artifacts/nano_h5/data/calibration.jsonl` (+ `manifest.json`) |
| License / permitted use | Self-generated synthetic data; owner-authored. Permitted: Nano training, calibration, evaluation inside this program. Not evidence of clinical validity; not for open-world claims (see `papers/EVIDENCE_MANIFEST.json` `known_absences`). |
| File count / size | 2 data files, ~9.4 MB total: fit 8.7 MB (11,200 records / 2,800 worlds), calibration 720 KB (800 records / 200 worlds). |
| Manifest + hashes | `artifacts/nano_h5/data/manifest.json` (generator sha256 `5f036831ac7385c49177d9b724f6939049873179654072bc19dc178144e2af40`). File hashes recomputed 2026-08-04: fit `79f7581efbd989f48a474bdd2079e7ca163c2a2682bb48fb8398bd99febe22fb`; calibration `ce0562ccb44ee83963eace0d873773addaee8e49f29499a6b720b16335930e70` (byte-identical to H4 calibration — H5 preserved H4's calibration partition by design). |
| Tokenizer / preprocessing | `sft/tokenizer.json`, sha256 `bae49648bfcc4904c50e2f006ee184bd26e74454ee170663e30a8e71640ce3c9` (matches recovered copy `artifacts/nano_h6/recovered/h6-recovery-20260803/sft/tokenizer.json`; H6 prereg authenticated it before any development open). Preprocessing = deterministic generator replay (`nano_ai/training/replay_mixture_data.py`, content-addressed above): 50:50 H3/H4 world mixture at fixed exposure budget. |
| Split definitions | **Fit (train):** 2,800 worlds / 11,200 records. **Calibration:** 200 worlds / 800 records (H4 partition, preserved). **Decision partition (never trained on):** known-development `native-state-span-dev-v0`, sha256 `9c893d8e64110287b433d567e0e9abb42c611ecba33b40de192741324d37e290` — still sealed; never opened. **Final confirmation:** `fresh_v1` — sealed, never read. |
| Dedup / contamination controls | H4 manifest `isolation` block: world/record/template/transcript/open-value disjointness all true; `development_records_in_training: 0`; `historical_fresh_v0_read: false`; `sealed_confirmation_read: false`. H5 manifest `overlap_audit`: `all_hard_intersections_zero: true`, plus a literal-substring occurrence census across 2,800 candidate worlds. |
| Private / development data included | **None.** H6 prereg `prohibited_access`: fresh_v1 not accessed, historical fresh_v0 not accessed, wedge private data not accessed; development absent during training (attested in both frozen training reports). |
| Upload / download method | Content-addressed tar bundle with `BUNDLE_MANIFEST.json`, transferred only through the guarded ledger operation (see `artifacts/nano_h6/runops/evaluation-controls/`), per-file sha256 verified at destination before admission. |
| Expected transfer cost / duration | Bundle ≈ 13 MB → seconds; transfer cost ≈ $0. |
| Required storage | < 100 MB including checkpoints; existing 20 GB volumes are ample. |
| Destination path on volume | Prior protocol: network volume mounted at `/workspace`, bundle unpacked under `nano-h6-runpod/`. A next run must declare its own destination in its own authorization; do not reuse H6 authority. |
| Post-transfer verification | `shasum -a 256 -c` against the bundle manifest before any execution; destination admission program refuses to run otherwise (pattern: `nano_ai/training/admit_evidence_query_h6.py`). |

**Known gap:** `artifacts/SHA256SUMS` currently covers nano_h2–h4 entries;
H5/H6 data-file hashes are recorded here and in the run artifacts but should be
folded into `artifacts/SHA256SUMS` when the in-flight restructure lands.

## 2. Approval status for the next training run

**No approved next training run exists.** H6 closed
`INCONCLUSIVE_INFRASTRUCTURE_NO_DEVELOPMENT_ACCESS` (2026-08-04T23:35:33Z):
training completed and recovered, but the one-shot development evaluation was
never admitted because RunPod cannot reproduce the frozen host-kernel string
(`6.8.0-90` required, `6.8.0-134` observed; no host-kernel selector exists).
The one-shot gate is unconsumed; `automatic_next_architecture_authorized_after_failure`
is false in the H6 prereg. Therefore this document **stops at the proposal
stage**, per the standing rule.

## 3. Dataset acquisition and preparation plan (proposal only)

Three paths, in order of readiness. Each ends at a go/no-go that requires an
explicit owner authorization before any paid pod exists.

**Path A — reuse `byte_exact_h5_balanced_replay_v1` (ready now).**
For any next hypothesis that, like H6, holds data fixed and varies
architecture/objective. Acquisition cost: zero — files verified above.
Preparation: re-verify the two sha256s and the tokenizer hash; regenerate the
input bundle with the new run's executables; write a fresh preregistration
binding those exact hashes. Ready as soon as a hypothesis is authorized.

**Path B — new synthetic family via existing generators (days, $0 compute for
data).** For a data-side hypothesis. Generate under a new
`nano.<name>-dataset.v1` version with: a new manifest carrying the full
isolation block (H4's field set is the template), literal-substring overlap
audit (H5's template), disjointness against `native-state-span-dev-v0` and
`fresh_v1` enforced at generation time, per-file sha256 recorded at creation,
and the generator itself content-addressed. No provider resource needed to
build or verify.

**Path C — external / more-representative transcripts (owner-gated, do not
start).** Corresponds to `papers/EXECUTION_QUEUE.md` Priority 12 ("Deferred
until the core loop is stable"). Requires before anything else: source
selection with license and permitted-use verification, privacy assessment (no
real patient data enters this repo without an explicit owner decision and a
compliance review), independent labeling protocol, and a new contamination
policy. No acquisition work is authorized by this document.

**Protocol lesson to carry into any next runtime freeze (from the H6
closeout):** pin the runtime by *verifiable, provider-controllable* identities
(Python, torch, CUDA, tokenizers, GPU model, image digest) — do not freeze the
host kernel string, which the provider can change under any container and
cannot be selected. Encode kernel as *recorded observation*, not *admission
requirement*, unless the evaluation is proven kernel-sensitive.

## 4. Recheck commands

```bash
shasum -a 256 artifacts/nano_h5/data/fit.jsonl artifacts/nano_h5/data/calibration.jsonl
python3 -m json.tool artifacts/nano_h5/data/manifest.json | head -20
shasum -a 256 artifacts/nano_h6/recovered/h6-recovery-20260803/sft/tokenizer.json
runpodctl pod list          # must be [] unless a run is authorized
runpodctl network-volume list
```
