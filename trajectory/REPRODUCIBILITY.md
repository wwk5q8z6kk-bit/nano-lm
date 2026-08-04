# Stage T / T-v2 — reproducibility manifest

Pins the exact instrument state that produced the measurements, so any rung can
be re-derived. Immutable inputs are content-addressed by SHA-256.

## Frozen tags

- `stage-t-v1` — Arm 1 v1 instrument (single-instance eval), pre-measurement.
- `stage-t-v2` — commit `101e429` — powered instrument (5×200 multi-instance
  eval + re-scoring script), pre-measurement.

## Scoring inputs (immutable eval instances) — SHA-256

```
9b8a966f…  scribe/scribe_eval.json          (instance 0, public, seed 7)
a118d34c…  trajectory/scribe_eval_T.json    (instance T, seed 20260717)
2377b0a3…  trajectory/scribe_eval_m0.json   (seed 20260720, 200 items)
c909f368…  trajectory/scribe_eval_m1.json   (seed 20260721)
c7759990…  trajectory/scribe_eval_m2.json   (seed 20260722)
ede686b6…  trajectory/scribe_eval_m3.json   (seed 20260723)
4830bbfc…  trajectory/scribe_eval_m4.json   (seed 20260724)
```

## Training recipe + scoring code — SHA-256

```
3fa1b251…  scribe/build_scribe_data_v2.py   (training data recipe, seed 11)
3c4631f9…  scribe/build_scribe_data.py      (eval distribution, v1)
66e7600d…  trajectory/kaggle_arm1_v2.py     (finetune + powered scorer)
```

## Anchor re-scoring (nano/scale on the multi-instance instrument)

Added 2026-07-17 (`PREREG_anchors.md`). The 3.15M/10M own-stack anchors were re-scored
on the same five fresh instances (m0–m4) as the Pythia rungs, using their native
ChatML/greedy scorer (`trajectory/rescore_anchors.py`), device MPS. Re-scoring only —
frozen v0.1 release checkpoints:

```
0e4f348e…  scribe.pt           (nano scribe v2, 3.15M)   — v0.1 release asset
f5aca5f0…  scale10m_scribe.pt  (scale scribe, ~10M)      — v0.1 release asset (matches scale/AUDIT.md)
```

Determinism cross-check passed before the multi-instance pass: nano inst0 reproduced
`gate_scribe_v2.log` byte-for-byte (parse 39/40, recall 81%, held 68/95, seen 94/100,
gap 22.4); scale inst0 reproduced Stage S exactly (parse 100%, recall 88%, gap 23.0)
despite the CUDA→MPS device change. Results: `results_anchors_v2_{nano,scale}.json`
(nano 18.3±1.3, scale 18.7±1.5 across m0–m4).

## Model provenance

Base models: EleutherAI/pythia-{160m,410m,1b} from HF Hub. Finetune is
deterministic on T4 at seed 20260717 (LoRA r=16 α=32, LR 1e-4, 3 epochs). The
frozen adapter is regenerated per run rather than uploaded; equivalence to the v1
adapter is verified IN-BAND by the determinism cross-check (below), not assumed.

## Determinism evidence (why "re-score", not "re-run")

- Headless-T4 pythia-410m (v1) reproduced the interactive-T4 pythia-410m byte-for-
  byte on every metric.
- T-v2 re-scores instance 0 and instance T alongside the fresh instances; the v2
  inst0/instT gaps matched the v1 archived JSONs exactly (160m 7.0/2.0, 410m
  8.0/2.0). Same finetune → same model → same gaps. The multi-instance mean is
  therefore a re-measurement of the SAME frozen model, not a new experiment.

## Library environment (recorded in every results JSON)

torch 2.10.0+cu128 · transformers 5.0.0 · peft 0.19.1 · python 3.12.13 · Tesla T4.

## Execution notes (operational, NOT part of the scientific record)

- Kaggle preinstalls torchao 0.10.0, incompatible with peft's LoRA dispatcher →
  `pip uninstall -y torchao` in every kernel (torchao unused).
- Headless kernels must pin `machine_shape: NvidiaTeslaT4`; the default GPU can be
  a P100 (sm_60) which the torch build cannot run.
- Kaggle caps batch GPU sessions at 2; the third rung waits for a free slot.
- Local watcher processes exited intermittently; this had
  no effect on the experiment — Kaggle jobs run server-side independently and all
  completed artifacts are pulled and archived. Recovery of any rung is by
  re-pulling the completed kernel output, not re-running.


---

## Program-wide reproducibility packaging (2026-07-30 SRC baseline)

### Dependencies and environment

| File | Role |
|---|---|
| `requirements.txt` | CPU CI: pytest + numpy |
| `requirements-ml.txt` | Training / scoring stack (torch, transformers, peft, …) |
| `environment.yml` | Conda env skeleton (Python 3.12 + CPU deps) |
| `pyproject.toml` | Project metadata + pytest discovery |
| `LICENSE` | MIT |

Recorded GPU scoring env (Stage T / own-stack kernels; see JSON `env` fields when present):
torch 2.10.0+cu128 · transformers 5.0.0 · peft 0.19.1 · python 3.12.13 · Tesla T4
(and venue-specific H100 / A6000 / RTX 4090 / MPS runs noted in respective PREREGs).

### Deterministic evaluation entry points (no new GPU work required to *check*)

| Check | Command | Needs GPU? |
|---|---|---|
| Fabric regression pins | `pytest fabric/test_fabric.py` or `python3 fabric/test_fabric.py` | No |
| C-3 recompute harness fixtures | `pytest trajectory/test_recompute_c3.py` | No |
| CI | `.github/workflows/ci.yml` runs both above on Python 3.12 | No |
| Anchor re-score (native) | `python trajectory/rescore_anchors.py` | Yes (or MPS) |
| Batched scorer parity | `trajectory/batched_scorer.py` (byte-identical vs native on anchors) | Yes |
| Own-stack / Pythia / sweep kernels | `trajectory/kaggle_*.py`, RunPod scripts in PREREGs | Yes |

### How to obtain / verify result artifacts

1. **Immutable JSONs live in-repo** under `trajectory/results_*.json`,
   `trajectory/sweep_eval/`, `trajectory/c3_eval/`, `trajectory/interference_eval/`,
   `trajectory/replications/`, and `fabric/results_slice_v1.json`.
2. **Do not overwrite** result JSONs; new runs write new files or
   `trajectory/replications/<slug>/`.
3. **Content-addressed eval inputs** — SHA-256 pins above (m0–m4, inst0, recipes).
4. **Git tags** (scientific freezes): `stage-t-v1`, `stage-t-v2` (commit `101e429`).
   Prefer tagging subsequent freezes the same way before mutating instruments.
5. **Verify a claim** by (a) finding the JSON path cited in the paper/program doc,
   (b) reading `diluted_gap_mean` / `clean_gap_mean` / `decisions` / fabric
   `presented_error_rate`, (c) confirming the matching PREREG RESULT section.
6. **Corner cells (established):**
   `results_corner_3p2b_lora_seed0.json` and `..._seed1.json` both report
   `diluted_gap_mean = 4.24`, `diluted_gap_sd ≈ 0.91` (|Δ|=0.00).
7. Checkpoints (`.pt`) are **not** in git (`.gitignore`); use release assets /
   documented mounts. Scoring code + frozen eval JSON is enough to re-derive
   metrics from a mounted checkpoint.

### Claim discipline reminder

Reproducing a number does not support a broader mechanism or product claim.
See `papers/EMPIRICAL_FOUNDATION.md` and `papers/EVIDENCE_LEDGER.md`.

---

## Post-α E1 / E3 evidence bundle (packaging, 2026-07-31)

Machine-readable map: `papers/EVIDENCE_MANIFEST.json`.

### Evidence basis for KILL / construct claims

| Claim | Prereg | Primary outputs | Local verify |
|-------|--------|-----------------|--------------|
| E1 KILL (H-substrate) | `PREREG_E1_nonlm_baseline.md` | `results_e1_utility.json`, `results_e1_utility_sensitivity.json` | `decision.verdict=KILL`; M1 U≈0.999 vs official M0 U≈0.925; margin≈+0.074; `sensitivity_flip=false` |
| E3 normalize | `PREREG_E3_faithfulness_construct.md` | `results_e3_normalize_construct.json` | `norm_rescue_count=0`, `both_fail=486` |
| E3 human (bounded) | same | `e3_human_rating_pack.json` + `results_e3_human.json` | faithful 0/100; EXACT_SURVIVES; IAA absent |
| E2 | `PREREG_E2_lora_universes.md` | *(none)* | **GATED/STOP** |
| R★ / E4 | `REGIME_P1_…`, `PREREG_E4_…` | *(none)* | **Protocol only; not measured** |

### Utility symbols (E1)

Authoritative definitions: PREREG_E1 + `trajectory/e1/common.py` (`rho = flagged / n_fields` = **review load**).
Do not read ρ as hallucination. Decision margin δ=0.05 ≠ sensitivity cost weight δ_C.

### M1 information constraint

M1 is a **rules-perfect template extractor** (`fabric._extract`) for this synthetic dialogue
generator — not a train-lexicon-only method. M2 is the train-dict + span baseline.
KILL remains valid under the frozen prereg; interpret M1 as classical/symbolic ceiling
for this world, not as a weak heuristic.

### Git packaging note

These E1/E3 files were **present locally and untracked**, not gitignored. Tag
`paper-alpha-v1` therefore predates the E1/E3 bundle. Use the current
`papers/EVIDENCE_MANIFEST.json` and repository history to locate the packaged evidence;
exclude transient logs and partial outputs from scientific bundles.
