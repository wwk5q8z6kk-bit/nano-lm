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
- Local watcher processes were repeatedly reaped by the session harness; this had
  no effect on the experiment — Kaggle jobs run server-side independently and all
  completed artifacts are pulled and archived. Recovery of any rung is by
  re-pulling the completed kernel output, not re-running.
