# E1 runtime / cost schema (L and C)

*Freeze-compatible documentation. No new experiments.*

## Utility terms

Under frozen E1 weights (`trajectory/e1/common.py` `DEFAULT_WEIGHTS`):

\[
U = \alpha P - \beta M - \gamma \rho - \lambda L - \kappa C
\]

with defaults \(\alpha=1, \beta=0.5, \gamma=0.3, \lambda=0.02, \kappa=0.05\).

| Symbol | Meaning in code | Source field |
|--------|-----------------|--------------|
| \(P\) | Presented precision | `InstanceMetrics.as_rates` → `P` |
| \(M\) | Miss rate \(=1-\mathrm{recall}\) | `M` |
| \(\rho\) | **Review load** = `flagged / n_fields` | `rho` — **not** hallucination rate |
| \(L\) | Median per-item latency (seconds) | `L_p50` from `latencies` median |
| \(C\) | Relative compute cost (dimensionless schedule) | method `cost_c` argument / row `C` |

Hallucination rate is logged separately as `halluc` and is **outside** \(U\) v1.

## How \(L\) was measured

- Wall-clock per item via `time.perf_counter()` around `predict_fn` in `evaluate_method`.
- Aggregated as the median of per-item latencies (`L_p50`).
- Official generative M0 scored on RunPod CUDA fp16; classical methods on the same harness host class used for that run.
- Raw monitor logs may exist under `trajectory/e1_official_m0*.log` / `trajectory/runpod_partial/` — useful for audit, not a substitute for the structured row fields.

## How \(C\) was assigned

`C` is a **relative cost schedule**, not a dollar invoice:

| Method family (examples) | Typical `C` in committed rows |
|--------------------------|-------------------------------|
| M1 template / rules | 0.02 |
| M2 dict+span | 0.03 |
| M3/M4-class methods | ~0.04–0.20 |
| Local/scale generative | higher |
| Official Pythia-160M LoRA | 1.0 (reference) |

Exact per-method values are those stored in `trajectory/results_e1_utility.json` rows. Recompute of \(U\) must use those stored `C` values; do not re-infer GPU pricing without a new prereg amendment.

## Recompute without GPU

```bash
pytest trajectory/test_e1_utility_recompute.py trajectory/test_e3_normalize.py
```

This validates arithmetic and locked decision numbers from committed JSON. It does **not** re-time models or re-derive `C`.

## Gap / future (not authorized here)

A dedicated `results_e1_runtime.json` with per-method device, CUDA version, n_warmup, raw latency vector SHA, and explicit `C` formula would strengthen auditability. Collecting it is optional packaging work, not an E4 unlock.
