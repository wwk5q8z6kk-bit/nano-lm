# E1 L/C Reconstruction Protocol (DESIGN ONLY)

**Status:** `DESIGN_DRAFT` — not authorization to run.  
**Purpose:** Separate **decision admissibility** from **cost-term plausibility** for E1 packaging (`PUBLIC_PARTIAL` on cost; KILL remains).  
**Companions:** `papers/FIRST_PRINCIPLES_RISK_MITIGATION.md` (B3, B18), `trajectory/E1_RUNTIME_SCHEMA.md`, `trajectory/e1/common.py`, `trajectory/test_e1_utility_recompute.py`.

## First principle

\[
\text{Decision reproducibility} \;\perp\; \text{Cost-term reproducibility}
\]

- **Admissible (decision):** From published row components \((P,M,\rho,L_{\mathrm{p50}},C)\) and frozen weights, recompute \(U\) and re-fire `aggregate_decision` → **KILL**. Already pinned offline by pytest.
- **Plausible (cost):** \(L_{\mathrm{p50}}\) and schedule \(C\) match a clean-clone / documented device table within ε. Not required to keep the KILL claim; required to upgrade cost reproducibility.

Do **not** “fix” PUBLIC_PARTIAL by widening the claim. Narrow and label.

## Frozen utility (code)

From `trajectory/e1/common.py` `DEFAULT_WEIGHTS`:

\[
U = \alpha P - \beta M - \gamma \rho - \lambda L_{\mathrm{p50}} - \kappa C
\]

with \((\alpha,\beta,\gamma,\lambda,\kappa)=(1.0,0.5,0.3,0.02,0.05)\).

- \(P\): presented precision  
- \(M = 1-\mathrm{recall}\)  
- \(\rho\): **review load** = flagged / \(n_{\mathrm{fields}}\) (**not** hallucination)  
- \(L_{\mathrm{p50}}\): median per-item wall time (seconds)  
- \(C\): relative compute schedule (dimensionless; not dollars)

Kill rule: best non-LM \(U \ge U(M0)-\delta\) with \(\delta=0.05\), else SURVIVE; sensitivity flip → GRADED.

## Proposed claim split (ledger migration — needs owner commit)

| New / keep | Role | Epistemic / repro target |
|------------|------|--------------------------|
| Keep `C_E1_GATE` | KILL under frozen rule | SUPPORTED / decision-repro via offline recompute |
| Keep `C_E1_MEASUREMENT` | Published utilities | SUPPORTED; note cost PARTIAL until L/C closes |
| Add `C_E1_DECISION_REPRO` (optional explicit) | Offline recompute of KILL | PUBLIC_REPRODUCIBLE via pytest |
| Add `C_E1_COST_REPRO` | L/C clean-clone | stays PARTIAL/ABSENT until auth replay |

**Context of use (B16):** task = closed synthetic scribe; \(U\) = frozen E1; world = oracle-strong verifier; venue = RunPod CUDA fp16 for official M0.

## L measurement recipe (historical)

1. Timer: `time.perf_counter()` around `predict_fn` per item in `evaluate_method`.  
2. Aggregate: median of per-item latencies → `L_p50`.  
3. Official generative M0: RunPod CUDA fp16 (see `runpod_official_m0.py` / utility `venue`).  
4. Classical methods: same harness host class for that run.

## C assignment recipe (historical)

`C` is a **relative schedule**, not a bill. Exact values are those in `results_e1_utility.json` rows. Recompute of \(U\) **must** use stored `C`; do not re-infer GPU pricing without a new prereg.

## Clean-clone replay (M2 — NOT AUTHORIZED)

```text
AUTHORIZE_E1_LC_REPLAY = NOT_PRESENT
```

When authorized, a replay job may only:
1. Load published item/method JSONs + timers if archived.  
2. Recompute medians / schedules per this doc.  
3. Diff against published `L_p50`/`C` within stated ε.  
4. Emit `results_e1_lc_reconstruction.json` — **no training**.

## Exit criteria

- Decision path: `pytest trajectory/test_e1_utility_recompute.py` PASS (already).  
- Cost path: reconstruction report PASS within ε **or** permanent split with honest PARTIAL.  
- Never retarget freeze tags to “complete” this protocol.

## Non-goals

No E2/E4 reopen · no weight changes · no clinical claims · no tag moves.
