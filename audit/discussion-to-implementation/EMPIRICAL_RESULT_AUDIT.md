# Empirical Result Audit

*Primary evidence: result JSONs, recompute, preregs. Docs reconciled secondarily.*

## Verification performed (this audit)

1. Listed all `PREREG_*.md` and `results_*.json`.
2. Re-ran `trajectory/recompute_c3.py` successfully against local `outputs_c3_seed*.jsonl`.
3. Confirmed `results_e1_utility.json` decision block: `verdict=KILL`, U_m0=0.925, U_best_nonlm≈0.999.
4. Confirmed E3 auto 0/486; E3 human pack faithful-rate 0.00 with rater `agent-rubric-pass-1`.
5. Confirmed no `results_e2_*.json`.
6. Confirmed RunPod pod list empty (no orphan paid compute visible via `runpodctl`).
7. Confirmed JSONL exists locally but is gitignored.

## Experiment-by-experiment

### Stage T / T-v2 ladder — CLOSED / SUPPORTED (measurement)
- **Observation:** multi-instance diluted gaps nano 18.3±1.3, scale 18.7±1.5, Pythia-160M 3.5±0.7, 410M 4.2±0.9, 1B interval [0,5].
- **Causal license:** none for “scale alone”; stack confound explicit in FINDINGS.
- **Overclaim risk:** citing inst0 as mean (docs warn against this).
- **Artifacts:** `results_arm1_v2_*.json`, `results_anchors_v2_*.json`.

### Field localization — CLOSED / SUPPORTED
- Closed fields ≈0 gap; open fields carry gap (`results_fieldwise_*.json`).

### Own-stack factorial + corner — CLOSED / SUPPORTED (behavioral)
- fullFT 16.9±1.7; LoRA ~7.1; Chinchilla ~7.0; corner 4.24±0.91 seed-stable.
- **Causal license:** adaptation×data interaction as behavior; **not** LoRA geometry.
- **Prereg drift:** `PREREG_ownstack_160m.md` still says not executed.

### Slot diversity — CLOSED / SUPPORTED
- +66.7 held-type recall D5→D80; H-slot SUPPORTED (`results_sweep_10m.json`).
- **Prereg drift:** Status “Nothing has been run”.

### C-1 coverage — SUPERSEDED
- Dry-run refuted; never GPU-band executed as C-1.

### C-1b interference — CLOSED / REFUTED
- `results_interference_10m.json` REFUTED; promoted C-3.
- Raw `outputs_if_seed*.jsonl` local only.

### C-3 binding — CLOSED / MIXED (T/B REFUTED, L UNRESOLVED)
- Primary + independent recompute + A6000 replication agree on polarity.
- Bug `823e1ca`: unstable exclusion not applied pre-fix; **verdicts unchanged**, point estimates were wrong.
- Morphology residual: **exploratory / descriptive**, not confirmed causal.

### Pointer P1/P2 — VOID then REFUTED (this impl)
- P2 manip pass; held gap not closed.

### E1 — CLOSED / KILL (H-substrate)
- Amendment 1 dominance: M1≈0.999 already implied KILL vs any U(M0)≤1 under δ=0.05.
- Official M0 RunPod confirmed margin +0.074.
- Sensitivity: no flip (`results_e1_utility_sensitivity.json`).
- **Not IMPLEMENTED_AND_VERIFIED** under audit bar: no pytest for U; large artifact set untracked.

### E3 auto — CLOSED / EXACT_NOT_OVERSTATING_BY_NORMALIZE
### E3 human — PARTIAL
- Gate language says human; artifact is agent rubric single pass; IAA null.
- Synonym ontology (H4) untested.

### E2 — BLOCKED / NO RESULT
- Runner exists; conflicting RUNNING vs STOP prose; no result file; pods empty.

### E4 / R★ — PREREG ONLY
- Utility/baselines frozen on paper; builder absent; classical probe absent.

### Stage M — PARTIAL CODE / NO MEASUREMENT

## Claim ↔ license table (high level)

| Claim | Observation | Licensed conclusion | Remaining confounds |
|-------|-------------|---------------------|---------------------|
| Held-out copying gap exists | Large own-stack gaps | Proven on exact-match instrument | Construct vs clinic |
| Open-field localization | Closed≈0 | Proven on instrument | — |
| Diversity causal | +66.7 | Causal for behavior under prereg | Mechanism unknown |
| LoRA helps | Factorial deltas | Behavioral Supported | Why unidentified |
| Interference drives residual | C-1b | **Refuted** | — |
| T/B binding factors | C-3 | **Refuted** | L unresolved |
| Generative preferred substrate | E1 U | **Falsified under frozen U** | Task isomorphism; U choice |
| Exact-match overstates via format | E3 auto | **No** | Synonymy open |
| Exact failures clinically trivial | E3 human pack | **Not supported on agent rubric** | Not dual clinician |

## Manuscript sync risks
- Paper α still says human study “pending” after Stage 1 rubric audit.
- Several preregs lack RESULT sections despite JSON.
