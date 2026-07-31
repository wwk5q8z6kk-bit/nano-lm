# Research Program — boundary conditions for reliable held-out copying

*Baseline frozen 2026-07-30 by Scientific Research Council audit (accepted).
Operating lockfile for priorities and claim discipline:
`papers/EMPIRICAL_FOUNDATION.md`. This file is the organizing frame; every experiment
is a consequence of that lockfile, not the other way around.*

## Surviving contribution (interpretation lock)

**Not claimed:** "Transformers cannot perform deterministic extraction."

**Claimed:** Under low-diversity extraction regimes, small transformers can converge
to closed-set prediction strategies that fail held-out symbolic emission; diversity,
adaptation regime, and deterministic verification change the reliability profile.

Roadmap goal: isolate **boundary conditions** under which each approach works —
not prove a broad architectural thesis.

## Claim discipline (non-negotiable)

1. **No mechanism claim beyond measured evidence.** Behavioral deltas are facts;
   circuit/pathway stories need their own preregistered tests (E2 for LoRA
   universes; Stage M only after E1–E3).
2. **Morphology remains descriptive** until a preregistered causal analysis
   (C-3 surfaced morphological re-inflection as the dominant miss pattern; causal
   status open).
3. **Pointer results** = "this implementation does not close the OOD gap," not
   evidence against all copy mechanisms (`scribe/pointer/`).
4. **Verification claims** are scoped to the verifier relation \(R\) and the
   measured distribution — not open-world hallucination elimination. Fabric is a
   regression harness until E1–E3 land.
5. **Science object ⊥ systems object** in claims (Paper 1 vs Paper 2).

## Mission

Measure when and why small LMs fail held-out *value copying* on a structured
extraction task, and characterize which interventions (diversity, adaptation,
data scale, deterministic verification) change the reliability profile — with
pre-registered decision rules and immutable JSON artifacts.

## Measured foundation (grounded in `trajectory/` JSONs)

| Fact | Status | Artifact anchors |
|---|---|---|
| Field localization (open-vocab only; closed fields = 0) | Established | anchors / fieldwise JSONs |
| Own-stack full-FT configs: no monotonic diluted-gap collapse with N (18.3 → 18.7 → 16.9; unequal token budgets) | Supported (descriptive) | `pretrain/AUDIT.md`; `scale/AUDIT.md`; `results_ownstack_v2_160m_fullft.json` |
| Stack-dominant vs Pythia at matched size | Established | fullft vs `results_arm1_v2_pythia-160m.json` |
| Weak-base × full-FT interaction; LoRA / Chinchilla substitutes | Established | lora / chinchilla JSONs |
| Factorial corner 3.2B+LoRA → 4.2±0.9 (≈Pythia); seed \|Δ\|=0.00 | **Established** | `results_corner_3p2b_lora_seed{0,1}.json` |
| Slot diversity causal (+66.7 pts, H-slot SUPPORTED) | Established | `results_sweep_10m.json` / sweep_eval/ |
| Lexical interference (C-1b) | **REFUTED** | `results_interference_10m.json` |
| C-3 T/B/L mechanistic factors | T/B **REFUTED**; L UNRESOLVED | `results_c3_10m.json` |
| Morphology residual | Descriptive only | C-3 error census |
| Fabric presented-error → 0 under rules-strong \(R\) | Existence proof (scoped) | `fabric/results_slice_v1.json` |
| Pointer/copy head (supervised) | H-copy **REFUTED** for this impl | `scribe/pointer/` |
| E1 non-LM utility kill-gate | **KILL** (H-substrate); ecology general | `results_e1_utility.json` |
| E3 normalize construct | Auto: exact **not** overstated (0 rescues); agent-rubric 0/100; dual-clinician open | `results_e3_normalize_construct.json`; `results_e3_human.json` |
| E2 LoRA universes | **GATED / STOP** (no RESULT) | `PREREG_E2_lora_universes.md` |

## Paper split (mandatory)

| Paper | Scope | Must not claim |
|---|---|---|
| **Paper 1 (α) — measurement** | Held-out copying failures; field localization; diversity effects; scaling/adaptation interactions; reproducible benchmark/instrument | Architecture thesis; fabric-as-product; LoRA "geometry preservation"; mechanism-solved language |
| **Paper 2 (β) — verification** | Deterministic verification framework; soundness conditions under decidable \(R\); abstention/review economics; adversarial verifier evaluation | Transformer mechanism dependency; open-world zero-hallucination |

Manuscript map:

- Paper 1 core: `papers/latex/paper1.tex` / `papers/paper1_draft.md`
- Paper 1 empirical companion (within-stack / factorial / diversity / Phase C):
  historically drafted as `papers/paper2_draft.md` — **reclassified as Paper-1
  measurement extension** under this baseline (filename retained for git history)
- Paper 2 (verification systems): Stage G/A + `fabric/` evidence; draft still to
  be written under β scope — do not sell as mechanism paper

Legacy P2/P3/P4/P5 labels in older notes meant "causality / mechanism / generality /
theory." Those are **retired as paper IDs**. Mechanism and generality remain
research stages gated by E1–E3, not Paper-2 content.

## Kill-gate priority order

Kill-gate status (2026-07-31 freeze):

1. **E1** — **KILL** (`trajectory/PREREG_E1_nonlm_baseline.md`, `results_e1_utility.json`).
   Generative nano core optional; α+β must be substrate-agnostic in product claims.
2. **E3** — auto arm **EXACT_NOT_OVERSTATING_BY_NORMALIZE**; agent-rubric
   **EXACT_SURVIVES**; dual-clinician/IAA open
   (`trajectory/PREREG_E3_faithfulness_construct.md`).
3. **E2** — prereg frozen, **GATED / STOP** (no RESULT)
   (`trajectory/PREREG_E2_lora_universes.md`).

Fabric/v2 / NanoScribe architecture expansion remains **STOP**. R★/E4 are under
**`AUTHORIZE_E4_DESIGN_ONLY`**: E4 = `DESIGN_IN_PROGRESS` / `EXECUTION_BLOCKED`
(not “next stage running”). Evidence posture remains **`IDLE_AFTER_FREEZE`**
(IDLE ≠ halted — see `papers/AMBITION.md`). Details:
`papers/EMPIRICAL_FOUNDATION.md`,
`audit/discussion-to-implementation/WITHDRAWAL_SPEC.md`.

## Null hypotheses (status)

- **H0c / H0d** (eval artifact / measurement error): substantially rejected —
  multi-instance instrument and clean metric sharpened the phenomenon.
- **H0b** (stack / recipe, not raw param count alone): **SUPPORTED** within this
  recipe — own-stack 160M full-FT flat; corner closes most of the cross-stack diluted
  gap via data×method.
- **H0a** (gap independent of size within-stack): supported only as a **descriptive**
  observation under evaluated recipes (no monotonic collapse); parameter count was
  not isolated from pretraining exposure (nano 32.8M vs ~200M / 3.2B).

Caveats remain: seed/factorial underpower on some cells; LoRA mechanism unidentified;
morphology descriptive; external validity of synthetic world open; E1 KILL demotes LM-product frame; E3 dual-clinician pending.

## Measurement principles

1. Consistency — one instrument across rungs.
2. Reproducibility — content-addressed inputs, frozen tags, byte-exact re-score checks.
3. Power — mean ± across-instance SD; multi-instance over single hard draws.
4. Calibration — clean vs diluted; template-vs-value control.
5. Independent validation — adversarial audit + fresh-eyes review before claims ship.
6. Claim discipline — see above; science ⊥ systems.

## Success ladder (recalibrated)

- **L1 — robust observation.** ✅ (gap; field localization; instrument)
- **L2 — empirical laws under boundary conditions.** ~here (diversity causality;
  descriptive own-stack scale/config comparison; adaptation×data interaction; shared residual floor)
- **L3 — substrate / construct validation.** E1 **KILL** · E3 auto + agent-rubric done · dual-clinician/IAA pending
- **L4 — adaptation-mechanism discrimination.** E2 GATED/STOP (no RESULT)
- **L5 — predictive theory of reliability regimes.** destination, after L3–L4

## Ambition (post-freeze)

`IDLE_AFTER_FREEZE` completes evidence packaging; it does **not** end the program.
Ambition: find whether ∃ R★ with \(U_{\mathrm{gen+verify}}(R★) > U_{\mathrm{classical}}(R★)\)
under matched Q/E/R/L/C/M — **not** “build NanoScribe anyway.” Authorized work:
E4 **design** (`REGIME_P1`, `PREREG_E4`, `AMBITION.md`). Execution requires separate
owner authorization. See `papers/AMBITION.md`.

## Resource allocation (post-freeze)

≈ **50% E4 design hardening (regime / utility / fairness / consequences) ·
30% writing / claim hygiene · 20% reproducibility / packaging.** E1/E3 measurement
closed for the freeze. Fabric / NanoScribe / E2 / E4 **execution** remain **out of
budget** until separate owner auth.

## Pointers

- Lockfile / kill-gates: `papers/EMPIRICAL_FOUNDATION.md`
- Evidence map / status: `papers/EVIDENCE_MANIFEST.json`,
  `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`
- Soft-claim withdrawals: `audit/discussion-to-implementation/WITHDRAWAL_SPEC.md`
- Master plan (with council override): `papers/MASTER_PLAN.md`
- Artifacts + env + CI: `trajectory/REPRODUCIBILITY.md`, root `README.md`
- E1 prereg + results: `trajectory/PREREG_E1_nonlm_baseline.md`, `results_e1_utility.json`
- E3 prereg + results: `trajectory/PREREG_E3_faithfulness_construct.md`
- E2 prereg (gated/stop): `trajectory/PREREG_E2_lora_universes.md`

## One-sentence annual target

> We measure reproducible boundary conditions under which small transformers fail
> held-out symbolic emission, and show how diversity, adaptation regime, and
> deterministic verification change the reliability profile on this instrument —
> without treating untested regimes or architecture sketches as next product stages.
