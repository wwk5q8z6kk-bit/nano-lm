# Prior art — what is ours, what is not

**2026-08-06.** Adversarial novelty check on the surface-robustness cycle.
Commissioned to be harsh; the result is that **most of what this cycle found is
already established**, and the papers in this directory must be reframed
accordingly. Recorded here so no claim leaves the repo overstated.

## Verdict by finding

| # | our finding | status |
|---|---|---|
| 1 | closed-vocabulary benchmarks confound concept-learning with list-memorisation | **established elsewhere since 2019** |
| 2 | 28%→99% accuracy swing from swapping a denial phrase | **established phenomenon, new data point** |
| 3 | OOD per-arm competence is seed-dependent, Kendall τ ≈ 0 between seeds | **closest to unclaimed** |
| 4 | `surface_robust_accuracy` / `surface_sensitivity` / `seed_instability` | **established individually under other names** |

## What we must cite rather than claim

**Finding 1** is the canonical shortcut-learning result, not a discovery:
- McCoy, Pavlick & Linzen, *Right for the Wrong Reasons* (HANS), ACL 2019 — arXiv:1902.01007
- Ribeiro et al., *CheckList*, ACL 2020 — arXiv:2005.04118 (its Minimum Functionality Tests target this exact failure, negation templates included)
- Gardner et al., *Contrast Sets*, Findings of EMNLP 2020 — arXiv:2004.02709
- Geirhos et al., *Shortcut Learning*, Nature MI 2020 — arXiv:2004.07780
- Sclar et al., ICLR 2024 — arXiv:2310.11324 (up to 76-point swings from prompt format alone)

**Finding 4**'s metrics already exist:
- `surface_robust_accuracy` ≈ **worst-group accuracy**, GroupDRO — Sagawa et al. 2020, arXiv:1911.08731
- `surface_sensitivity` ≈ **Format Sensitivity Index** / robustness gap; SCORE (NAACL-industry 2025, arXiv:2503.00137) already reports min/max across 10 semantically-equivalent prompts
- `seed_instability` ≈ **prediction churn** — Milani Fard et al., NeurIPS 2016

A reviewer will ask how this differs from GroupDRO + SCORE. The honest answer is
that the *packaging* is new (three numbers replacing one held-out score, applied
to training-time surface arms rather than inference-time prompt variants) and the
*mathematics* is not.

**Finding 3** is the strongest candidate, and even it has a parent:
- D'Amour et al., *Underspecification Presents Challenges for Credibility in Modern Machine Learning*, arXiv:2011.03395 (2020) / JMLR 2022 — already demonstrates that seed-only changes produce arbitrarily different behaviour under distribution shift, **including in clinical settings**.
- arXiv:2503.07329 (2025) measures per-instance seed consistency but does **not** split ID/OOD and does **not** use rank correlation.

So the unclaimed slice is narrow and specific: **per-arm accuracy decomposed
into ID (mean |Δ| 2.5 pts) vs OOD (31.5 pts), with Kendall τ = 0.00 between two
seeds' rankings of which phrasing is easiest.** That is *a new measurement of
underspecification*, not a discovery that underspecification exists.

## The genuine domain gap

Clinical assertion/negation SOTA 2024–2026 reports headline accuracy (0.90–0.96)
and **essentially never reports surface robustness or seed variance**:
- arXiv:2503.17425 (ECIR 2025) — fine-tuned LLM 96.2% vs GPT-4o 90.1%
- arXiv:2606.18471 — *Possible or Definite?* diagnostic-uncertainty benchmark
- arXiv:2605.30646 — *Same Patient, Different Words, Different Diagnosis?*

**Two of these must be read in full before anything is written up.**
arXiv:2605.30646 is methodologically the closest published work to our
instrument — it holds documents fixed and varies clinical wording — and
arXiv:2606.18471 is the nearest neighbour on epistemic states. The novelty check
could only partially extract both; neither is ruled out as closer prior art.

## Consequence for this repo

The correct positioning is **not** "we discovered that surface vocabulary
confounds evaluation." It is:

> A well-established failure mode from general NLP — never measured in clinical
> epistemic-state extraction — is severe there, and it silently misdiagnosed two
> model generations in a real research programme. Here is the instrument, the
> per-state factorial, and the seed-instability measurement that reframes it.

That is a legitimate contribution. It is a domain-transfer and
instrumentation contribution, not a conceptual one, and the papers should say so.

## Action taken

`RESULT_DP1_AND_THE_VOCABULARY_CEILING.md`, `RESULT_SURFACE_HARNESS_RUN1.md`,
`RESULT_PER_STATE_DIAGNOSIS.md` and `nano_ai/surface.py` all currently read as
if these findings originate here. Each should carry a pointer to this file.
Done below.

*Caveat from the check itself: several PDF fetches returned degraded text
(contrast sets, NADE, ClinicalBench, arXiv:2606.18471). Those are "not ruled out
as closer prior art", not "confirmed distant."*
