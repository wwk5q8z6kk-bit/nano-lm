# Evaluation Contract & Contamination Safeguards

This is part of the [November 17 research proposal](FOUNDATION_INTELLIGENCE_PROGRAM.md). Its size limits, experiments and reported scores belong to that proposal. The [current charter](../PROJECT_CHARTER.md) defines NanoLM’s scope.

Operates under [EVALUATION_FRAMEWORK.md](../EVALUATION_FRAMEWORK.md). Resolves the historical gap between 42-item smoke diagnostics (`suite_15pillar.py`) and defensible foundation-model evaluation.

---

## 1. Multi-Tier Evaluation Hierarchy

1. **Internal 42-Item Diagnostic (`CROSS_BRANCH` `nanolm/arch-evolution-v3:nanolm/eval/suite_15pillar.py`):** Preserved as an immutable zero-LLM-judge unit/regression diagnostic (`SHA-256 db5fbc68...`). Because `N=42` (`3` items/pillar) has wide binomial uncertainty (e.g. `Qwen2.5-1.5B` `45.24%` `[28.57%, 59.52%]`), it **cannot** rank competing 2B architectures or establish general intelligence.
2. **Tier-2 Foundation Benchmarks (`Track 1` — Naked Weights):** Hundreds to thousands of held-out executable items per pillar across:
   1. Coding (`LiveCodeBench`/`BigCodeBench`/unseen repo debugging, executed `pass@1`, cheat screening)
   2. Mathematics & logic (exact arithmetic, symbolic algebra, competition math, algorithmic puzzles)
   3. Summarization (longitudinal fact tables, exception preservation, atomic claim P/R)
   4. Structured extraction (strict schema, byte-exact span anchoring, state F1)
   5. Diagrams & charts (graph topology exactness, Mermaid/SVG validity, chart conservation)
   6. Long-context (`RULER` multi-needle `1..32` needles at `8k–128k`, distractor density, state updates)
   7. General knowledge (held-out factual QA, recency/supersession, contradictory context)
   8. Multi-step reasoning (compositional 4–8 step plans and unseen task graphs)
   9. Instruction following (`IFEval`-style verifiable structural/priority/abstain constraints)
   10. Tool use (schema validity, missing-arg refusal vs hallucination, multi-turn error recovery)
   11. Planning (resource-constrained DAG/graph scheduling, simulator worlds)
   12. Robustness / OOD (unseen templates, novel symbols, multilingual drift, distractors)
   13. Hallucination resistance (false premises, missing evidence, selective risk vs coverage)
   14. Memory retention (temporal corrections, identity swaps, interference, verifiable deletion)
   15. Efficiency (`CPU`/`MPS`/`CUDA` cold/warm prefill & decode at `512/4k/32k/128k`, peak bytes, joules/verified task)
3. **Domain Holdout Stress (`Track 2` — System + DomainPacks):**
   - Medical P1 (`NanoScribe`, fresh blinded cohorts + external clinician dual-IAA before any clinical claim), P2/P3 interface contracts;
   - Non-medical domain transfer (`X16`): legal/financial source-based QA, software incident diagnosis, scientific literature attribution, causal world simulators.

---

## 2. Mandatory Evaluator Negative Controls (`X01`)

To prevent any repeat of the `H7` all-missing gold transport defect (`INVALID_GOLD_ANNOTATIONS`), every structured/state-span evaluator **must pass all six automated controls before any model runs**:
1. **Gold Oracle:** Feeding exact gold annotations yields `1.0` state and grounded-exact accuracy (**PASS**).
2. **All-Missing Negative:** Feeding all-`MISSING` predictions fails admission whenever gold contains non-missing assertions, and quota mismatch raises `ValueError` (**FAIL**).
3. **Shuffled-Labels Negative:** Permuting field/case predictions drops accuracy strictly below oracle and fails admission (**FAIL**).
4. **Wrong-Patient Negative:** Swapping transcripts across distinct encounter worlds fails span-grounding / contract checks (**FAIL**).
5. **Corrupted-Offset Negative:** Shifting evidence offsets by `+1` or placing spans inside `Doctor:` turns raises `ContractValidationError` or scores `0` grounded exact (**FAIL**).
6. **Denial / Conflict / Abstain Controls:** Verifies `ABSENT` $\neq$ `MISSING` $\neq$ `UNCERTAIN`, `CONFLICTING` rejects single-span collapse, and an empty/irrelevant transcript yields abstention with zero fabricated assertions (**PASS**).

---

## 3. Split Integrity, Decontamination & Statistical Protocol

- **Group-aware splitting:** Partition strictly by source document, synthetic world ID, patient ID, or repository — never by derived prompt record.
- **Multi-signal decontamination:** Run exact, fuzzy/MinHash, line/paragraph, normalized entity/numeric value, and template-genealogy checks (13-gram overlap = 0 alone is insufficient).
- **One-shot cohort hygiene:** Consumed screens (`H6` `cb612410…`, `H7` `9b18c5ee…`, exposed `C12–C14` cohorts) remain permanently frozen and cannot be re-promoted as fresh confirmatory tests.
- **Uncertainty reporting:** Small-scale experiments (`L0`–`L1`) require $\ge 3$ independent seeds (mean $\pm$ std / bootstrap CI accounting for world-level clustering). Every table discloses exact denominator $N$, coverage, failure rate, energy, and wall-clock time.

---

## 4. Promotion Gates (`T0`–`T6`)

| Gate | Requirement |
|---|---|
| **T0 — Correctness** | All 6 evaluator negative controls pass; zero syntax, transport, or annotation corruption. |
| **T1 — Mechanistic Validity** | Intended mathematical property verified with $\ge 3$ seeds or deterministic proof; parallel-vs-recurrent numeric parity verified. |
| **T2 — Matched Quality** | Statistically supported improvement over strongest matched-budget baseline (total params, optimizer, tokens, FLOPs). |
| **T3 — Cross-Domain Transfer** | Transfers to $\ge 3$ distinct held-out non-prototype families without regressing grounding, instruction following, or OOD calibration. |
| **T4 — Systems Pareto** | Actual target-device (`MPS`/`CUDA`) peak memory, `p95` latency, KV+recurrent+evidence bytes, and energy/task Pareto verified across `512..32k+`. |
| **T5 — Safety** | Zero silent source mutation, denial-as-absence confusion, cross-subject leak, or unverified commit. |
| **T6 — Publication** | Independent replication run + reviewer sign-off, immutable content-addressed artifacts, explicit scope and negative results. |
