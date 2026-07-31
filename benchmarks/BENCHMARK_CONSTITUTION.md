# Evaluation infrastructure (inside nano-lm)

**Was labeled:** Benchmark Supremacy Lab — **reframe:** supporting evaluation harness for the nano-lm research vehicle, not the project identity.

**Subtitle:** Pareto-aware, contamination-resistant evaluation infrastructure.

> **Parent:** [`papers/LABORATORY_CONSTITUTION.md`](../papers/LABORATORY_CONSTITUTION.md) — research governance for **this repo**. This file is the Program 0 / harness charter for nano-lm’s benchmark infrastructure only.

**Adopted:** 2026-07-31  
**Authorized unit:** `BENCHMARK_SUPREMACY_LAB_PROGRAM0_INFRA`  
**Status:** Infrastructure only (not a claim of leadership)  
**Layer:** Research portfolio / technology roadmap companion — **not** Layer-1 evidence

```text
MISSION = COMPREHENSIVE_BENCHMARK_SUPREMACY
METHOD = FIRST_PRINCIPLES_RESEARCH_AND_ENGINEERING
EVALUATION = PUBLIC + HIDDEN + DYNAMIC + OOD + PRODUCT
PARETO_ROLE = FAIRNESS_AND_SEARCH_METHOD
PROGRAM0 = INFRA_COMPLETE_STOP_EXPANSION
PROGRAM1 = NOT_AUTHORIZED
TRAINING = NOT_AUTHORIZED
E4_RESULT = KILL
LAYER1_FREEZE = UNTOUCHED
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_DOGFOOD
AUTHORIZED_NONEXECUTION_WORK: NONE
BENCHMARK_LAB_STATUS: INFRASTRUCTURE_ONLY
PARENT_POSTURE: see papers/LABORATORY_CONSTITUTION.md
```

## Dual mandate (both required)

### 1. Ultimate ambition (aspirational — not achieved)

> Lead every **relevant** benchmark through first-principles research and engineering,
> under reproducible and fair conditions, on public, hidden, dynamic, OOD, and product worlds.

### 2. Evaluation discipline

Report **absolute** and **resource-conditioned** performance. Never hide trade-offs in one
unweighted mean. Maintain boards for:

- absolute score;
- score / parameter;
- score / training FLOP;
- score / dollar;
- score / joule;
- robustness;
- reliability at fixed coverage;
- quality under fixed latency.

**Pareto analysis is the measurement method, not the ambition ceiling.**  
Finding a niche does not satisfy the supremacy mission. Claiming supremacy without
hidden/OOD/product agreement and contamination review is forbidden.

## Separation of records

```text
Benchmark result  ≠  evidence-ledger claim  ≠  product authorization
```

See `BENCHMARK_RESULT_POLICY.md`.

## Freeze relationship

The post-α evidence freeze is the **foundation, not the ceiling**. This lab must not
modify:

- `papers/EVIDENCE_LEDGER.md` or ledger JSON;
- `artifacts/` freeze manifests / `SHA256SUMS`;
- tags `paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31`;
- E1/E2/E3/E4 result status;
- Fabric scientific claims.

E1 KILL remains a **sentinel and architectural constraint** (classical baselines
mandatory on extraction-like tasks; generation is not the default solver). It is
**not** the new north star. Use every substrate necessary to win *where justified*.

## Living target

A genuine win for benchmark \(b\) requires public ≈ hidden ≈ dynamic ≈ OOD ≈ product
scores within preregistered tolerances, with contamination risk below \(\epsilon_b\).

## Pins (identity)

Every countable run binds:

- `lm-eval` package version **and** git commit;
- task name, task version, task YAML hash;
- dataset / instrument hash, record count, schema version;
- prompt/template hash, metric implementation hash, filter-pipeline hash;
- model or solver manifest hash;
- code git commit;
- content-addressed `run_id`.

## Attack loop

Observe → failure tensor → competing hypotheses → smallest discriminating experiment →
invention when needed → validate (seeds, matched compute, hidden/OOD, contamination) →
promote / revise-once / kill → repeat.

## Bench gates (0–8)

0 truth → 1 localize (≥80%) → 2 construct → 3 pilot → 4 replicate → 5 OOD → 6 full suite →
7 product transfer → 8 public claim.

Program 0 ends only in `INFRA_SMOKE_PASS` or `INFRA_SMOKE_FAIL`. Never `PROMOTE`.

## Tracks (portfolio stubs)

Data · Tokenizer · Architecture · Pretraining · Optimization · Post-training · Reasoning ·
Retrieval · Verification · Memory · Agents · Multimodal · Systems.

## Program 0 scope

Docs, registry, schemas, adapter split, held-value sentinel smoke, tests.  
**Not:** training, paid compute, E4 execute, Program 1 census, MMLU/HELM/MLPerf, leaderboard
promotion, ledger edits.


## Program 0 six properties (must all pass)

1. **Identity** — suite/task/dataset/model|solver/scorer/config/code content-bound.
2. **Reproducibility** — equivalent inputs → same run identity and result.
3. **Auditability** — every aggregate score traces to per-item outputs.
4. **Isolation** — infrastructure cannot silently amend Layer-1 evidence.
5. **Failure visibility** — FAILED/PARTIAL/VOID/CANCELLED runs preserved.
6. **Extensibility** — Program 1 can add models/tasks without rewriting the provenance contract.

Terminal decisions only: `INFRA_SMOKE_PASS` | `INFRA_SMOKE_FAIL`.  
Never `PROMOTE`. `leaderboard_eligible=false`, `evidence_ledger_eligible=false`, `program1_authorized=false`.

## Gate 0 (Program 1 authorization prerequisite)

Program 1 (world census) may be proposed only if all rows PASS:

```text
REGISTRY_SCHEMA
TASK_VALIDATION
SOURCE_DIGEST_BINDING
PER_ITEM_TRACEABILITY
REPEATED_RUN_REPRODUCIBILITY
FAILED_RUN_PRESERVATION
CLEAN_CLONE_EXECUTION
LAYER1_BOUNDARY
TAG_INTEGRITY
DEFAULT_TEST_SUITE
```

Any FAIL → `PROGRAM0_STATUS=INFRA_SMOKE_FAIL`, `PROGRAM1_STATUS=BLOCKED`.

See `benchmarks/reports/GATE0_PROGRAM0.md`.
