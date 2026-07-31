# Laboratory Constitution

**Scope:** Research governance **inside the `nano-lm` repository** — not a separate institution, company, or multi-lab organization.  
**Adopted:** 2026-07-31  
**Does not:** authorize unpaid compute, amend Layer-1 evidence, or invent product platforms.

```text
PROJECT: nano-lm
PROGRAM_EXECUTION_STATUS: IDLE_AFTER_FREEZE
AUTHORIZED_NOW: E4_DESIGN_ONLY | BENCHMARK_SUPREMACY_LAB_PROGRAM0_INFRA
PROGRAM1: DEFERRED_PENDING_GATE0
TRAINING: NOT_AUTHORIZED
E4_EXECUTION: BLOCKED
LAYER1_FREEZE: UNTOUCHED
NANOSCRIBE_PRODUCT_EXPANSION: STOP
OLD_TASK_RUNS_UNDER_OLD_TASK_U: FORBIDDEN
EVIDENCE_STANDARDS: CONSERVATIVE
VISION_STANDARDS: EXPANSIVE   # ambition allowed in portfolio/roadmap docs only
```

---

## What this project is

`nano-lm` currently contains four real surfaces:

1. **Empirical research** — held-out copying, Paper α, E1–E4 gates, trajectory results.  
2. **Evidence packaging** — Evidence Ledger, freeze manifests, claim discipline.  
3. **Verification fabric** — scoped propose→verify→abstain slice (`fabric/`); not NanoScribe / not a cognitive OS.  
4. **Benchmark infrastructure** — Benchmark Supremacy Program **0 only** (`benchmarks/`); a trustworthy harness for this repo, not a new product.

Do **not** invent additional “labs” (Discovery, Theory, AI Scientist, Hardware, Product, …). Portfolio questions may exist as bullets in `RESEARCH_PORTFOLIO.md`; they are not organizations and are not authorized work.

---

## Separation of concerns (keep these mixed things apart)

| Concern | Question | Document | Standard |
|---------|----------|----------|----------|
| **Truth (Layer 1)** | What have we demonstrated? | `EVIDENCE_LEDGER.md`, Paper α, freeze `artifacts/` | Conservative |
| **Ambition (Layer 2/3)** | What might we ask / build someday? | `RESEARCH_PORTFOLIO.md`, `TECHNOLOGY_ROADMAP.md` | Expansive; non-evidential |
| **Execution** | What is authorized *now*? | `EXECUTION_QUEUE.md` | Tiny; gate-bound |
| **Promotion rules** | How does an idea earn a run or claim? | `DECISION_GATES.md` | Strict |
| **Benchmarks** | How do we score reproducibly? | `benchmarks/` + `BENCHMARK_RESULT_POLICY.md` | Infra ≠ ledger |
| **Products** | What may ship? | Default **STOP** until gated | Not the organizer of the repo |

```text
Benchmark result  ≠  evidence-ledger claim  ≠  product authorization
```

**Constitutional rule:** Negative evidence kills **hypotheses**, not **curiosity** — but curiosity does not create fake org charts or queue entries.

**Post-E1 rule:** Generation is not the default solver. Prefer cheapest sufficient solver → verify → abstain → provenance. E1 KILL applies to the old closed scribe task under frozen \(U\); it does not license NanoScribe expansion.

---

## How research is conducted in this repository

### Hypotheses

Write competing explanations in preregs or `research/hypotheses/` when useful. Prefer questions that a small experiment can kill.

### Preregistration

Freeze: intervention, controls, primary estimand, success/kill thresholds, cost ceiling — before the load-bearing run.

### Evidence classification

Only Layer-1 documents and freeze artifacts may claim measured truth. Use `CLAIM_GLOSSARY.md`. Leaderboard rows and Program 0 smokes do **not** enter the Evidence Ledger automatically.

### Benchmark reproduction

Pin harness version + commit, task YAML hash, source instrument digest, model/solver manifest, config hash, code commit. Content-addressed `run_id`. Per-item logs required. See `benchmarks/BENCHMARK_CONSTITUTION.md` (Program 0 charter for this repo’s harness).

### From result to claim

```text
Prereg → authorized run → audit (hashes, recompute, contamination as needed)
  → optional Layer-1 ledger update (owner + gates)
  → optional paper claim
```

### Authorization of implementation

Portfolio/roadmap items do **not** auto-enter the queue. Only `DECISION_GATES.md` + explicit owner auth strings on `EXECUTION_QUEUE.md`.

---

## Document authority

| Doc | Measured truth? | Future systems? | Authorizes run? |
|-----|-----------------|-----------------|-----------------|
| `EVIDENCE_LEDGER.md` | Yes (scoped) | No | No |
| `RESEARCH_PORTFOLIO.md` | No | Questions only | No |
| `TECHNOLOGY_ROADMAP.md` | No | Conditional | No |
| `EXECUTION_QUEUE.md` | Cite ledger only | No | **Yes** |
| `DECISION_GATES.md` | Process | No | Defines promotion |
| `benchmarks/BENCHMARK_CONSTITUTION.md` | No | Harness rules | No |
| `MASTER_PLAN.md` / `NANOSCRIBE_VNEXT.md` | Historical | Legacy | **No** |

---

## Anti-contamination

1. Do not put roadmap modules into the Evidence Ledger until measured.  
2. Do not treat empty queue as “stop thinking,” or full portfolio as “start building.”  
3. Do not move immutable evidence tags (`paper-alpha-v1`, `post-alpha-evidence-freeze-2026-07-31`).  
4. Do not expand Fabric/NanoScribe/memory/agents/product work without queue auth.  
5. Do not grow organizational scaffolding instead of finishing authorized programs.

---

## Authorized work now

| Item | Auth | Notes |
|------|------|-------|
| Freeze integrity | Standing | Do not rewrite Layer-1 / tags |
| E4 design docs only | `AUTHORIZE_E4_DESIGN_ONLY` | No world/data/GPU/result |
| Benchmark Program 0 | `BENCHMARK_SUPREMACY_LAB_PROGRAM0_INFRA` | One reproducible path; Gate 0; **no** Program 1 census |

**Program 0 purpose:** one task → one solver/model → one reproducible run → per-item outputs → deterministic score → manifests/hashes → Gate 0.  
Not a laboratory empire.

**Explicitly not authorized:** Program 1 world census; training; E4 execute; NanoScribe; autonomous scientist; distributed agents; long-term memory product; multi-lab org design; large suites (MMLU/HELM/MLPerf).

Gate 0 report: `benchmarks/reports/GATE0_PROGRAM0.md`.
