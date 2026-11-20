# Project Authority

When sources conflict, use **typed authority** — not one global document stack.

**No aspirational document may overrule measured evidence.**

---

## 1. Scientific / empirical truth

```text
tagged primary artifacts (trajectory/, artifacts/ at freeze tags)
        ↓
RESULT_* / PREREG_* / decision records tied to those artifacts
        ↓
papers/EVIDENCE_LEDGER.md · papers/EMPIRICAL_FOUNDATION.md
        ↓
scientific summaries and manuscripts (papers/)
        ↓
docs/ narrative (must not contradict tagged artifacts)
```

If `docs/` disagrees with a tagged result JSON or ledger entry, **the artifact wins**.

---

## 2. Program mission

```text
explicit owner decision (when issued)
        ↓
docs/PROJECT_CHARTER.md
        ↓
docs/CAPABILITY_LADDER.md
```

Charter defines *what we are building and why* — not whether E1 fired.

---

## 3. Architecture

```text
docs/SYSTEM_ARCHITECTURE.md
        ↓
docs/subsystems/*.md
        ↓
implementation READMEs under component paths
```

---

## 4. Current execution

```text
docs/ACTIVE_NOW.json  (machine-readable, canonical for status fields)
        ↓
docs/ACTIVE_NOW.md    (human mirror — must agree exactly)
        ↓
docs/EXECUTION_PLAN.md
        ↓
frontier/ branch-local notes (non-canonical)
```

---

## 5. Implementation reality

```text
code + tests + artifacts present in this tree
        ↓
documentation claims about what is integrated locally
```

Docs must not claim paths exist on this branch unless marked **cross-branch** (see integration table in [research/MODEL_RESEARCH_PROGRAM.md](research/MODEL_RESEARCH_PROGRAM.md)).

---

## Directory roles

| Path | Role |
|------|------|
| `README.md` | Public overview — points to `docs/` |
| `docs/` | Current program truth (non-evidential) |
| `papers/` | Science — preregistrations, results, manuscripts |
| `trajectory/` | Experimental records |
| `artifacts/` | Machine evidence bundles |
| `frontier/` | Branch notes only — **not** canonical |

## Superseded planning (stubs + browsable archive)

| Stub | Browsable archive |
|------|-------------------|
| `papers/STRATEGIC_RESET.md` | [archive/legacy/STRATEGIC_RESET_20260731.md](archive/legacy/STRATEGIC_RESET_20260731.md) |
| `papers/AMBITION.md` | [archive/legacy/AMBITION_20260731.md](archive/legacy/AMBITION_20260731.md) |
| `papers/WEDGE_V1.md` | [archive/legacy/WEDGE_V1_20260731.md](archive/legacy/WEDGE_V1_20260731.md) |
| `papers/EXECUTION_QUEUE.md` | [archive/legacy/EXECUTION_QUEUE_20260731.md](archive/legacy/EXECUTION_QUEUE_20260731.md) |
| `papers/AZ_EXECUTION_PLAN.md` | [archive/legacy/AZ_EXECUTION_PLAN_POST_E1_20260731.md](archive/legacy/AZ_EXECUTION_PLAN_POST_E1_20260731.md) |
| `papers/PROGRAM_AUTHORITY.md` | [archive/legacy/PROGRAM_AUTHORITY_WEDGE_20260731.md](archive/legacy/PROGRAM_AUTHORITY_WEDGE_20260731.md) |

## Evidence-protected (do not relocate casually)

`papers/EMPIRICAL_FOUNDATION.md`, `papers/EVIDENCE_LEDGER.*`, `PREREG_*`, `RESULT_*`, freeze manifests, tagged `trajectory/results_*.json`.
