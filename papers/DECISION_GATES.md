# Decision Gates

**Operating doc — promotion rules from dream → design → execution.**  
**Adopted:** 2026-07-31  
**Constitution:** `LABORATORY_CONSTITUTION.md`

## Principle

```text
Portfolio question  →  Design protocol  →  Measured evidence  →  Execution queue
     (Layer 3)            (docs)             (Layer 1)            (build/run)
```

Skipping layers is forbidden. Ambition may exist at Layer 2/3 without Layer 1 support.

## Gate G0 — Speak carefully (always on)

- Use `CLAIM_GLOSSARY.md`.  
- Ledger statuses only for measured claims.  
- Roadmap language must say “aspirational / conditional.”

## Gate G1 — Open a research program (portfolio)

**Pass if:** clear question; falsifiable eventual experiment class; no product claim required.  
**Fail if:** question is only “build NanoScribe.”  
**Output:** entry in `RESEARCH_PORTFOLIO.md`.

## Gate G2 — Design-only protocol

**Pass if:** owner auth for design (e.g. `AUTHORIZE_E4_DESIGN_ONLY`); utility/fairness/admissibility draft; revision budget stated.  
**Fail if:** design implies world freeze or GPU.  
**Output:** prereg/design markdown; status `DESIGN_DRAFT`.

## Gate G3 — Execute measurement

**Pass if:** separate execute auth (e.g. `AUTHORIZE_E4_EXECUTE`); frozen \(U\); locked data hashes; information parity; compute budget.  
**Fail if:** only design auth exists.  
**Output:** RESULT artifacts → Evidence Ledger update.

## Gate G4 — Promote into Execution Queue (build)

**Pass if:**  
1. Ledger shows supportive evidence for the *specific* wedge; and  
2. Feasibility note (cost, maintenance, safety); and  
3. Owner build auth; and  
4. Regression suite includes relevant failure modes.  

**Fail if:** promotion is justified only by Technology Roadmap aspiration or by SURVIVE without feasibility gate.

## Gate G5 — Product / architecture expansion

**Pass if:** repeated wedge wins across tasks/regimes; still no open-world overclaim; explicit product auth.  
**Default:** **STOP** for NanoScribe-scale expansion.

## Special note on negative evidence

| Event | Effect on Layer 1 | Effect on Layer 2/3 |
|-------|-------------------|---------------------|
| E1 KILL | Product thesis falsified for scoped \(U\)/task | Portfolio Program H remains; Roadmap hybrid/classical rows remain |
| E2 STOP | No mechanism claim | Program A remains open |
| E3 agent-rubric | Exact construct not clinically closed | Program F remains open |
| E4 KILL (future) | Stop gen-substrate for that R★ (≤1 revision) | Portfolio does not collapse |

Negative evidence **narrows claims**. It does **not** delete the laboratory’s right to ask bigger questions.
