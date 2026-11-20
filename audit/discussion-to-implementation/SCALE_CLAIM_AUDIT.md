# Scale Claim Audit

*Generated 2026-07-31T14:01:32.226455+00:00*

Authoritative token budgets: `TOKEN_BUDGET_RECONCILIATION.md`.

| ID | Class | File:line | Must change |
|----|-------|-----------|-------------|
| SC-P1-ABS-FLAT50 | `UNSUPPORTED` | `papers/paper1_draft.md:41` | abstract, conclusion, §7.2, ledger, paper2, foundation, research_program |
| SC-P1-METHODS-200M | `FALSE_METHODS_STATEMENT` | `papers/paper1_draft.md:295` | methods, latex methods, token table if any |
| SC-P1-SEC72-FLAT50 | `UNSUPPORTED` | `papers/paper1_draft.md:494` | §7.2, latex withinstack |
| SC-P1-CONCL-50X | `SUPPORTED_WITH_SCOPE_CHANGE` | `papers/paper1_draft.md:654` | conclusion, latex conclusion |
| SC-P2-FLAT50 | `UNSUPPORTED` | `papers/paper2_draft.md:30` | abstract, §3.1 title/body |
| SC-P2-HELD-IDENTICAL | `FALSE_METHODS_STATEMENT` | `papers/paper2_draft.md:80` | §2 design |
| SC-LEDGER-SCALE | `SUPPORTED_WITH_SCOPE_CHANGE` | `papers/EVIDENCE_LEDGER.md:27` | ledger |
| SC-RP-H0A | `DESCRIPTIVE_ONLY` | `papers/RESEARCH_PROGRAM.md:102` | research_program, empirical_foundation |
| SC-TEX-METHODS | `FALSE_METHODS_STATEMENT` | `papers/latex/paper1.tex:134` | latex |
| SC-P1-E1-PLURAL | `UNSUPPORTED` | `papers/paper1_draft.md:45` | abstract, §0 title/body, conclusion, foundation |

## Detail

### SC-P1-ABS-FLAT50 — `UNSUPPORTED`

- **Current wording:** diluted gap still 16.9±1.7 across ~50× of own-stack scale (flat)
- **Location:** `papers/paper1_draft.md:41`
- **Data required:** Matched pretraining tokens across 3.15M/10M/160M OR explicit descriptive-only estimand
- **Data available:** 3.15M=32.8M tok; 10M≈200M; 160M fullFT=200M; gaps 18.3/18.7/16.9
- **Confound:** Parameter count entangled with total pretraining tokens and tokens/param
- **Corrected wording:** Across evaluated own-stack configs, diluted gap did not decrease monotonically with parameter count; parameter count not isolated from pretraining exposure → descriptive, not parameter-only scale law.
- **Must change:** abstract, conclusion, §7.2, ledger, paper2, foundation, research_program

### SC-P1-METHODS-200M — `FALSE_METHODS_STATEMENT`

- **Current wording:** 3.15M … and 10M … pretrained on ~200M FineWeb tokens
- **Location:** `papers/paper1_draft.md:295`
- **Data required:** Both anchors total pretrain tokens ≈200M
- **Data available:** nano 32.8M (pretrain/AUDIT.md); scale ~200M (scale/AUDIT.md)
- **Confound:** False methods statement
- **Corrected wording:** 3.15M pretrained on 32.8M tokens (~3.1 epochs of 10.96M shard); 10M on ~200M tokens
- **Must change:** methods, latex methods, token table if any

### SC-P1-SEC72-FLAT50 — `UNSUPPORTED`

- **Current wording:** flat versus 3M/10M (~18) across ~50× parameters
- **Location:** `papers/paper1_draft.md:494`
- **Data required:** Parameter-only isolation
- **Data available:** Unequal token budgets; descriptive gaps
- **Confound:** tokens/param collapse 10→~1.26 from 10M→160M/200M
- **Corrected wording:** No monotonic gap collapse observed across evaluated configurations; not a parameter-only causal result.
- **Must change:** §7.2, latex withinstack

### SC-P1-CONCL-50X — `SUPPORTED_WITH_SCOPE_CHANGE`

- **Current wording:** within own-stack the diluted gap stays large across ~50× scale
- **Location:** `papers/paper1_draft.md:654`
- **Data required:** Matched-data scale law
- **Data available:** Descriptive multi-config comparison only
- **Confound:** Unequal pretraining exposure
- **Corrected wording:** Across evaluated own-stack configurations the diluted gap remained large under full FT; parameter count was not isolated from pretraining exposure.
- **Must change:** conclusion, latex conclusion

### SC-P2-FLAT50 — `UNSUPPORTED`

- **Current wording:** Across 50× of within-stack scale … the copying gap is flat
- **Location:** `papers/paper2_draft.md:30`
- **Data required:** Isolated parameter scaling with matched tokens
- **Data available:** Unequal tokens; 160M held ~200M recipe intentionally undertrained
- **Confound:** Paper2 design held 200M recipe fixed for 160M — not matched tok/param to nano
- **Corrected wording:** Descriptive: gap did not collapse monotonically with N under the evaluated recipes; nano token budget was 32.8M not ~200M.
- **Must change:** abstract, §3.1 title/body

### SC-P2-HELD-IDENTICAL — `FALSE_METHODS_STATEMENT`

- **Current wording:** Held identical to the anchors: pretraining recipe (~200M FineWeb tokens…)
- **Location:** `papers/paper2_draft.md:80`
- **Data required:** Anchors also ~200M
- **Data available:** Only 10M anchor ~200M; 3.15M is 32.8M
- **Confound:** False identity for nano
- **Corrected wording:** Held identical to the 10M scale recipe (~200M); nano used 32.8M — acknowledge mismatch.
- **Must change:** §2 design

### SC-LEDGER-SCALE — `SUPPORTED_WITH_SCOPE_CHANGE`

- **Current wording:** Within-stack scale (~50×) does not collapse diluted gap under full FT | Proven
- **Location:** `papers/EVIDENCE_LEDGER.md:27`
- **Data required:** Protocol licensing parameter-only claim
- **Data available:** Descriptive config comparison; unequal tokens
- **Confound:** Parameter×data entanglement
- **Corrected wording:** Across evaluated own-stack full-FT configs, larger N not associated with monotonic diluted-gap reduction | Supported | not parameter-only law
- **Must change:** ledger

### SC-RP-H0A — `DESCRIPTIVE_ONLY`

- **Current wording:** H0a … strongly supported … flat across ~50×
- **Location:** `papers/RESEARCH_PROGRAM.md:102`
- **Data required:** Causal parameter independence
- **Data available:** Descriptive flatness under unequal exposure
- **Confound:** Same as SC-P1-ABS
- **Corrected wording:** No monotonic collapse observed under evaluated own-stack recipes; not a parameter-only law.
- **Must change:** research_program, empirical_foundation

### SC-TEX-METHODS — `FALSE_METHODS_STATEMENT`

- **Current wording:** pretrained on ~200M FineWeb tokens
- **Location:** `papers/latex/paper1.tex:134`
- **Data required:** Both models 200M
- **Data available:** nano 32.8M
- **Confound:** Methods falsehood
- **Corrected wording:** Split token budgets per DIFF H
- **Must change:** latex

### SC-P1-E1-PLURAL — `UNSUPPORTED`

- **Current wording:** non-generative baselines dominate official generative LM references
- **Location:** `papers/paper1_draft.md:45`
- **Data required:** All named non-gen baselines meet dominance vs best generative
- **Data available:** Only M1 (0.999) exceeds official M0 (0.925); M2 (0.886) does not
- **Confound:** Plural overclaim
- **Corrected wording:** Under frozen U, M1 scored 0.998999 and exceeded best evaluated generative reference (official M0 0.925217).
- **Must change:** abstract, §0 title/body, conclusion, foundation

## Classification summary

- `UNSUPPORTED`: 4
- `FALSE_METHODS_STATEMENT`: 3
- `SUPPORTED_WITH_SCOPE_CHANGE`: 2
- `DESCRIPTIVE_ONLY`: 1

No claim in this audit is `SUPPORTED_AS_WRITTEN` for parameter-only 50× flatness or matched 200M anchors.
