# Conflict and Drift Register

*Do not silently edit owner lockfiles. Patches listed in `PROPOSED_OWNER_LOCKFILE_PATCHES.md`.*

| Conflict ID | Source A | Source B | Nature | Highest-quality evidence | Required correction | Owner approval? |
|-------------|----------|----------|--------|--------------------------|---------------------|-----------------|
| C-E2-STATUS | PREREG_E2 / FINDINGS: RUNNING U3 | DECISION_P1 / EMPIRICAL_FOUNDATION: GATED/STOP; RESEARCH_PROGRAM: BLOCKED | Live vs stopped vs blocked | No results_e2_*; runpodctl pods empty; e2_monitor.log empty | Converge all to GATED/STOP (no RESULT); archive pod ids | Yes |
| C-E3-HUMAN | PIPELINE_GATE_LOG / results_e3_human.json: EXECUTED | FINDINGS / RESEARCH_PROGRAM: BLOCKED; foundation operating-state pending; Paper α pending study | Done vs pending | results_e3_human.json (rater=agent-rubric-pass-1) | Sync status; rename human→rubric audit where accurate; keep IAA limitation | Yes |
| C-REGIME-STAGE1 | REGIME_P1: Stage 1 skipped | Gate log: Stage 1 executed | Skip vs execute | Gate log + Stage1 note | Fix REGIME status line | Yes |
| C-RHO-DEF | PREREG_E1 + e1/common.py: ρ = flagged/review load | DECISION_P1 §2.2: ρ = hallucination | Semantic mislabel | rho = self.flagged / nf | Correct DECISION_P1 table | Yes |
| C-UTILITY-SKIP | results_e1_utility.json construct: SKIPPED | Stage 1 executed later | Stale JSON note | Human JSON timestamp | Annotate note as superseded | Yes |
| C-PREREG-STALE | Slot/ownstack/Tv2 preregs "not run" | Result JSONs exist | Status drift | JSON artifacts | Add RESULT sections | Yes |
| C-FABRIC-APPEND | README lineage / append-only vibe | open(...,"w") truncates | Overclaim | slice.py ledger open mode | Soften docs | Yes |
| C-FABRIC-INTENT | slice header Intent→Control | No kernel code | Doc overclaim | Code path | Remove implemented implication | Yes |
| C-MASTER-RESIDUAL | Older interference-"leading" language | C-1b REFUTED | Obsolete causal account | results_interference_10m.json | Scrub residual leading phrasing | Yes |
| C-GIT-LOCKS | Prose locked/immutable | Most E1/E3/P1 files untracked | Archival gap | git status | Commit or declare working-tree-only | Yes |
| C-JSONL-GITIGNORE | Preregs cite raw JSONL | *.jsonl gitignored | Repro gap | .gitignore + local files exist | Document local-only; optional release bundle | Yes |
| C-PAPER2-NAME | paper2_draft.md filename | RESEARCH_PROGRAM: Paper-1 extension | Naming drift | RESEARCH_PROGRAM | Rename or banner | Optional |
| C-AZ-NEXT | AZ plan P2 next in places | Gate 3 PASS / P2 frozen | Stale next-step | PIPELINE_GATE_LOG | Next = E4 authorize or Idle | Yes |
| C-LEDGER-PROVEN | EVIDENCE_LEDGER Proven KILL | Kill under frozen U/task | Language hardening | E1 prereg | Keep scoped wording | Yes |
| C-AUTONOMOUS-E2 | .autonomous tried E1→E2 | E2 gated / old-task forbidden | Process drift | session.log | Do not resume | N/A |
