# ONTOGENESIS research ledger

The one authoritative status list. Updated 2026-11-21 (see the dated section at the end; it overrides earlier rows where they conflict). Status values: Established (published elsewhere), Derived (mathematics checked here), Demonstrated (measured here under a committed protocol), Pilot (measured, not confirmatory), Unverified, Disproven, Abandoned, Open.

## Measured in this repository

| Item | Status | Evidence |
|---|---|---|
| Exact copying helps on code: M21 1.79 vs dense 2.15 bits/token on copyable targets | Demonstrated | `research/frontier/m21_dense_code/TIME_MATCHED_RESULTS.md` |
| Dense beats M21 at equal training time: 4.65 vs 5.02 bits/token | Demonstrated | same |
| M21 beats dense at equal tokens only through copying | Demonstrated | `research/frontier/m21_dense_code/RESULTS.md` |
| S2 hybrid edges dense at 350M: 0.86339 vs 0.86774 bits/byte, one seed, 0.48x speed | Pilot (one seed, below its own 1% rule) | handoff `antigravity-continuation/` |
| Pythia-410M beats all native models on public text and grammar | Demonstrated | `experiments/native_public_language`, `experiments/grammar_contrast` |
| ACE router original gain came from reading future tokens | Disproven (original claim) | `research/ace_x9/` |
| Wide rotation registers (U10) | Disproven (null) | `research/frontier/u10_registers/` |
| DEMOS energy-minus-price-times-mass is non-increasing | Derived, checked numerically | `check_lyapunov.py` |
| Sort gives latest-write lookup for all positions; (min,max) closure gives reachability at every prefix; symmetric closure equals union-find | Derived, checked numerically | `theory/prefix_structure_checks.py` |
| DEMOS learns one-hop lookup, 32 static facts: 99.7% (dense 100%) | Pilot | handoff `ontogenesis-pilots/` |
| Dense fails one-hop lookup with updates after 7 min: 20.4% (guess rate) | Pilot | same |

## Ideas

| Idea | Status | Note |
|---|---|---|
| Interaction mesh, space-time mesh, UII candidates | Disproven at toy scale | dense won |
| ERN surprise-driven recurrence | Disproven at 125M | 2/16 |
| Holonomy rotation registers | Abandoned | solves group composition; <1% on text at ~8x cost |
| Gödel max–min model | Pilot (synthetic only) | algebra reused by RELATOR |
| LEDGER, DYADIC, LEXICON | Abandoned | largely published; known failure modes |
| METRIPLEX | Abandoned | parallel form is linear attention |
| ALGEBRA-EQUILIBRIUM | Abandoned | cost and capacity inconsistent |
| DEMOS | Open (pilot running) | no expressivity-per-cost gain over looped sparse message passing; sample efficiency untested |
| KEYSTORE | Open | partly published; learned keys untested |
| RELATOR | Open, blocked | smooth training leaks future; needs hard causal training |
| FILTRA | Open, not built | not found inside a language model |
| Prefix-scannable models | Established | ICLR 2026 |
| Matrix prefix products for state tracking | Established | 2024–2026 line |
| Datalog + union-find unification | Established | egglog |
| Time-threshold (filtration) for causal relational memory | Derived | one-line corollary of provenance semirings; use in LM training not found |
| STRATA | Unverified | every component published; learning of structure is the open row |
| PATHWAY (unified design) | Abandoned as written | critic: algebra misnamed, linear star weaker than a looped layer, cost claim unsupported, components published; see section 11 of `theory/UNIFIED-ARCHITECTURE-20261120.md` |

## Open scientific questions, in order of information per cost

1. How large is the long-context gain on our text and code, and on which tokens? (L0)
2. Can discrete keys be learned from a prediction loss? (KEYSTORE learned keys)
3. Does iteration over a persistent graph beat a looped Transformer at equal compute?
4. Does any structured memory keep its advantage once the dense model grows?

## Duplicates and conflicts resolved

- Names Gen3/4/5, S2, M0/M2, Omega each mean several things; cite path and commit, never the bare name.
- Five competing 2B–3.9B "flagship" plans in the archive: none funded; all superseded by this ledger.
- CLOSURE is merged into RELATOR.

## Research tree (2026-11-20)

Root: one unified language model with better capability per parameter, token and unit of compute.

| Branch | Question | State | Next action | Where |
|---|---|---|---|---|
| B1 Learning speed on changing facts | Why did DEMOS learn updates when dense did not? | Running | tuned dense, looped, message passing, KEYSTORE; then ablations | `train.py`, handoff `ontogenesis-pilots/` |
| B2 Knowledge in the weights | Does two-way binding remove one-way fact storage? | Not started | critic review, then REVERSE | `MISSION.md` lead C |
| B3 Reusable operations | Does an operation store beat a meta-trained Transformer? | Not started | critic review, then REUSE | `MISSION.md` lead A |
| B4 Size of the long-context gain | How much, on which tokens, prose vs code? | Blocked | obtain a corpus of long documents | `theory/ONTOGENESIS-PHASE2-20261120.md` L0 |
| B5 Exact keyed memory in context | Can discrete keys be learned from a prediction loss? | Arm built | pilot queued | `structured.py` |
| B6 Relational closure | Hard causal training for RELATOR | Blocked | design training without the smooth semiring | round 2, section 9 |
| B7 Identity merging | FILTRA | Not built | after B5 | round 2 |
| B8 Copy and recurrent baselines | Port S2 and M21 to the harness | Not started | after B1 baselines | master `research/frontier/m21_dense_code`, `experiments/native_public_language/archived/gen3.py` |
| B9 Compute | TPU price and availability | Partly done | verify hourly price on the billing page before any run | handoff `tpu-probe/RESULT.md` |

A branch is closed when its decisive experiment is reported, pass or fail. New branches are added here before work starts.

## Archive, with where each item lives

Source: two read-only surveys on 2026-11-20 (repository and Antigravity records). Sub-readers opened most files; treat rows as Observed unless marked.

| Item | Mechanism | Largest scale | Result against dense | Location | Status |
|---|---|---|---|---|---|
| Gen2 hybrid (X4, X8) | recurrent delta blocks between attention blocks | 46M | 0.9575 vs 0.9645 bits/byte, two seeds | branch `nanolm/discovery-lab-v1`, `lab/arch/` (local only) | Pilot |
| S2 / Gen3 | Gen2 plus erase and differential attention | 350M | 0.86339 vs 0.86774, one seed | branch `nanolm/arch-evolution-v3`, `lab/arch/gen3.py`; archived copy on master | Pilot |
| S2 at 125M, second implementation | same name, different code | 125M | 0.91355, worse than dense | worktree `nanolm-v3` | Contradicts the row above |
| Gen5 / M2 | Gen3 plus physics-style attention | 350M | ahead historically, behind on public text | `gen5_foundations.py`; evals on master | Demonstrated (public text) |
| BOBO, Kalman landmarks | fast-weight mixer, surprise anchors | 125M | document claim only | `nanolm/arch-evolution-v3` | Unverified |
| ERN | surprise-driven constant-state recurrence | 125.8M | 2/16 in every arm | branch `nanolm/autonomous-discovery-v1` | Disproven |
| Holonomy / HSPA | rotation registers | 14M | 0.3751 vs 0.3784 at ~8x cost | PR #174 | Abandoned |
| "Omega-1.7B beats Qwen3-8B" | a Qwen LoRA, not a native model | — | 40 vs 39 of 56; 32 replies truncated | PRs #171, #174 | Disproven as stated |
| G3/G4 discovery backbone | Householder scan | 20.6M | 2.1112 vs 2.1188, within noise | PR #176 | Unverified |
| Tool-recovery 60/60 vs 17/60 | — | — | outcome was hard-coded | PR #176 | Disproven |
| Mechanism lab X03–X07 | recurrent cells | toy | mixed after correction | PR #175 | Pilot |
| M13–M22 zoo, M21 | rotation, slots, copy pointer | 5M | see measured table | branch `research/unified-discovery-loop-v1`; port on master | Demonstrated |
| Omega core, ACE-X9.2, IVE | router, memory tiers, verifier | 47M | ACE lost at scale; leak fixed in #191 | unpublished local commits on `codex/p1-h7-scribing-path` | Disproven (ACE) / Unverified (rest) |
| Interaction mesh E27, bitemporal E28 | signed node mesh | <65k | 0.258–0.516 vs 0.521; one tie | same local commits, `reports/E27`, `reports/E28` | Disproven at toy scale |
| UII candidates A–F | mesh, field, attractor, automaton | 9k | 0.573 vs 0.621 | same, `reports/UII` | Disproven at toy scale |
| V3 exact recall | copy pointer | toy | original contaminated; clean rerun seed-dependent | branch `research/v3-exact-recall`; #187 on master | Demonstrated (rerun) |
| V4 first principles | event-state operator | — | never run | untracked files in worktree `nanolm-v4-fp` | Open, at risk (uncommitted) |
| V5 Gödel model | max–min message passing | 19k | 0.845 vs 0.39 on 5-hop graphs | branch `research/v5-distributed-systems` | Pilot (synthetic) |
| FP1–FP4 | first-principles benchmark | 2.5M | FP4 zero-parameter arm 11/30; learned arm not run | branch `recovery/nanolm-cycle4-first-principles-20261119` | Open |
| 2B and 4B presets | configurations | 1.98B–3.93B | untrained | `nanolm/arch-evolution-v3`, #171 | Unverified |
| "1.54B NanoLM" | — | — | was unmodified Qwen2.5-1.5B | Antigravity audit note | Disproven |

Not yet audited: seven openresearch worktrees and the `orx/*` branches; 12 foundation documents on `origin/research/foundation-intelligence-program`; the holonomy publication draft. V4's work exists only as untracked files and should be committed by its owner before any cleanup.

## Pilot results added 2026-11-20 (evening)

One-step lookup with updates, 48 live facts, 7 minutes, one seed, guess rate 20.4% (`results/pilots/pilot2-*`, `pilot3-*`):

| Arm | Accuracy |
|---|---|
| KEYSTORE, given keys | 100% |
| Looped Transformer, lr 1e-3 with warm-up | 99.95% |
| Dense Transformer, lr 1e-3 with warm-up | 99.8% |
| DEMOS | 98.3% |
| Sparse message passing with forgetting | 98.0% |
| Dense, lr 3e-3 no warm-up; dense, lr 3e-4 | 20.4% |
| KEYSTORE, learned keys | 20.4% |

- DEMOS learning-speed advantage: **Disproven** (baseline was mis-tuned).
- Learned discrete keys by straight-through estimation: **Pilot, failed**.
- Mixed one- and two-step task: every arm run so far is at guess rate; under-budget, **Open**.
- DEMOS first-order hop step cannot remove the current key: **Derived**, overlap checked numerically (`MISSION.md` section 8).

PLASTICITY v0, 200 successive tasks of 256 facts, three seeds, predictions written before the run (`plasticity.py`, `results/pilots/plasticity-v0-summary.json`):

| Arm | Steps to 95%, first 50 tasks | Last 50 tasks | Retention of earlier tasks (chance 1.6%) |
|---|---|---|---|
| Plain MLP | 61–72 | never reached within 300 | 1.5% |
| MLP with layer normalisation and weight decay | 33–34 | 37–38 | 4% |
| Sparse key-value memory | 254–267 | never reached within 300 | 3.4% |

- P1 (a plain dense network loses its ability to learn): **Demonstrated at pilot scale**, all three seeds.
- P2 (a sparse key-value memory keeps its speed and retains more): **Disproven** in this implementation. It was slower from the start, lost plasticity, and retained nothing.
- A standard remedy (normalisation plus weight decay) keeps learning speed almost flat. It is the strongest simpler alternative for any plasticity claim.
- No arm retains earlier tasks. Plasticity and retention are separate problems here.


## Update 2026-11-21

Full write-ups with registered predictions: `INVESTIGATION.md`. Decision note: `theory/RESEARCH-DECISION-20261121.md`.

### Measured today

| Item | Status | Evidence |
|---|---|---|
| Looped learns two-step chains in 3 of 5 seeds, dense in 1 of 5, at 26,000 steps; dense does make the jump | Demonstrated (registered verdict: mixed) | `results/loop2/` |
| The first acquisition benchmark is passed by a lookup table | Derived | `INVESTIGATION.md` Question 6 |
| In a rule-governed world a gradient learner keeps 99.6% of old regular facts with no replay | Demonstrated, toy | `results/forgetting_split/` |
| Corrections, not exceptions, cause the loss of old regular facts | Demonstrated, toy | same |
| Oracle routing raises inference 78.6% to 89.2%; this is label-noise removal | Demonstrated, toy | `results/routing/` |
| A write-time detector without an oracle recovers a third of that gain and costs 4 points on clean entities | Demonstrated, toy; registered bar not met | `results/routing_loo/` |
| Fixed-address memory stops acquiring because of concentrated addresses together with carried-over optimiser state; a reset restores 86%, hashing 100% | Pilot (one seed, large effects) | `results/memory_saturation/` |
| A harmless address-only edit exists for 99.8% of corrections | Demonstrated, toy | `results/correction_interference/` |
| Correction damage is caused by missing constraints: not by capacity, write size, the optimiser, or lack of general replay | Demonstrated, toy | same, `margins-*`, `optimizer-*` |
| A correction written into the model decays under later training whatever its margin; only the exact store keeps it | Demonstrated, toy | same |
| Exact 4-gram store added after training: 0.058 bits per byte better for a 14M model trained on 33M tokens | Pilot (one seed) | `results/lm/store-pilot.json` |
| Training the model through that store: 0.14 worse than adding it afterwards | **Unsettled** (2026-11-22): the joint arms' gate could not drop the store's weight below 0.25; see `INVESTIGATION.md`, corrections 2026-11-22 | same |
| The same store adds 0.004 bits per byte to the same model once trained on 800M tokens | Pilot (one checkpoint) | `results/lm/store-exchange-S.json` |
| JAX model for TPU matches the PyTorch reference to 2e-6 | Verified | `tpu/parity_check.py` |
| Dense baseline, 14.3M parameters, one pass of 800M tokens: 1.144 bits per byte | Provisional (one seed; first-version sampler oversampled the first 37% of the corpus by about 12%; fixed-sampler rerun queued) | bucket `runs/ladder-v5e/` |
| Exact in-context cache adds under 0.01 bits per byte on every domain to the trained 40M model (code 0.006) | Registered rule read: explicit-copy line closed at this scale | `results/lm/cache-exchange-M.json` |
| The n-gram store adds 0.004 to the trained 14M model and −0.001 to the 40M one | Pilot (one checkpoint each) | `results/lm/store-exchange-*.json` |

### Status changes

| Idea or claim | New status | Reason |
|---|---|---|
| DEMOS | Disproven (learning-speed claim) | tuned dense reached 99.8% |
| KEYSTORE with learned keys | Disproven at toy scale | stayed at guessing |
| Nested dense-loop architecture | Open; registered test running | published family; adds no function class at equal depth |
| "Exceptions damage regular knowledge" | Disproven | two controls |
| "Route by regularity" as an architectural principle | Not adopted | needs an oracle; fails on text when trained jointly |
| "Fixed addresses trade retention for acquisition" | Disproven as stated | optimiser reset removes most of the trade |
| "Never forgetting equals order-independence"; "pick two" trilemma | Withdrawn | not theorems as stated; the second is published in a narrower form |
| n-gram store as a core component | Close unless the 101M checkpoint shows a gain of 0.01 or more | rule registered in the decision note |
| Exact in-context copy mechanism (M21 line) | Closed at this scale and context | registered rule read on the 40M checkpoint, 2026-11-22 |

### Compute facts

- Cloud TPU v6e: no on-demand capacity for single chips in any permitted zone on 2026-11-22 UTC; a four-chip node was obtained in asia-northeast1-b. v5e (4 chips per zone allowed) obtained in us-west1-c.
- Measured training speed per chip, 101M-parameter model: v5e 130k tokens per second, v6e 230k. At list prices the v5e is the better value at these sizes.
- Global cloud CPU quota is 12 cores; GPU quota is still 0.
- The laptop cannot host more than light jobs: twice today training runs pushed it to 40 GB of swap.

## Operations, 2026-11-22 (UTC)

**Corpus v2 built.** 10.0B tokens, `gs://nanolm-lab-eu-0478806098/corpus/v2/`, train file sha256 prefix df0e0a2f894a4945.
- Shares: web 6.2B, textbook 1.3B, maths 1.3B, code 1.2B.
- Documents: 6.55M, 1.87M, 0.93M and 0.65M respectively.
- Same pinned sources, filter, tokenizer and interleaving as v1. The held-out sets are v1's, unchanged; the builder reproduced them and v1's first 20.6M training tokens exactly before building.
- The 13-gram held-out overlap filter removed 172,563 code and 111,804 maths documents, mostly shared boilerplate. It is the same rule v1 used.
- Built in 77 minutes on 16 spare cores of machine A.

**Deletion safety.** Each machine has four independent ways to be deleted:
1. the controller's own end-of-work delete;
2. a persistent systemd timer that sends the delete request first and retries until the API confirms;
3. a Cloud Scheduler delete in the project (rehearsed with a read-only call: HTTP 200);
4. a laptop watcher.

Deadlines:
- Machine A (v5e-4, us-west1-c): 11:06 systemd, 11:08 Scheduler.
- Machine B (v6e-4, asia-northeast1-b): moved to 12:30 systemd, 12:32 Scheduler, 12:40 watcher, so that the 40M recipe grid and Question 12 can run.

**Hand-overs.**
- The batch-1 controller on machine A survived the first hand-over, because the hand-over killed its launch wrapper. It would have deleted the machine when the CPU runs ended, under a later batch. It was removed by pid.
- On machine B I first removed the wrong shell: the log-push loop instead of the time-limit shell, because both had `sleep` children. I checked again and removed the right one, after the new deadline was confirmed. No cost or data effect.
- Machine B now runs a queue (`tpu/job_queue.sh`): work is added as scripts in `~/queue/`, and the machine deletes itself when the queue is empty.

**Spend estimate for this cycle** (list prices $1.20 per v5e chip-hour and $3.24 per v6e chip-hour; actual billing not yet read):

| Machine | Expected hours | Expected cost | Hours to deadline | Cost to deadline |
|---|---|---|---|---|
| A, v5e-4 | about 9.5 | about $46 | 10.1 | $49 |
| B, v6e-4 | about 10.5 | about $136 | 11.1 | $144 |
| **Total** | | **about $180** | | **about $195** |

Kaggle is free. No RunPod spend.

## Sections added on research/first-principles-architecture-ledger-20261121 (PR #195)

Reconciled 2026-11-22: the two branches appended at the same point. Both are kept verbatim: the sections above are from research/ontogenesis-phase1, those below from PR #195, in their original order. Where a row below conflicts with a later dated row above, or the reverse, the later date governs, as the header says.

- A standard remedy (normalisation plus weight decay) keeps learning speed almost flat. It is the strongest simpler alternative for any plasticity claim.
- No arm retains earlier tasks. Plasticity and retention are separate problems here.

### 2026-11-21 companion diagnostic (additive, no change to prior verdicts)

| Decision ID | Scoped result | Experiment provenance | Interpretation and next gate |
|---|---|---|---|
| D-023 | Synthetic 16D orthogonal basis migration: no inverse 21.0%, estimated generic inverse from 4/8/12/16 paired anchors 46.8/59.8/73.4/100% top-1 agreement (5 seeds) | [Candidate architecture and rank proof](architecture/01_ARCHITECTURE_AND_THEOREMS.md); companion downloadable code/raw results | Consistent with full-rank linear map identifiability; **not** neural-model continual learning or efficiency; test structured real feature drift against replay and total write/read cost |

## Additive research reconciliation — 2026-11-21

**Scope:** Proposed review-branch update; earlier results, preregistrations and negative findings above remain unmodified. The linked [architecture/theory package](architecture/README.md) is a candidate, not a selected Nano LM backbone or claim of a scientific breakthrough.

| Decision ID | Updated observation / design status | Source | Required next gate |
|---|---|---|---|
| D-005 | Dense code LM 4.65 BPB/token vs M21 5.02 at time-oriented budget; M21 conditional-copy subset remains interesting | [Merged PR #194](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194), including ~30% timing drift | Conditional-copy control in real text/code |
| D-006 | DEMOS 'learns faster' interpretation superseded by tuned dense result | Historical pilots and [INVESTIGATION.md](INVESTIGATION.md) | Preserve negative result; no large DEMOS scale |
| D-007 | Exception-collateral interpretation refuted: corrections-only reduce old regular fact retention; exception-only does not | [DISCOVERY-CYCLE-1](theory/DISCOVERY-CYCLE-1-20261121.md) §8.1 | Same-entity replay × optimizer control |
| D-008 | Hashing and Adam-history reset separately improve late acquisition on one-seed addressing pilot (original 3.4%; hash 99.9%; reset 86.1%) | [INVESTIGATION.md](INVESTIGATION.md), Question 7 | Multi-seed address × optimizer factorial |
| D-009 | Non-oracle leave-one-out write-time routing yields 81.4% vs baseline 77.6%, oracle 88.9% in 3-seed toy study | [INVESTIGATION.md](INVESTIGATION.md), Question 8 | Distinguish exceptions from changing general rules without oracle |
| D-019 | Five *conditional* mathematical/algorithmic properties specified; five numerical sanity checks passed locally; **no LM superiority proof** | [Propositions](architecture/01_ARCHITECTURE_AND_THEOREMS.md) | Independent derivation audit, then experiments |
| D-020 | Native dense baseline remains reference; APS, hybrid, and recurrent candidates unselected | [Decision register](architecture/02_DECISION_LEDGER.md) | Preregister [18-gate plan](architecture/03_EXPERIMENTS_AND_EXECUTION.md) in authorized environment |

**Maintenance rule:** integrate an update into this canonical ledger on each meaningful completed experiment/independent review, with exact code revision, data provenance, seeds, hardware and supersession links; not simply because a theory was proposed. This review branch does not merge or modify running experiments or raw results.

## Review supplement V2 — theory and empirical gaps (2026-11-21)

**Canonical precedence:** preserve the original experimental record, independent referee corrections and registered protocols above. The [v2 review](architecture/05_GAP_CLOSURE_AND_FORMAL_REVIEW.md) is a *candidate scientific review*, not an adopted backbone or new language-model result. Its [machine decision events](architecture/ledger_events.jsonl) and [schema](architecture/ledger_event.schema.json) are an additive audit supplement. [Validator](architecture/checks/validate_research_package.py) checks documentary integrity, not scientific truth.

| ID | Decision and reason | Evidence status | Next gate |
|---|---|---|---|
| D-024 | Do not promote elementary conditional transport, projection, memory and compute equations into a proof of LM superiority | Scoped math only | Independent mathematical review; G05/G06/G15 |
| D-025 | Unknown feature-map identification, migration and end-to-end state rewrite costs must be measured | Hypothesis | G05 against frozen/replay controls |
| D-026 | Null-space edits guarantee only local first-order anchor invariance, with a second-order remainder under smoothness | Conditional bound; prior art | G06; compare acquisition and retained old skill |
| D-027 | Bitemporal, permission-aware records are **augmented-system** consistency, not intrinsic knowledge or truth | Deterministic policy | Version/conflict/retraction tests |
| D-028 | One-step compute threshold is myopic; multi-step stopping needs budgeted value and causal estimates | Mathematical correction | G15 fixed-depth and scratchpad comparisons |
| D-029 | Dense D0, published hybrid H1, recurrent R1, APS candidate A1 and new X0 remain competing architectures | Unselected | Fresh native language/code Pareto evaluation |
| D-030 | Plasticity, retention, correction, transfer and sample efficiency require separate endpoints | Existing toy results; open transfer | G03/G07 |
| D-031 | A live write router must distinguish rule changes, exceptions and noise without oracle future labels | Repo pilot, mechanism unresolved | G04 prequential stream |
| D-032 | Human decisions now have an append-only event schema and validation contract; canonical history stays in this ledger | Review-only process | Independent review before PR merge |
| D-033 | Research scripts must not silently overwrite frozen result JSONs | Reproduced operational defect, fixed in review branch | Read-only test and no-overwrite negative test |

**Current explicit open gaps:** no learned state-migration result on actual language models; no proven positive forward transfer; no confirmed model-level Pareto gain; no authoritative inventory of owner-local uncommitted worktrees; no verified TPU/GPU price for scale; no blanket rights to restricted data. Historical comparison in D-005 is **bits per token**, not “BPB/token”; calibrated wall time drifted in [PR #194](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194). Preserve the paper's and repo's actual domains/metrics.

**Maintenance policy:** new evidence is an append-only event with ID, code/data checkpoint and protocol SHA, independent unit/seeds, time/resource accounting, reviewer, contradicting alternatives and next gate. Avoid daily prose churn when nothing changed. This branch remains a draft PR; no registered results modified or paid run authorized.

## Addendum — store pilot result and validator repair (2026-11-21, late)

Additive. Nothing above is changed.

| Item | Status | Evidence |
|---|---|---|
| Adding an exact n-gram store after training helps: 0.058 bits per byte on the mix at 8 blocks, every domain (S1) | Pilot (one seed) | `INVESTIGATION.md`, "Language-model store pilot: result", commit `e9ba4f3` on `research/ontogenesis-phase1`; raw file `results/lm/store-pilot.json` there |
| Training the core with the store beats adding it afterwards (S2) | Disproven in this implementation: 0.14 worse than post-hoc, 0.085 worse than no store | same |
| Four blocks trained with the store match eight blocks alone (S3) | Disproven: 1.581 against 1.497 | same |
| The store costs 1.02 GB against about 57 MB of weights | Observed | same; any gain is an augmented-system gain |

- The pre-recorded caveat stands: same-document leakage in training biases this test against the joint arms. A leave-document-out store is the control that would separate the caveat from the claim. Until it is run, "routing memorisable content away from the core during training carries over to text" is **not supported**, not refuted.
- The pilot is complete with a negative primary result. The architecture package in this branch was written before commit `e9ba4f3` and does not mention it; the 2026-11-21 blueprint's evidence table lists it as having no final result, which is now out of date.
- `architecture/checks/validate_research_package.py` as first committed returned FAIL: its three patterns were escaped twice inside raw strings, so the link check matched none of the package's 114 links and the decision-row check could not match. With the patterns corrected it returns PASS, and it fails as intended on a planted broken link, a duplicated event and a missing human-ledger row. The earlier "PASS" for link and package integrity was not reproducible from the committed file. CI does not run this validator.


## Review supplement V3 — executable architecture, corrected evidence and maintainable history (2026-11-21)

**Date:** November 21, 2026, America/New_York. This is an additive review: every preceding byte of this ledger is preserved. Principal completed-result evidence is pinned to research commit **2ef9ce5827cb8ea1fb6c8946df32fbcf5e06dec0**. The integration cutoff is **8d3b03e08a5c14d2e9ab330ac8d5006e87e61613**, which adds protocols and operational updates, not a newly verified result. The [V3 package](architecture/README.md) keeps Nano LM's native-model objective and leaves the final backbone unselected.

### Decisions, reasons and scope

| Decision area | Change and reason | Outcome / scope |
|---|---|---|
| Executable architecture | Specify tensors, causal masks, tied output, logical-depth caches, loss and reset behavior; diagrams alone were insufficient | Tiny untrained dense/shared-block reference in [07](architecture/07_IMPLEMENTATION_CONTRACT.md) |
| Formal transport | Distinguish functional conjugacy from an allowed model parameterization; normalization, RoPE and embedding ties impose compatibility conditions | Conditional proofs and counterexamples in [01](architecture/01_ARCHITECTURE_AND_THEOREMS.md) |
| Constrained updates | Protect against the complete optimizer displacement; raw-gradient projection can be undone by preconditioning or decay | Local anchor bound, residual and acquisition requirements; no universal no-forgetting theorem |
| Historical records | Add valid-time and immutable recorded-time cutoffs; backdated events can change single-time reconstructions | Declared augmented-store consistency, not truth or native knowledge |
| Empirical status | Incorporate completed store and correction pilots instead of leaving them queued as unrun | Existing evidence reanalysed in [09](architecture/09_EVIDENCE_REFRESH_AND_PROTOCOL_REPAIRS.md) |
| Experimental controls | Repair common-generator coupling, solver-status interpretation and changing durability cohorts | G03b successor specification; original results and protocols unchanged |
| Decision maintenance | Separate stable decisions from successive events and validate previous-history prefixes | Extensible [event log](architecture/ledger_events.jsonl) with explicit revisions |
| Literature | Inspect primary identities, supervision and plugin dependencies; narrow claims to inspected evidence | [36-source audit](architecture/08_LITERATURE_AUDIT.md), without treating citations as local replications |
| Latest upstream changes | Record correction-margin predictions and final nested-device deviation separately from measurements | D-046; protocol/source update only |

### Completed evidence now incorporated

The **one-seed language-model store pilot** reports weighted BPB of 1.4972 for eight blocks alone, 1.4391 with a post-hoc store, and 1.5826 after joint training. The post-hoc system improves by 0.0581 BPB, while joint training is 0.1435 worse than post-hoc mixing. The store occupies **1,022,224,980 bytes**. Preserve the original failed S2/S3 outcomes: a same-document training-exposure caveat motivates a new document-excluded control; it does not turn a failure into success. Crowded-machine timings cannot establish an efficiency advantage. This is augmented-system evidence, distinct from the earlier M21 **bits-per-token** comparison.

The **three-seed correction pilot**, seeds 8001–8003, shows why immediate locality and durable retention require different endpoints. Correction-only writes achieved 100% immediate correction accuracy but 64.70% accuracy on measured neighbouring facts. Same-entity replay achieved 100% on both immediate measures, yet its late correction accuracy was 62.28%. Computed address edits achieved 99.84% immediate correction accuracy and 98.03% immediate neighbour accuracy, with 21.47% late correction accuracy. These are synthetic bilinear-model results, not native-language-model measurements. Immediate and late aggregates use different cohorts; subtracting them is not a paired forgetting curve.

New source diagnostics found that random replay consumes the generator also used for future corrections: **38 of 39 correction phases differ in each audited seed**. This invalidates treating that arm's seed-matched scores as the same-event paired comparison. It does not alter the matched single-write/same-entity-replay streams, erase descriptive results, or quantify the bias. The solver helper also collapses different non-success statuses into one failure; it does not certify representational impossibility. The successor requires immutable common event manifests, independent replay generators, actual constraint residuals, deployed-precision checks and per-edit survival cohorts. Raw source/result copies and hashes remain under [evidence](architecture/evidence/source_manifest.json).

### Architecture and proof implications

The illustrative dense configuration has **15,208,704** learned scalars; four shared blocks executed at eight logical depths have **11,798,784**. Both require eight logical KV caches. These are exact proposed-configuration counts, not recovered trained checkpoints or a parameter-matched quality comparison. Sharing weights does not halve executed depth, cache storage or training cost.

The formal revision also makes input reinjection optional, defines causal CE before adding auxiliary losses, and separates checkpoint prediction equivalence from optimizer-dynamics equivalence. Finite-sequence drift bounds require a prefix-wise premise. Historical-query invariance requires fixed policy/view and recorded cutoffs. Adaptive stopping requires consistent cost units and a budgeted decision model. One candidate's failure cannot falsify the existential APS ambition; promotion uses a finite preregistered candidate claim with resource and regression guardrails.

### Actual validation and decision history

| Executed check | Result | Supported scope |
|---|---:|---|
| Tiny native forward reference | **13/13 passed** | Causal/full-prefix/incremental agreement, reset/cache ownership, inventories and input contracts |
| Mathematical examples | **12/12 passed** | Explicit examples and counterexamples |
| Original NumPy suite | **5/5 passed** | Small algebraic checks |
| Package regression suite | **35/35 passed** | Valid extensions accepted and corrupt records rejected |
| Synthetic transport diagnostic | **Five seeds reproduced** | Existing 16-dimensional linear-anchor example |

The [validation report](architecture/06_VALIDATION_AND_UNRESOLVED_WORK.md) and [final receipt](architecture/results/validation_v3_final.json) identify commands, hashes and limitations. Independent review found and repaired stale-cache identity, unsigned-loss and fractional-byte defects after the initial 12-case forward pass. Passing tests did not validate all training, every configuration, remote CI or deployment efficiency.

The final register contains **46 human decision IDs**, **27 machine events**, and **23 machine-covered decisions**. The original 23 human rows and ten machine events remain unchanged; missing historical timestamps or reviewers were not invented. Revisions use unique event IDs and explicit previous-event pointers. Schema validity checks record structure and continuity, not scientific truth or authorization.

### Latest cutoff and next gates

At the integration cutoff, [correction_margins.py](architecture/evidence/upstream/correction_margins.py.txt) adds prerecorded M1–M4 predictions comparing minimal edits and preservation/margin choices. No new margin result was verified. The [nested deviation record](architecture/evidence/upstream/PROTOCOL-NESTED-DEVIATION.md.txt) now reports a final cloud-host CPU execution plan after another laptop-GPU stop. It calls for dense/looped controls on the same CPU and both registered and device-consistent evaluations; disagreement means device-sensitive, with no rule declared met. This review did not verify a live job or completed nested result.

Next: reconcile new upstream receipts before declaring completion; recover the native baseline's code/tokenizer/data/checkpoint lineage; implement G03b's common-stream, same-edit durability protocol; complete G02's address-by-optimizer controls; then bridge a supported mechanism into native text/code evaluation against tuned dense and targeted replay. Whole-document exclusion precedes broader store-training conclusions. Learned transport, positive forward transfer and scaling remain conditional on fresh evidence and complete costs. The [execution plan](architecture/03_EXPERIMENTS_AND_EXECUTION.md) records dependencies and stopping rules.

Publication belongs in existing draft [PR #195](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/195), without merge. No protected experiment, frozen result, original ledger entry or spending authorization is changed. Future maintenance appends evidence and reasons when they materially change a decision.


---

## 2026-11-21 — V4 native training, controlled-update integration and margin evidence

Recorded at 2026-11-22T01:50:45Z. V3 is preserved above. Current architecture additions are [training/measurement10](architecture/10_TRAINING_AND_CONTROLLED_UPDATE_CONTRACT.md) and [margin evidence/theory11](architecture/11_MARGIN_RESULTS_AND_DURABILITY_THEORY.md). Research cutoff:9d88a2c20233f42b4a49c4430264b20f842895ad. Existing research PR195 remains the draft, unmerged review destination.

| Outcome | Evidence and interpretation |
|---|---|
| Margin protocol completed | Five committed arms × three reused seeds; M1 and finite M4 hold, M2/M3 compound targets fail. Source/results remain byte-preserved. |
| Acquisition timing repaired | Source corrected_now is phase end; per-write and later-write interference were unobserved historically. New event instrument separates them and records same-value revisions. |
| Native training implemented | Dense/shared analytical gradients, AdamW, immutable snapshots and checkpoint/resume;17 behavioral groups and62 finite-difference directions. |
| Actual-state receipt defects fixed | Overlapping tensor storage and overflowing diagnostic norms now fail before mutation; layout and crash-atomicity boundaries explicit. |
| Event measurement implemented |26 behavioral tests and24,000 independent oracle cohort rows across320 synthetic histories. |
| Native integration executed | Eight fixed events,16 real updates,16 scored target exposures; actual native outputs and complete event/state/horizon receipts; no efficacy threshold. |
| Conditional durability theorem | Joint address/readout/interacting drift plus telescoping sufficient budget;9 numerical test cases. No universal no-forgetting claim. |
| Decision maintenance | D-046 revision plusD-047–D-054;54 human IDs,36 machine events covering31 decisions; full prior27-event machine log and canonical ledger prefix preserved. |
| Attribution |40 primary identities, with component-method sources and source-specific inspection depth. |

The new pilot result is upstream evidence reanalysed here. Local native learning is limited to synthetic engineering fixtures; no corpus-scale/GPU/TPU training, paid compute, held-out benchmark, architecture selection or merge was performed by this update. G03b still requires a separately frozen scientific protocol, fresh seeds, native data, unrelated-quality/new-learning guardrails and total cost accounting. Existing source protocols were not revised retrospectively.

Full current decisions, alternatives and next gates are in [architecture ledger02](architecture/02_DECISION_LEDGER.md); actual verification and limitations are in [validation06](architecture/06_VALIDATION_AND_UNRESOLVED_WORK.md).

---

## 2026-11-21 — Address × optimiser factorial result and reference check (additive)

Recorded 2026-11-22T02:15Z by a separate reviewing session. Everything above is unchanged. No decision ID is assigned here; the maintainers of the decision register should assign one if they adopt these rows.

### D-008 follow-up: the multi-seed factorial has been run

Protocol committed before any confirmatory run (`b82d0f0`), results in `767857e`, both on branch `research/memory-saturation-factorial`, draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199) into `research/ontogenesis-phase1`. Five fresh seeds (3101–3105). Late new-fact accuracy, phases 46–60:

| Cell | Range over seeds | Mean | Status |
|---|---|---|---|
| Original addresses, Adam state carried | 0.019–0.029 | 0.024 | Demonstrated: the collapse replicates (F1 held) |
| Hashed addresses, Adam state carried | 1.000 | 1.000 | Demonstrated (F2 held) |
| Original addresses, Adam reset each phase | 0.547–0.694 | 0.635 | F3 (at least 0.70) **failed** on all five seeds |
| Hashed addresses, Adam reset each phase | 1.000 | 1.000 | F4 held: no acquisition effect under hashing |
| Original addresses, Adam with β2 = 0.9 | 0.334–0.482 | 0.419 | F5 partial by its registered rule; nothing claimed |
| Original addresses, plain gradient descent | 0.495–0.648 | 0.585 | F6 (at least 0.70) **failed** on all five seeds |

- This narrows D-008. The one-seed figure "optimiser reset 86.1%" was above every fresh seed. Carried Adam state accounts for about two thirds of the collapse, not nearly all of it.
- Concentrated addresses limit acquisition on their own: with no optimiser state at all, the original addresses reach 0.585 while every hashed arm reaches 1.000.
- The second-moment hypothesis recorded in `INVESTIGATION.md`, Question 7, is not supported as the whole mechanism and is not refuted.
- "With the optimiser reset, the same fixed addresses acquire 86% and retain 83%" does not hold on fresh seeds: 55–69% acquired, 50–66% retained.
- Not predicted, so an observation only: resetting Adam raised late retention under hashed addresses from 0.892 to 0.997 on every seed.
- Limits: a value table trained on arbitrary facts, which an exact lookup table solves; the 0.70 thresholds were set after seeing the one-seed figure; the driver was killed once and three unfinished runs with no output were restarted; timings are unusable. Details in `RESULTS-MEMSAT-FACTORIAL.md` on that branch.

### Reference check of the 2026-11-21 master blueprint (R01–R44)

A separate agent looked up all 44 references of the blueprint at primary sources on 2026-11-21. All 44 were located; none was found fabricated or contradicted, and every headline figure that could be checked matched. This is an existence-and-headline check through a page-extraction tool. It is not a methods audit, and it does not replace the source-specific inspection in `architecture/08_LITERATURE_AUDIT.md`. Corrections to carry:

| Reference | Correction or weaker confirmation |
|---|---|
| Qwen3.5-4B-Base model card | Layer layout and 248,320 padded vocabulary confirmed. The card's file information shows 5B parameters beside the "4B" label; do not use it as a sub-4B reference point without counting. |
| Kimi Linear | "Up to 6×" is decoding throughput at 1M context specifically. |
| CopyGen (copy-constrained decoding) | Listed as a preprint; its arXiv entry says NeurIPS 2026. |
| MEMIT, TRACE, OpenAI SWE-bench Verified post | Minor title errors: hyphenation, truncated title, browser-tab title used instead of the page headline. |
| SmolLM3 | The source gives 11T, 11.1T and 11.2T in different places. |
| Byteification (Nature), Loss of plasticity (Nature), DNC (Nature), TokenPowerBench (AAAI), Sealevel | Publisher page could not be opened; confirmation rests on Crossref, arXiv or a search result. "Continual backprop" for the Nature 2024 paper was not read from a source. |

### Checks of this package rerun independently at `176ece7`

Mathematical examples 12/12, NumPy checks 5/5, forward reference 13/13 and the package validator (PASS, 27 events) reproduced on a separate machine. The 35-case package regression suite was not rerun there. Nothing at `0c0a369` (V4) has been rerun independently.


---

## V5 architecture continuation — November 21, 2026, America/New_York

This additive continuation preserves the complete preceding PR ledger, including the parallel factorial/reference contribution. Reviewed research cutoff: 0afd5629990b4e442fb3aa8f7c700b030d8faede. Separate factorial protocol/result: b82d0f018ebcbe1e6604afc129731d96774d0f69 / 767857e72bfa51e3fa6f32d047507aabef077fde. Existing experiment source/results are retained.

### Current high-level outcomes

| Workstream | Evidence and result | Decision |
|---|---|---|
| Address × optimizer factorial | Five fresh seeds: original carried Adam2.4167%, reset63.5%, hash variants100%; F1/F2/F4 hold, F3/F6 fail, F5 intermediate | Narrow D-008;61.083pp reset contrast and62.596% conditional recovery fraction do not uniquely allocate causes |
| Correction optimizer controls | Finite collateral condition holds across9 runs; fitted efficacy and update counts differ; duplicate entity candidates can enter assignment | Require deployed post-update predictions and exact cost counters; no universal optimizer-irrelevance claim |
| Exact edit geometry | Target-only minimum L1 is4; protected margin mu>=0 costs4+10mu/3; endpoint(2,5) is tied | Correct tie language and scope all minimality to coordinates/metric |
| Store/cache evidence | Store-S reported1.1442→1.1398BPB with1022224980 bytes; unequal crop context and0.0001 baseline differences remain | Qualified result; old scalar control and0.62 cache claim remain invalidated;101M/40M gates pending |
| Causal comparator | New prefix-only, resettable cache with proper distributions and reviewed floating-point handling | Small optional augmented reference; no new language score |
| Native update allocation | Three frozen seeds, four common-state arms; H1means51.32/44.71/43.65/22.22% for current/matched replay/additive/frozen | Preserve mixed/negative primary result; additive replay uses50%extra post-prefix updates |
| Evidence maintenance |85 frozen source copies;43 primary identities;62humanIDs,45events covering39decisions | Full prior history and external contribution preserved |

The native main run executes876 optimizer calls and1236 scored target presentations. Its model has840 learned scalars and a native22-token head. These are synthetic engineering counts; validation reruns are additional work. Replay improves some new-fact/current-fact endpoints while worsening the prespecified one-phase correction mean. The permanently unobserved random-label control is not a generalization benchmark, and long horizons with no eligible events remain undefined.

### New decision identities

D-055 audits the factorial; D-056 separates candidate fit from shared-state deployment; D-057 corrects exact edit geometry; D-058 qualifies store/cache evidence; D-059 implements a causal comparator; D-060 retains all native comparison outcomes; D-061 defines the quality/resource frontier gate; D-062 preserves evidence and parallel contributions. D-054 receives a linked continuation event. The human and machine detail is in [the architecture ledger](architecture/02_DECISION_LEDGER.md).

### Current architecture and next gates

The dense/shared native causal core, analytical optimizer and versioned inference snapshots remain the executable architecture. Optional replay affects training, while optional causal caches affect an explicitly augmented distribution; their state and cost are distinct. No novel backbone, learned transport, adaptive controller or store is adopted. APS/AETHER remain unselected research hypotheses; Nano LM is the project.

The full theory and new results are in [chapter12](architecture/12_OPTIMIZER_FACTORIAL_AND_UPDATE_GEOMETRY.md), [chapter13](architecture/13_CAUSAL_MEMORY_AND_RESOURCE_FRONTIER.md) and [chapter14](architecture/14_NATIVE_UPDATE_COMPARISON.md). Unresolved scientific gates require real data/checkpoint identities, corrected equal-information evaluation, meaningful uncensored horizons, independently tuned baselines and measured quality/resource trade-offs. They are not closed by mathematical examples or artifact validation.

Maintain PR #195 as draft and unmerged. This continuation authorizes no new material spend, external training job, corpus-scale run, protected result rewrite or tag move.


### V5 bounded late-evidence addendum: D-058 revision

After the core V5 snapshot, exact cache-S, cache-M and store-M JSONs were recovered at fixed cutoff c597d5fefdd378c3e5c66a51e7690cdb90853287. Five new source copies supplement the unchanged85-copy manifest; the machine ledger adds one linked D-058 revision, for46events covering39decisions and62human IDs. All45core V5 events remain an unchanged prefix.

Cache-M's four domain gains(0.0012,−0.0003,0.0036,0.0059BPB) satisfy the registered numerical stopping branch at the source-derived40,379,904-parameter size. Full control clearance is not claimed: maximum baseline differences remain0.0001, checkpoint/raw-byte/crop identities are incomplete and foreign controls have unresolved information assumptions. The old0.62 result remains invalidated; these are new qualified results.

Store-M's mixed gain is−0.0014BPB; it does not decide the source's100,684,032-parameter L/approximately101M gate. Cache-S mixed gain is0.0036 and cache-M mixed gain0.0018. Source model/evaluation ASTs and exchange-script blobs match their earlier versions, while the new training sampler is a separate implementation change without a completed repaired-sampler result in this addendum. No corpus or model was run by this audit.

The package now includes a24-group final local gate and90frozen source copies. Publication uses the fixed c597d5f research integration base; later unreviewed artifacts, if any, are separately queued.

---

## 2026-11-21 — Address concentration dose-response (additive)

Recorded 2026-11-22T03:35Z by the separate reviewing session that ran the factorial. Everything above is unchanged. No decision ID is assigned here.

Second protocol on branch `research/memory-saturation-factorial` (draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `c492b0e` committed before any confirmatory run, results `b292420`. Adam reset each phase in every arm, seeds 3101–3105. M of each query's 8 slots is replaced with uniformly random slots; read weights are kept.

| M | Late acquisition, range over seeds | Mean | Mean late retention | Effective slots, range |
|---|---|---|---|---|
| 0 (from the factorial) | 0.547–0.694 | 0.635 | 0.600 | 212–313 |
| 2 | 0.842–0.924 | 0.885 | 0.846 | 369–549 |
| 4 | 0.969–0.995 | 0.984 | 0.955 | 799–1,166 |
| 6 | 0.997–1.000 | 0.999 | 0.987 | 2,599–3,659 |
| 8 | 1.000 | 1.000 | 0.996 | 12,977–13,045 |

- G1 (acquisition never falls as M rises) held on five of five seeds. G2 (the original read weights on random slots reach at least 0.95) held. G3 (a graded response) **failed**: the response saturates early, with half of each query's reads spread recovering nearly all of the gap.
- With the factorial this gives an account of the collapse that holds on five seeds: queries concentrate on a few hundred effective slots of 16,384; carried Adam state turns that into near-total failure (0.02); removing the carried state recovers about two thirds (0.64); decongesting half of each query's reads recovers the rest (0.98).
- Seen after the fact, so a hypothesis: effective slot count tracks late acquisition across seeds and doses. Exploratory seed 3001 has 526 effective slots at M = 0 against 212–313 for the five confirmatory seeds, which is why the one-seed reset figure of 86.1% was high.
- Correction to a figure quoted in several summaries: "two queries with the same relation share 0.60 of their 8 slots" in `memory_saturation.py` is a count of 0.6 slots (0.076 of the 8), not 60%. Pairwise overlap is small; the concentration is in the effective slot count.
- Limits: arbitrary facts that an exact lookup table solves, so this cannot show what random replacement costs a learner that generalises through its addresses (`cab_s.py` is the place to test that); thresholds set after the factorial; mixed addresses at M > 0 are a manipulation of concentration, not an isolation of it; not evidence about a language model.

---

## 2026-11-22 — Rule-world generalisation cost, other sessions' gates, and coordination (additive)

Recorded 2026-11-22 by the session that owns `research/memory-saturation-factorial`. Everything above is unchanged. No decision ID is assigned here.

### Spreading memory reads costs generalisation when addresses carry structure

Third protocol on draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199): protocol `0764fb6` committed before any confirmatory run, results `6333bcf`. `cab.py`'s fixed-address memory builds its addresses from frozen random entity vectors, so it cannot generalise in `cab_s.py`'s rule world at all, and a test there would have been empty. The probe gives the addresses the hidden type (prototype plus noise). The type is given, not learned. Seeds 4101–4105; Adam reset each phase in every arm.

| Arm | Acquisition | Inference on unseen relations | Corrections applied |
|---|---|---|---|
| typed, M = 0 | 0.836 | 0.737 | 0.034 |
| typed, M = 4 | 0.858 | 0.496 | 0.085 |
| typed, M = 6 | 0.959 | 0.246 | 0.244 |
| typed, M = 8 | 1.000 | 0.042 | 0.481 |
| random, M = 0 | 0.622 | 0.106 | 0.196 |

- H1–H3 held. H4, the question the test existed for, **failed** on all five seeds: no dose keeps acquisition at least 0.95 with at least 80% of inference. The best was M = 6, which keeps a third.
- The earlier arbitrary-fact result ("spreading half of each query's reads restores acquisition") is **not a free remedy** once addresses carry structure. For a fixed-address memory, shared addresses generalise but block corrections (0.03 applied); private ones acquire and correct but do not generalise.
- Raised, not answered: whether separating a shared generalising read from a private exact read, with a learned choice between them, keeps both. `INVESTIGATION.md`, Question 8 (write-time routing), is the existing work on that choice.

### Gates reported by another session (nano-lm-d5), recorded here at its request

Not rerun by this session; figures are as that session reported them.

| Gate | Pull request | Reported result |
|---|---|---|
| G07 forward transfer, pilot | [#198](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/198) | Multi-task or sequential fitting gave no transfer to new combinations of known primitives for dense, loop or menu learners: 2.8–4.4% against 4.3% chance, three seeds. The positive control passed. |
| G07 follow-up | [#200](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/200) | Few-shot adaptation training recovered trained families from 2–8 examples at 72–100%, but new combinations stayed at chance (4.4–6.4%). Disproven for these two training signals; training on held-out combinations remains open. |
| G00 local-work inventory | [#197](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/197) | 386 commits on 76 local branches were on no origin branch. All were bundled to the external drive and are being mirrored to a private archive repository. |

### Coordination among the Claude sessions on this machine (2026-11-22)

- `research/ontogenesis-phase1`, `INVESTIGATION.md`, `tpu/`, corpus v2 and the benchmark harness belong to one session, which will also reconcile this file's conflict with `research/ontogenesis-phase1`, keeping every section word for word.
- G00, G06 (in progress) and G07 belong to a second session, on its own branches.
- The memory and addressing line (G03 family) belongs to a third session, on `research/memory-saturation-factorial`.
- All three only add new files on their own branches and append to this file. At most two local CPU jobs per session; heavy local runs go one at a time.
- The `architecture/` package and the V3–V5 sections on this branch were not written by any of these sessions. They are likely from an external assistant that edits this branch on a schedule, which cannot see this coordination. Nothing above its sections has been changed by these sessions.

---

## 2026-11-22 — G06 and G06b: protecting corrected answers (reported by nano-lm-d5; additive)

Recorded by the session that owns `research/memory-saturation-factorial`, at the request of the session that ran these (nano-lm-d5). The tables below are copied from that session's records; this session read the records but did not rerun anything. Draft [PR #202](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/202) into `research/ontogenesis-phase1`, branch `research/g06-nullspace-durability`. G06: predictions `9a6db1d`, scorer `bc56027`, results `dd8923f`. G06b: predictions `5adc2b0`, scorer `b916aba`, results `298e565`. Bilinear learner from the correction-interference pilot, three seeds.

| Arm | Width | Corrections kept 20+ phases | Old facts of corrected entities | Unshown relations of new entities |
|---|---|---|---|---|
| No protection | 24 | 21.5% | 83.5% | 99.7% |
| Project updates off all old entities (OWM, published), α = 1 | 24 | 98.5% | 100% | 66.2% |
| Rehearse every correction, no projection | 24 | 97.4% | 40.4% | 98.4% |
| Protect corrected entities only | 24 | 98.4% | 99.9% | 91.4% |
| Protect the corrected relation's output only | 24 | 93.9% | 92.1% | 99.3% |
| Protect corrected entities only | 96 | 99.6% | 100% | 99.1% |

- Whole-model protection makes corrections durable but costs the rule (G06). Localising the protection to what was corrected keeps both (G06b); at width 96 nothing measured is lost.
- Caveats as that session recorded them: the readout is bilinear, so the projectors are exact; one world; three seeds.
- Relation to this session's rule-world memory test (PR #199): both find that durability of individual facts and use of shared rules compete through shared structure. They are different learners. PR #199's fixed-address memory could not be separated by spreading reads; G06b separates the conflict by restricting which update directions are protected. G06b does not show that a fixed-address memory can be fixed the same way.

---

## 2026-11-22 — Why corrections fail in the typed shared-path memory (additive)

Recorded by the session that owns `research/memory-saturation-factorial` (draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)). Everything above is unchanged. No decision ID is assigned here. Same rule world and typed-address probe as the previous section; seeds 4101–4105; each protocol was committed before its confirmatory run.

| Test | Protocol → results | Result | Status |
|---|---|---|---|
| Shared + private read, summed, no router | `1f05f7e` → `a2f61a5` | The single shared path at learning rate 1e-2 already reaches 0.90 acquisition and 0.79 inference. At that rate the private path adds 0.06 acquisition and 0.18 corrections (0.04 → 0.22) at 0.03 inference. | E1, E2, E4 held; E3 (second path free at the same rate) failed narrowly |
| 5× steps for every phase | `c67b4b7`; interpretation addendum `3c8d0d7` recorded before any result → `84b923c` | Corrections flat (0.034 → 0.035). Acquisition, inference and retention all rose. | S1 held; read as "a uniform budget change does not move corrections", not as a refutation of a budget account |
| Corrections written alone, 200 vs 2,000 steps for the correction pass only | `0126fb7` → `9ee8701` | Alone with the phase's budget: 0.026 → 0.035. With 10× steps: 0.029 → 0.919. Collateral at writing at most 0.022. Late inference on new entities 0.74 → 0.61. | W1–W4 held |

- **Demonstrated, in this synthetic memory:** corrections fail for lack of budget relative to what same-type neighbours have accumulated in shared slots. Conflict inside the phase's objective is not the main cause. Nor is reachability: the corrected values are reachable here. Reachability is the different limit nano-lm-d5 found in a frozen nonlinear decoder (flat from 100 to 3,000 steps; commit `d6e05c3` on `research/g06-nullspace-durability`, development seeds only). The write-alone falsifier came from that session.
- **Cost:** forcing corrections in is cheap at the moment of writing but damages later generalisation, because the write lands in slots shared with same-type entities. The write is already as local as the addressing allows (sparse reads), so a narrower write cannot help. A different place to write, such as a private slot, might.
- **Side finding:** the inherited default (200 steps at 1e-3) under-trains this memory. Comparisons within each test are unaffected, because all arms of a test shared the budget.
- **Limits:** the correction pass is given an explicit edit label; type information is given, not learned; value table, frozen addresses, synthetic world. Not evidence about a language model.

---

## 2026-11-22 — Private-slot corrections (additive)

Recorded by the session that owns `research/memory-saturation-factorial` (draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)). Protocol `9df7ddf` committed before any confirmatory run; results `e265596`. Seeds 4101–4105. Two separate value tables (shared typed slots and private random slots, summed); corrections written alone after each phase.

| Correction pass | Corrections after the pass | Later inference, new entities | Retention |
|---|---|---|---|
| private table, 200 steps | 0.029 | 0.735 | 0.820 |
| private table, 2,000 steps | 0.891 | 0.714 | 0.881 |
| both tables, 2,000 steps | 0.998 | 0.659 | 0.874 |

- P1–P4 held on all five seeds. A correction written to a private slot with a larger budget takes, costs 0.02 of later inference instead of 0.08, and disturbs nothing at writing.
- With the previous section, this closes the line on the typed fixed-address memory in this world. Shared slots carry generalisation and resist corrections. Corrections need a larger relative budget. Written into shared slots they cost later generalisation; written into private slots they mostly do not.
- **Status: demonstrated in this synthetic memory only.** The edit label is given to the learner, and the private table is per-fact storage (an augmented-memory result, not native model knowledge). Type information is given, not learned. Not evidence about a language model.
- **Open, not started:** whether a learner can make this choice itself without the edit label. `INVESTIGATION.md`, Question 8, is the existing evidence there: a detector with no oracle recovered about a third of the oracle's gain.

---

## 2026-11-22 — Label-free routing to private slots (additive)

Recorded by the session that owns `research/memory-saturation-factorial` (draft [PR #199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)). Protocol `6da758f` committed before any held-out run; results `d5e5bab`. Conditions agreed with nano-lm-ad (owner of `INVESTIGATION.md`, Question 8). Seeds 4101–4105.

| Arm | Corrections | Inference, new entities | Acquisition | False positives (regular facts routed) |
|---|---|---|---|---|
| No routing | 0.037 | 0.736 | 0.892 | — |
| Detector, τ = 0.1 | 0.810 | 0.701 | 0.968 | 0.089 |
| Random, count-matched | 0.194 | 0.703 | 0.911 | 0.189 |
| Detector, τ = 0.2 | 0.499 | 0.728 | 0.920 | 0.010 |
| Labelled ceiling (given labels) | 0.891 | 0.714 | 0.887 | — |

- R1 and R2 held (5/5). R3 **failed on its inference clause**: the detector beats random by 0.38–0.64 on corrections on every seed, but its inference equals random's (±0.008). The later inference cost depends on how many facts go private, not which ones. R4 held at τ = 0.1 and failed narrowly at τ = 0.2 (3/5, mean 0.010).
- A confidence-gated disagreement detector, with no edit label, recovers most of the labelled result in this memory: 0.81 against 0.89 corrections at the same inference.
- **Development seed only:** an ungated disagreement detector cannot start from an empty memory. Everything is flagged in phase 0, the shared path never learns the rule, and inference stays at chance (0.048). Q8's leave-one-out detector avoids this by judging each fact against the entity's other facts.
- Limits: stationary world (in a world with rule changes this router would hold on to the old rule; see nano-lm-d5's G04, PR #204); τ chosen on one development seed; type information given; the private table is per-fact storage. Not evidence about a language model.
