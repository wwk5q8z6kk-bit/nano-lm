# 02 — Nano LM decision, evidence and hypothesis ledger

**V3 current-status index — 2026-11-21.** Decisions D-001–D-033 remain as historical rows. D-034–D-046 below record this continuation; machine events also explicitly revise D-026, D-032, D-044 and D-045 without deleting their earlier events. Read [09](09_EVIDENCE_REFRESH_AND_PROTOCOL_REPAIRS.md) for the completed store/correction studies and new control-flow findings. The reviewed architecture is still unselected. A review event is neither a training result nor permission to run one.

**Ledger snapshot:** 2026-11-21 | **Change policy:** append-only decision history; explicit supersession, never silent rewrites.  
**Existing authority:** [`research/frontier/ontogenesis/LEDGER.md`](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/LEDGER.md) on `research/ontogenesis-phase1`. This is a **proposed reconciled decision supplement**, not a replacement for that canonical research ledger or the original protocols. It may be integrated through review.  
**Source hierarchy:** registered raw results and code > later reviewed corrections > original experimental reports > transcripts > hypotheses and prose blueprints > outside analogies. Repository contents read through the GitHub connection; local running jobs and unpublished worktrees were **not** directly inspected.

> **V2 correction notice:** The decision rows below are a historical snapshot, not universally complete executable evidence. **Do not read 'mathematically checked' as 'empirically proven'** or 'experiment specified' as 'experiment executed'. The existing `research/frontier/ontogenesis/LEDGER.md` remains canonical once the draft PR is merged. The machine-readable new decision events in [`ledger_events.jsonl`](ledger_events.jsonl) are an audited supplement, not a replacement for frozen protocols. Consult [the V2 formal review](05_GAP_CLOSURE_AND_FORMAL_REVIEW.md).

## A. One-page decision register

| ID | Question | Decision made now | Why / alternatives | Evidence classification | State and next test |
|---|---|---|---|---|---|
| D-001 | What is being built? | One native-trained general-purpose Nano LM remains the destination | Product-like external tools are separately scored | Repository charter | Retain |
| D-002 | What is the core architecture? | **Not selected**; start with a strong dense control and competing hybrid/recurrent alternatives | M21 and earlier theories have not won robustly | Repo experiment + prior-art | Open; G0–G4 |
| D-003 | May Nano LM be renamed APS/ONTOGENESIS? | **No**; APS is a test hypothesis, ONTOGENESIS a research label | Avoid confusion between projects and theories | User/repo charter | Retain |
| D-004 | Can we claim 2B–4B frontier quality now? | **No**; make size contingent on real scaling evidence/funding | Public 3B training uses many more tokens and resources | Training reference / resource reasoning | Open; scale gate |
| D-005 | Is M21 the backbone? | **No promotion** unchanged | Dense 4.65 vs M21 5.02 bits/token on 5M code time-proxy comparison; M21 had copy advantage | Merged [PR #194](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194) | Conditional-copy arm, not deletion |
| D-006 | Was DEMOS faster-learning than dense? | **Earlier claim retracted** | Tuned dense reached 99.8% vs DEMOS 98.3%; original dense LR mis-tuned | Repo `INVESTIGATION.md` / transcript | Preserve as cautionary example |
| D-007 | Did exceptions cause regular-fact forgetting? | **No general support**; refuted in tested toy world | Corrections alone reduced old regular recall to ~91%; exception-only ~99.6% | Refereed `DISCOVERY-CYCLE-1` | Promote correction/replay tests |
| D-008 | Is fixed-address learned memory capacity exhausted? | **Not demonstrated** | 16,384 nominal, ~526 effective; hash addresses and Adam reset restore acquisition by different means | `INVESTIGATION.md` one-seed pilot | Need beta2, multi-seed and occupancy controls |
| D-009 | Do learned write-time detectors work? | **Partially**; neither impossible nor solved | Inference 81.4 vs 77.6 baseline vs 88.9 oracle; only 3 seeds | `routing_loo.py` result | Dynamic rule-change test and cross-seed replication |
| D-010 | Do normalization+weight decay solve forgetting? | **No** | Maintained speed, old-task accuracy near chance; distinct endpoints | `plasticity.py`, published results | Replay/continual BP comparator |
| D-011 | Can exact records be included? | Yes **only as explicitly counted augmented system option** | Provides a correct history/measurement baseline, not intrinsic LM knowledge | Mathematical proposition D | Full byte/joule accounting |
| D-012 | Is representation drift inherently forgetting? | Not always | A known invertible transport and adjusted readout preserve outputs exactly | Algebraic proposition A | Learned transport experiment needed |
| D-013 | Does first-order constrained updating solve continual learning? | No; it has a limited local guarantee only | Nullspace update may block new learning entirely | Algebraic proposition B | Compare replay, old anchors, compute |
| D-014 | Can arbitrary independent facts be encoded with negligible memory? | No under lossless worst-case assumptions | Pigeonhole lower bound $n\log_2 V$ | Mathematical proposition C | Count external storage and side information |
| D-015 | Is 3:1 hybrid a default? | No, reference competitor only | Qwen3.5 data is strong but ICML 2026 shows scope-limited expressivity separation | [Qwen docs](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5), [ICML](https://proceedings.mlr.press/v306/ye26h.html) | Mixer placement tournament |
| D-016 | Is adaptive looping always efficient? | No | Extra steps use real FLOPs and memory; looped LM pilot reportedly regressed | Repo `INVESTIGATION.md` + LOTUS | Equal joule and written scratchpad |
| D-017 | Which metric is primary? | Vector of verified quality, learning/retention, accuracy per joule/latency | A single 'intelligence per parameter' can hide expensive off-model work | Defined objective, hardware papers | Preregister per workload |
| D-018 | Are novel operators required? | No; novelty is subordinate to verified result | Previous proposals failed review; published work can be used | Historical research | Equal-cost baseline gate |
| D-019 | What can we prove now? | Five scoped mathematical/algorithmic properties; **zero** LM superiority theorems | Proofs have explicit assumptions and counterexamples | `01_ARCHITECTURE_AND_THEOREMS.md`, deterministic checks | Independent math review |
| D-020 | Should paid scale training begin? | No new material spend on this task | Need baseline, provenance, confirmed budgets, and authorization | `docs/ACTIVE_NOW.md` | Future experiment-scoped approval |
| D-021 | How to maintain the ledger? | Record every proposal, experiment, review and reversal by ID | Multiple research branches share overlapping names | Existing canonical ledger | Review PR, keep source of truth singular |
| D-022 | How to deploy this package? | Safe additive documents and tests, no destructive changes to research | Protect pending registered experiments and uncommitted worktrees | Repo inventory | Draft PR, review before merge |
| D-023 | How many anchors uniquely identify an unrestricted linear feature migration? | Full-span paired anchors suffice and are necessary for the general linear map, absent structural assumptions | Elementary rank proof; 16D five-seed constructed example, not LM training | Derived + synthetic check | Test structured low-rank/orthogonal transport vs replay and true write/energy cost |

## B. Contradiction and correction chain (never delete the original claim)

| Event | Initially plausible interpretation | Corrective evidence | Current interpretation | What remains unresolved |
|---|---|---|---|---|
| C-01 DEMOS one-hop | Novel learning-speed advantage | Tuned dense 99.8% vs DEMOS 98.3% | Baseline optimization artefact in this comparison | DEMOS's other structural uses not established |
| C-02 M21 copying | Better neural core | Dense lower overall bits/token at time-oriented code budget | Specialization may help copy subset but harms other tokens | Whether a gated, lightweight copy branch wins at equal cost |
| C-03 Mixed two-hop failure | Fundamental representational incapacity | Dense/looped have late learning jumps; DEMOS implementation has a proven first-order hop defect | Mix of implementation geometry and optimization difficulties | Full confirmatory 5+ seed outcomes; non-toy transfer |
| C-04 Irregular-content interference | Exceptions damage other facts | Exception-only normal retention ~99.6%; correction-only ~91% | Correction updates are principal observed damage | Adam vs missing replay vs both |
| C-05 Memory full | 16k slots exhausted | Effective ~526; resetting Adam history late acquisition 86.1%, hash 99.9% | Interacting congestion and optimizer state | Multi-seed, β2, Adam vs SGD, state snapshots |
| C-06 Self-router fails | Autonomous routing cannot learn | Leave-one-out detector raised inference to 81.4% with false alarms | Some benefit without oracle, still modest/variable | Detect genuine new rules, drift, noise, cost |
| C-07 'Order independence equals no forgetting' | Commuting updates keep facts | An append-only log preserves old facts and remains order-sensitive | No equivalence in general | State specific domain and acquisition conditions |
| C-08 2B affordable | $500 credits comfortably fund large run | Repo itself marked TPU prices unverified; published pretraining uses far more data | Large model needs measured cost and separate authorization | Verified quota, hardware throughput and training target |

## C. Hypothesis registry (this section has no experimental victory claims)

| Hypothesis | Main mechanism | Distinguishing observable | Strongest competing explanation | Registration |
|---|---|---|---|---|
| H-APS-01 Predictive-state transport | Learned reindexing of memories when embeddings change | Better retention–new acquisition frontier after all migration work | Frozen memory + replay | Proposed, unregistered |
| H-APS-02 Update constraints | Gradient components orthogonal to selected old outputs | Lower old-logit drift *without* slower new learning | Same-entity replay, Adam tuning | Proposed, unregistered |
| H-APS-03 Cross-episode procedures | Reused computation survives when demonstrations leave context | Positive forward transfer and systematic OOD generalization | Meta-trained dense + ICL + explicit scratchpad | Proposed, unregistered |
| H-APS-04 Adaptive compute | Conditional extra steps improve verified successes/joule | Positive net benefit after controller overhead | Fixed extra depth, fixed thinking budget | Proposed, unregistered |
| H-APS-05 Selective global access | Learned efficient local state with sufficient exact global communication | Better BPB/J without composition and exact-span failures | Dense/full attention and modern hybrids | Proposed, unregistered |
| H-APS-06 Correction-scope inference | Selectively preserve old relationships under evolving facts | Low collateral old loss and low stale-response rate | Exact versioned store + same-entity replay | Proposed, unregistered |
| H-APS-07 Useful self-organization | Learned operation structure rather than picking a supplied menu | Transfer to new operation family with retained skills | Fixed menu + wider dense + search | Exploratory, needs formalization |

## D. Evidence map with the observed numerical scope

| Evidence ID | Artifact | Measurement and scope | Reliability warning |
|---|---|---|---|
| E-01 | [PR #194](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194) | 5M code, 3 seeds: dense 4.65, M21 5.02 bits/token | Training time calibrated and drifted up to 30% |
| E-02 | [INVESTIGATION.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/INVESTIGATION.md), Question 7 | One seed, original late new-fact 3.4%; hash 99.9%; optimizer reset 86.1% | Changed interventions have multiple effects |
| E-03 | Same, Question 8 | 3 seeds: non-oracle 81.4%, baseline 77.6%, oracle 88.9% unseen inference | Small toy, per-seed variation |
| E-04 | [DISCOVERY-CYCLE-1](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/theory/DISCOVERY-CYCLE-1-20261121.md), section 8.1 | Exceptions-only ~99.6% old regular, corrections-only ~91% | Confounded optimizer versus replay |
| E-05 | [LEDGER.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/LEDGER.md) | Historical discovery archive, positive/negative results and program map | Existing ledger dated 2026-11-20 and not fully synced with latest `INVESTIGATION` |
| E-06 | `checks/formal_sanity.py` and `results/formal_sanity_20261121.json` | Five deterministic numerical checks | **Not** neural training; numerical checks are not proofs or novelty review |
| E-07 | [ACTIVE_NOW.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/master/docs/ACTIVE_NOW.md) | Current gate, compute and data constraints as inspected | External approvals and live prices not verified |

## E. Ledger maintenance protocol

**One decision = one stable identifier, append-only events and explicit supersession.** Each row must capture UTC timestamp, author/reviewer, exact code/data/benchmark/checkpoint commit, evidence status, alternatives considered, decision rationale, affected experiments, permission class, next gate, and reversal link. Do not treat a draft proposal, independent referee note, raw measurement, or merged code as equivalent authority.

Statuses classify evidence at a particular event; they are not one mandatory linear lifecycle. A mathematical claim may remain mathematically_checked, an exploratory run may precede a later preregistration, and an integration needs its own evidence. The schema defines legal labels; scientific review determines whether a label is deserved. A later event keeps the same decision ID, gets a new unique event ID, and points to that decision’s latest preceding event. A supersession names an earlier decision explicitly. Results never disappear when theory changes. Every metric stores numerator, denominator, distribution, seeds, experimental budget, model size, hardware/precision, and evaluation lineage. Distinguish *unknown* from *zero*.

```mermaid
flowchart TD
    N[New hypothesis or empirical anomaly] --> I[Register ID + source and ownership]
    I --> H[Alternatives + explicit falsifier]
    H --> R[Independent mathematical and literature review]
    R --> P[Preregister a decisive matched-cost test]
    P --> T[Execute only within authorized resource gate]
    T --> E[Immutable results, deviations, exact commits]
    E --> D{Does evidence justify adoption?}
    D -->|No| X[Retain rejection or inconclusive verdict]
    D -->|Yes| V[Language/code validation and integration]
    V --> U[New decision with evidence and next gate]
    X --> U
    U --> N
```

**Cadence:** update on every completed meaningful experiment or new verified contradictory source, not by rewriting research documents daily without new evidence. A periodic automation may review the ledger, but it cannot claim that a new result exists unless actual commits/artifacts demonstrate one.

## F. V2 decisions, consequences and evidence gaps (append-only)

The following rows are **review decisions about how to investigate**, not selections of a final Nano LM model. Each is repeated in the machine-readable `ledger_events.jsonl` with a UTC timestamp, status, alternative, reason, evidence URL and next gate; `checks/validate_research_package.py` checks the event contract.

| ID | Reviewed decision | Reason and competing alternative | Evidence status | Next experiment / dependency |
|---|---|---|---|---|
| D-024 | Do **not** treat the original five propositions as LM superiority or lifelong learning proofs | Known bijection, local linearization, storage bound, versioned retrieval, and single-step cost decision are independent elementary results | Mathematics, conditional; [05 §2](05_GAP_CLOSURE_AND_FORMAL_REVIEW.md#2-theorems-exactly-what-has-and-has-not-been-proved) | Independently check assumptions, then G05/G06/G15 |
| D-025 | Require *unknown* representation-map identification and all-dependent-state migration | A known inverse is an oracle and migration cost can exceed replay | Hypothesis not experimentally validated | G05 vs frozen/replay including state/caches and cost |
| D-026 | Compare nonlinear finite-step edits, not just projected Jacobians | Local anchor Jacobian only controls first order; constraint may block acquisition | Conditional second-order bound; direct prior art exists | G06 same-entity replay, damping, trainability |
| D-027 | Define versioning as bitemporal, permission-aware **system-only** option | Latest stored event is neither a medical truth nor a learnable concept; event time differs from ingestion time | Deterministic data structure, no model result | Bitemporal conflict/retraction tests |
| D-028 | Adaptive compute must be budgeted multi-step decision; one-step gate is only myopic | Counterfactual value/calibration and future state matter | Mathematical formulation, open learned policy | G15 vs equal-budget fixed depth and scratchpad |
| D-029 | Continue architecture competition D0/H1/R1/A1/X0; do not freeze APS | Hybrid is established but has task-specific expressiveness limits; dense is currently best verified code baseline | Repository + primary abstracts; model choice open | G10–G16 |
| D-030 | Split plasticity, retention, generalization and positive forward transfer into separate endpoints | The project's normalized MLP retained almost no past facts; an 'adaptable' model can still forget | Repo synthetic pilot; published LLA work | G03/G07 with zero-oracle controls |
| D-031 | Keep true revision/exception/new-rule distinction and no future-label routing | Correction damage refuted the earlier exception hypothesis; LOO detector recovers only part of oracle benefit | Repo 3-seed synthetic result; mechanism incomplete | G04, prequential dynamic change data |
| D-032 | Register all new decision events in a validated append-only machine ledger; retain original human-ledger history | Free-text-only decision tables drift, duplicate metrics and mix hypotheses with results | Procedure implemented in this PR; does not validate historical artifacts | Validate manifest, review merge, independent source audit |
| D-033 | Rerunnable checks must work without writing inside the package | Prior numerical scripts used fixed output paths and failed on protected backing files | Local reproducibility defect reproduced | Run checks read-only, explicit --out to writable location |

**Known blockers requiring a real experiment, not another document:** unfinished/owner-local worktrees; full audits of source-specific licenses and held-out corpora; full papers/implementations beyond primary abstracts; unknown model-scale effects; verified hardware prices/energy; and proof of non-interfering *learning* rather than algebraic transport. These remain open and cannot be closed by the v2 document.


## G. V3 current decisions and reasons

| ID | Decision | Reason | Evidence state | Next gate |
|---|---|---|---|---|
| D-034 | Record the completed store pilot as a one-seed augmented-system result | Post-hoc mixing improves weighted BPB by 0.0581; joint training loses 0.1435 to post-hoc. The 1022224980-byte store and exposure mismatch matter. | repository; reviewed; pilot | Document-excluded store control and isolated runtime measurement |
| D-035 | Separate immediate edit locality from durable retention | Same-entity replay protected measured neighbouring facts immediately, while late corrections remained 62.28% in the three-seed bilinear pilot. | repository; reviewed; pilot | G03b same-event survival measurements and native-model bridge |
| D-036 | Precompute identical exogenous events and isolate replay randomness | The random-replay branch changes future correction draws in 38 of 39 correction phases for all three audited seeds; same-entity and single-write streams remain matched. | engineering; reviewed; pilot | G03b immutable event manifests and independent RNG streams |
| D-037 | Classify solver outcomes and evaluate actual protected targets and cohorts | Nonzero solver status is not a representation impossibility certificate; predicted targets, true targets and later retained edit versions differ. | engineering; reviewed; document_only | Per-event solver receipts, post-dtype residuals, censoring and denominators |
| D-038 | Instantiate a small executable dense/shared-block architecture contract | Tensor flow, logical-depth caches, tied-head count and token/reset rules turn the schematic into a reviewable implementation. | engineering; reviewed; document_only | Forward correctness first; then independently recovered native training baseline |
| D-039 | Restrict representation transport to coherent model transformations | Generic invertible functional conjugacy does not automatically preserve fixed normalization, rotary coordinates or tied embedding/readout parameterizations. | mathematical; reviewed; document_only | Boundary probe, compatible transforms, conditioned map fitting and measured rewrite cost |
| D-040 | Protect outputs against the realized optimizer displacement | Preconditioning, momentum and decay can move a projected raw gradient out of the protected nullspace; nonlinear finite updates also have a remainder. | mathematical; reviewed; document_only | G06 actual-step residual, new-learning and targeted-replay comparison |
| D-041 | Define historical reads at both valid time and recorded cutoff | A backdated later insert can change a single-time as-of answer; immutable recorded cutoffs preserve the eligible historical event set. | mathematical; reviewed; document_only | Explicit authority, interval, retraction and permission policy tests |
| D-042 | Use finite candidate hypotheses and distinct transfer metrics | One failed candidate cannot refute an existential architecture conjecture; sample-efficiency transfer need not improve monotonically on every task. | hypothesis; proposed; not_executed | Fresh-family paired learning curves and prespecified cost/retention guardrails |
| D-043 | Validate an extensible append-only event history with explicit revision pointers | The old validator rejected later events for an existing decision and hardcoded exactly ten decisions; full schema and prior-log prefix checks now support maintenance. | engineering; reviewed; document_only | Validate each update against the complete preceding Git ledger |
| D-044 | Charge the actual supervision, plugin and checkpoint dependencies of literature baselines | LOTUS uses gold intermediate supervision; CopyGen trains a plugin; nominal model names do not determine full tensor counts. Direct OGD and model-editing precedents are required. | engineering review of mixed primary evidence; reviewed; document_only | Pin implementation/config/license and reproduce selected comparator |
| D-045 | Attach check receipts and code hashes and distinguish diagnostic from training evidence | The old curvature check was tautological; final review also repaired stale-cache, unsigned-loss and fractional-byte defects. The final 13-case receipt remains scoped to tiny forward correctness. | engineering; reviewed; document_only | Final package checks and review of exact pushed source |
| D-046 | Preserve the new margin protocol and revised device plan as pending work at a fixed cutoff | The research branch advanced to 8d3b03e; a registered protocol is not a completed result. G03b remains a separately labeled repair of event/solver/cohort measurement. | repository; reviewed; document_only | Await immutable results; keep registered rules and device-consistent controls |

The machine log contains 27 events covering 23 decisions. The older D-001–D-023 remain human-history snapshots; no fictitious machine timestamps, experimental hashes or reviewers were backfilled for them. The human register contains 46 stable decision IDs. D-026, D-032, D-044 and D-045 each have a second explicit event; this is legitimate history, not duplication. The final D-044 event corrects an overbroad peer-reviewed label without deleting the earlier record. The final D-045 event records the three prototype defects and links the 13-case receipt; the earlier 12-case receipt remains an intermediate snapshot.

### Event maintenance contract

Before appending, retain the exact previous Git ledger bytes. New records use `schema_version: 2`, an actual UTC recording time, `previous_event_id`, provenance, alternatives, open questions and a next gate. For a new decision the previous pointer is null; for an existing one it must name that decision’s latest event. `source_commit` identifies inspected source, not an imaginary commit containing a future result. Local evidence artifacts carry SHA-256; remote source truth and permissions are not inferred from a well-formed string.

Run `python3 -B checks/validate_research_package.py --baseline-ledger /absolute/path/to/previous-ledger.jsonl`. This checks the entire old file is an unchanged byte prefix. The permanent [history pin](ledger_history.json) separately protects the original ten live events. Check both the human decision row and machine event, then commit them together with evidence. Preserve original experimental protocols and results; a repaired experiment receives a new ID and a deviation/relationship record.

**Maintenance cadence:** update on new evidence, reviewed contradiction, implementation change or a resolved gate. A periodic review may find no change; it must not create invented research progress. The automation’s existence does not imply a background experiment has been run.
