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


## H. V4 current decisions, completed gates and remaining questions

**Current counts:**54 stable human IDs,36 machine events covering31 decisions. The entire27-event V3 log remains an unchanged byte prefix, in addition to the permanent original ten-event pin. D-001–D-023 retain their human-history status; no invented machine metadata was added. Sections above are decision snapshots at their recorded dates; this section records the V4 state.

**Revision of D-046:** the margin follow-up is now a completed upstream three-seed pilot at9d88a2c. The new event points to the prior pending event. The finite M1/M4 conditions hold; compound M2/M3 success conditions fail. Newly committed TPU infrastructure remains source evidence without a completed nested/native result in the inspected compare.

| ID | Current decision | Reason | Evidence state | Next gate |
|---|---|---|---|---|
| D-047 | Adopt finite margin-pilot conclusions without universal retention claims | M1 and finite M4 hold; M2 monotonic direction holds but60% fails; M3 solver success holds but93% old-regular accuracy fails. Stronger protection trades old-regular and old-correction endpoints. | repository; pilot; reviewed in V4 | Fresh-seed paired native comparison with fixed primary horizon and guardrails |
| D-048 | Separate individual write acquisition, phase-end survival and semantic revisions | The source scores corrected_now after all phase writes. Reconstructed collisions and a source-function counterexample show why successful local writes need not survive; same-value assignments also reset revision age. | engineering; pilot; reviewed in V4 | Log each write, actual output, later same-phase writes and explicit revision censoring |
| D-049 | Implement native analytical training for dense and shared causal models | Exact forward parity,62 finite-difference directions and shared-use gradient sums support training mechanics; both parameterizations learn a fixed repeated-token fixture. | engineering; pilot; reviewed in V4 | Recover a legitimate native-data baseline with explicit tokenizer and split identities |
| D-050 | Make optimizer displacement and checkpoint receipts match actual valid state | Independent review reproduced between-parameter and internal-stride aliasing. Disjoint contiguous storage and pre-commit receipt validation now reject demonstrated failures atomically; resume claims are scoped to the tested layout/environment. | engineering; pilot; reviewed in V4 | Preserve actual-step accounting in any future projected or constrained optimizer |
| D-051 | Use immutable common events and complete revision-aware horizon cohorts | The implemented instrument isolates world/replay randomness, delivers visible prefixes, distinguishes same-value revisions and retains failed acquisition in the primary denominator; missing observations remain explicit. | engineering; pilot; reviewed in V4 | Bind every real arm to identical manifests and predeclared censor/endpoint policy |
| D-052 | Audit actual deployed constraints and model outputs separately from solver status | Candidate status is not a certificate. Tests demonstrate cast and addition rounding; original thresholds, target provenance and actual logits are checked separately. | engineering; pilot; reviewed in V4 | Construct the real arm constraints and record post-application values and predictions |
| D-053 | State retention as a conditional cumulative actual-drift bound | Exact bilinear expansion includes address, decoder and interaction terms. A telescoping norm budget below every initial rival margin is sufficient over a finite fixed-semantic horizon; failure of the bound is inconclusive. | mathematical; document_only; reviewed in V4 | Test a costed drift-monitor/protection hypothesis on actual native updates |
| D-054 | Integrate native updates and predictions with correction measurement on a fixed fixture | Eight visible events cause16 actual AdamW steps and16 scored target exposures; real forward outputs, causal state identities and complete horizons are recorded without an accuracy target. | engineering; pilot; reviewed in V4 | Freeze a native-data G03b protocol with fresh seeds, equal experience, broad quality and total cost |

The event log records alternatives, open questions, source identities and actual UTC recording time. The current local training result is a synthetic engineering fixture; it corrects the earlier forward-only status without inventing held-out model evidence. Existing experimental source/results and every previous ledger event remain preserved. The latest source cutoff is9d88a2c; new implementation artifacts are independently hashed until their published commit exists.

## I. V5 core snapshot and append-only continuation

**Core snapshot counts:** 62 stable human decision IDs; 45 machine events covering 39 decisions. All 36 V4 events remain an unchanged byte prefix, as does the complete canonical ledger at PR head 0a1d11f, including its external contribution. The 17 V4 result files remain unchanged. Earlier sections retain their recorded historical counts and claims.

**Revision of D-054:** the small bridge is now exercised by a frozen three-seed, four-arm synthetic update comparison. Its new event links to the existing D-054 event. Primary one-phase correction means favor current-only over both replay arms; some secondary endpoints favor replay. This is progress in controlled measurement, without a natural-language efficacy claim.

| ID | Current decision | Reason and evidence | Next gate |
|---|---|---|---|
| D-055 | Use the five-seed factorial to narrow the earlier D-008 acquisition claim | Original-address reset Adam raises mean late acquisition from2.4167% to63.5%, but F3 and F6 fail, F5 is intermediate, and the62.596% recovery ratio is descriptive rather than unique optimizer attribution. Evidence: repository; reviewed; pilot. | Separate representation and optimizer interventions with explicit occupancy and cost controls |
| D-056 | Separate detached correction fit from actual shared-entity deployment and cost | Duplicate entity indices can commit incompatible candidates without a portable assignment-order guarantee; loop-index and phase-end summaries do not certify individual acquisition or matched optimization cost. Evidence: engineering; reviewed; pilot. | Use ordered or joint entity updates with deployed post-update predictions and exact per-write counters |
| D-057 | Use the exact coordinate-scoped L1 protection trade-off and distinguish ties from errors | Target-only minimizers cost4 and have nonpositive protected margin; requiring margin mu>=0 costs4+10mu/3. The tie endpoint is not an ordering-independent argmax failure. Evidence: mathematical; mathematically_checked; document_only. | Use semantic margin constraints and actual deployed values in future edit comparisons |
| D-058 | Preserve the qualified store-S result and keep invalidated cache/control results excluded | Reported BPB1.1442 to1.1398 uses1022224980 store-array bytes, extra pre-crop context and a baseline mismatch. Scalar-shuffle control and historical cache result are invalidated; registered101M/40M gates remain pending. Evidence: repository; reviewed; pilot. | Run a valid same-prefix, document-aware evaluator with checkpoint/crop/byte identities at the registered scale |
| D-059 | Add an optional causal n-gram comparator with explicit state and numeric contracts | Prefix-only prediction, document/crop reset, score-before-observe, positive smoothed distributions and validated mixing address demonstrated leakage and finite-precision defects. Eight grouped checks and independent review support the small reference. Evidence: engineering; reviewed; pilot. | Integrate only through a same-prefix augmented evaluation with counted state and independently valid calibration |
| D-060 | Retain all outcomes of the frozen four-arm native update comparison | Common weights and Adam states separate update allocation. Replay improves some secondary endpoints but lowers mean primary H1 from51.32% to44.71% or43.65%; additive replay spends50% more post-prefix updates. Evidence: engineering; exploratory; pilot. | Freeze a new adequately long, tuned development/evaluation design with meaningful held-out quality and uncensored horizons |
| D-061 | Require an explicit quality-resource frontier before choosing augmented memory or a new backbone | Memory construction, fitting, state, inference, update cost and data exposure interact; parameter count and a small loss gain do not establish the feasible workload trade-off. Evidence: hypothesis; proposed; not_executed. | Measure paired quality, independent variation, total bytes, latency and energy under concrete workload budgets |
| D-062 | Preserve parallel contributions and exact historical evidence while adding source-specific review | The complete latest PR canonical ledger, full36-event V4 prefix and17 old result files are retained;85 frozen source copies and43 primary-source identities have explicit scope. The external44-reference headline check is attributed, not promoted to methods review. Evidence: engineering; reviewed; document_only. | Validate the complete package, publish with an expected-head lease and retain a separate publication receipt |

The new events include actual UTC recording time, alternatives, open questions, immutable source commits and local evidence hashes. A repaired historical evaluator does not retroactively validate its prior numbers: corrected evaluations receive new receipts. Failed primary outcomes, missing/undefined cohorts and overheads remain visible.

**Relationship to the parallel contribution.** The external canonical append reports a five-seed factorial, a 44-reference headline check and an independent V3 subset rerun. V5 audits the factorial from exact raw files, records its literal finite verdicts, and retains the other contribution's limited attributed scope. The old V3 subset check is not relabeled a V4/V5 integration run, and overlapping bibliography counts are not summed.


## J. V5 late evidence addendum — current status after the core snapshot

**Current counts:** 62 human decision IDs,46 machine events covering39 decisions. The preceding45-event V5 core log is preserved byte-for-byte; one new D-058 event links to its earlier V5 event. The85-copy core source manifest remains unchanged and a separate five-copy addendum brings the inspected source-copy total to90. Earlier count/status statements remain dated snapshots.

At fixed late cutoff c597d5fefdd378c3e5c66a51e7690cdb90853287, cache-S/cache-M/store-M result files are now available. The approximately40M cache result has domain BPB gains0.0012,−0.0003,0.0036 and0.0059, all below the registered0.01 stopping threshold. The numerical stopping branch is met; the repeated0.0001 baseline discrepancy and control/provenance qualifications prevent calling the full evaluation contract cleared. This is a finite result for this cache recipe, not a proof against all copying or external memory.

The M store reports a mixed-BPB worsening of0.0014. Its source configuration contains40,379,904 parameters; it cannot decide the separately registered approximately101M/L store gate, whose source configuration contains100,684,032. The S configuration contains14,292,224 parameters. These are source-shape counts, not independently loaded checkpoint inventories.

The historical0.62 cache result remains invalidated. The new results are separate evidence, with a recorded passing source causality check whose execution was not independently reproduced on the corpus. New host-sampling and distillation infrastructure is preserved but does not establish a completed corrected-sampler training result. See [the late source review](reviews/store_cache_late_review.md) and [chapter13 addendum](13_CAUSAL_MEMORY_AND_RESOURCE_FRONTIER.md#7-late-evidence-addendum-at-c597d5f).


## K. V6 current decisions — November 23, 2026

The current register contains **73 human decision IDs**, **58 machine events** and **50 machine-covered decision IDs**. The 46-event V5 machine prefix is byte-identical. Eleven new decisions D-063–D-073 and one new event for D-058 extend it. The 23 older human-only IDs remain historical narrative; V6 does not backfill invented machine provenance.

The current full design is [00_FULL_ARCHITECTURE.md](00_FULL_ARCHITECTURE.md). New source evidence is in [15](15_NATIVE_BASELINE_AND_RESEARCH_UPDATE.md), the generative instrument in [16](16_GENERATIVE_REASONING_EVALUATION.md), and corrected temporal/protection contracts in [17](17_CORRECTION_MEASUREMENT_AND_PROTECTION.md).

| Decision | Current decision | Reason and evidence boundary | Next gate |
|---|---|---|---|
| D-063 | Use the audited native 386M recipe as the current dense reference while retaining the own-foundation-model direction | The reported 386,062,080-parameter run completes 7,698,841,600 token presentations and improves all seven recorded plain accuracies over 211M. Its recipe components change together, so no component receives causal credit. | Recover checkpoint/tokenizer/data identities and compare native candidates under a fixed quality/resource protocol |
| D-064 | Correct benchmark denominators and distinguish sampled BPB, continuation scores and generated responses | Raw task counts total 22,348; 0.8049 is a fixed weighted sampled-domain BPB index. Source diagnostics expose missing identity binding and conditional token-boundary risks without demonstrating historical misbinding. | Save item-level tokenizer lengths, target offsets, checkpoint/data identities, omitted-task failures and paired outputs |
| D-065 | Retain shared depth as a candidate and preserve the nested learning-rate sweep mixed verdict | Whole-block reuse succeeds 4/5 versus 0/5, but single-layer reuse 2/5 versus 1/5 fails the required two-success gap. All 60 files reproduce the combined mixed predicate. | Prospectively tuned native dense/recurrent comparison with fresh seeds and exposure plus measured-compute accounting |
| D-066 | Keep retention, finite-procedure forward transfer and representation existence separate | Dense replay retains 100% of experienced tasks yet reaches 2.78% on new combinations. Post-hoc probes also miss known solutions, so finite adaptation failure cannot prove no task vector or representation exists. | Calibrate adaptation on available solutions, freeze fresh families and count 12.5× replay exposure plus task-vector/storage state |
| D-067 | Report completed exposure and unmeasured endpoints separately from hardware plans | Q14 reaches 65,536,000 targets, not its 799,539,200 planned total. 1.05106× speed misses 1.25×; quality and kernel parity are unmeasured. | Require completed-step receipts, peak memory, end-to-end timing and independent quality/parity checks before adoption |
| D-068 | Repair temporal targets and preserve corrected G04 policy trade-offs | A perfect final-truth predictor can fail revise_next5 after repeated changes. The V6 reference binds targets to prediction time and separates early adaptation from final truth; N/K/Q gates remain unchanged because they use other fields. | Integrate the reference into a prospective evaluator, retain per-query outputs and add semantic epochs plus schedule-matched router controls |
| D-069 | Treat finite-alpha G06 protection as soft attenuation and require fixed-address evidence or a joint drift bound | All eigenvalues alpha/(sigma^2+alpha) are positive; normalized trace is mean attenuation. Actual W displacement is filtered, but old-address invariance is unenforced and unmeasured. Existing corrected bilinear retention remains scoped evidence. | Log actual protected-row and decoder drift, numerical residuals, margins and optimizer-state behavior before causal attribution |
| D-070 | Add a locally frozen prompt-only generative evaluation with complete failure denominators | 96 worlds and 480 records separate direct retrieval, two-hop composition, same-query counterfactual answers, order invariance and missing-information abstention. Real model eligibility requires actual tokenizer counts and no silent cropping. | Bind an actual native checkpoint/tokenizer/decoder adapter and score its complete predictions under the unchanged protocol |
| D-071 | Bind the scored dataset and token receipts and preserve the independent-review defect history | Independent review found mutable in-memory dataset identity, first-newline bypass and mutable token-buffer gaps after 27 tests passed. Repairs pass 31 focused and 21 independent checks without changing generated data or control metrics. | Verify real adapter decoding, stopping and state reset; retain actual model outputs and hashes for independent replay |
| D-072 | Describe overlap checks by exact coverage and keep unmeasured contamination explicit | A one-word shift creates 18 shared 13-word windows with zero stride-seven matches. This demonstrates the filter coverage limitation, not observed corpus leakage; disjoint generated worlds do not certify pretraining cleanliness. | Version corpus and overlap rules, retain per-item evidence and exact matching coverage before clean-subset claims |
| D-073 | Publish a standalone full architecture while preserving V5 history and the relocated workstream ledger | The main architecture now has one direct entry point. The branch owner moved this workstream into LEDGER-FIRST-PRINCIPLES.md; V6 appends to that current prefix and keeps all 46 prior machine events and 27 prior result files. | Validate exact files, publish with an expected-head lease, keep PR195 draft/unmerged and verify Notion/Drive copies |
| D-058 | Close the tested 101M LDO store complement and 40M cache line under their registered numerical stopping branches | Corrected 101M LDO gain 0.0017 is below 0.01; all 40M cache domain gains are below 0.01. These replace V5 pending statuses only for the tested store/context/scale. Historical invalid 0.62 cache evidence stays excluded. | Reopen only under a changed prospectively defined hypothesis with valid same-prefix control, complete state/cost accounting and checkpoint/data identities |

### Ledger ownership and preserved history

At the inspected PR head, the branch owner had moved this workstream's historical contributions verbatim into `LEDGER-FIRST-PRINCIPLES.md`, leaving `LEDGER.md` with one writer. V6 preserves the entire current first-principles prefix and appends its release pointer there. It leaves the current shared ledger unchanged. The prior V5 prefix guarantees retain their historical path/date scope; the relocation is not concealed by restoring an obsolete file.

All 27 V5 result files remain byte-identical, as do existing scientific implementation files. The V6 initial generation receipt and its exact source are also preserved after independent review found and repaired three defects. Current results identify the repaired scorer; they do not retroactively label the initial code defect-free. No candidate is adopted, no raw G04/G06 score is rewritten, and no primary native-model performance is attributed to instrument controls.
