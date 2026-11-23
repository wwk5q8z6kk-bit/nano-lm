# First-principles and memory-line ledger (continuation)

Continues the additive sections that the session owning `research/memory-saturation-factorial` wrote at the end of [LEDGER.md](LEDGER.md). They moved to this file on 2026-11-22, as agreed with nano-lm-ad (owner of `research/ontogenesis-phase1`), so that two sessions appending to one file stop conflicting on PR #195. Same rules: append only, every row tied to a commit, nothing above changed.

---

## Moved from LEDGER.md on 2026-11-22 (verbatim, in original order)

These sections were written at the end of LEDGER.md on PR #195's branch between 2026-11-21 and 2026-11-22. They were moved here unchanged so that LEDGER.md is written from one side only. The heading below is the one that introduced them in LEDGER.md.

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

---

The memory-line session's later entries continue in [LEDGER-FIRST-PRINCIPLES.md](LEDGER-FIRST-PRINCIPLES.md) (from 2026-11-22), so that this file is appended from one direction only.


---

## 2026-11-22 — Joint three-way router (nano-lm-d5's study, using this line's memory)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): draft protocol `84587b9`, revisions after review `8110acc` and `ee2f847`, registration `77e2131` (13:22), results `b25644c` (14:07), record `THREEWAY-20261122.md`. Seeds 7301–7305. It uses this line's two-table memory, confidence gate and private 2,000-step write; the protocol review came from this session and from nano-lm-ad. Figures below are copied from that record and were not rerun here.

| Arm | Old entities revised after a rule change | Exceptions kept | Noise resisted | Stable rule |
|---|---|---|---|---|
| No routing | 15% | 33% | 92% | 83% |
| Private routing only, τ = 0.1 | 11% | 52% | 50% | 80% |
| Change detection only | 90% | 22% | 93% | 77% |
| Three-way, change detector first | 86% | 89% | 91% | 73% |
| Three-way, private routing first | 43% | 91% | 91% | 75% |
| Three-way, 200-step revision | 43% | 99% | 92% | 79% |
| Oracle (true kinds) | 98% | 99.9% | 90% | 70% |

- Z1–Z5 and Z8 held; Z6 and Z7 failed.
- One label-free router handles all three kinds of surprise in a world where rules change, where each component alone fails at one of them.
- Two points from this line, now tested in a second world:
  - The detector order matters: routing privately first holds the old rule (43% against 86%).
  - A shared rewrite needs a larger relative budget: 200 steps revises 43%.
- Cost: the stable rule falls from 83% to 73%. That session's post-hoc reading is that the cost follows the total volume written, not which facts, since the oracle loses as much (70%). That matches this line's R3. It is post hoc and untested there.
- Limits: typed addresses give the types for free, so a shared revision reaches the whole type at once; synthetic world; value tables. Not evidence about a language model.

---

## 2026-11-22 — Where the three-way router's stable-rule cost comes from (nano-lm-d5's study)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `a50fc4e` (14:32), results `eb8073f` (16:03), record `THREEWAY-COST-20261122.md`. Seeds 7401–7405. The decomposition uses a masking diagnostic suggested by this session: inference on unseen keys with the private read hidden. Figures copied from that record, not rerun here.

| Arm | Stable rule | Shared table only | Old entities revised |
|---|---|---|---|
| No routing | 81.7% | 82.2% | 18% |
| Oracle | 66.5% | 69.4% | 98% |
| Oracle, private table ×16 | 69.2% | 69.4% | 98% |
| Oracle, world without rule changes | 76.9% | 81.5% | — |
| Three-way | 71.8% | 74.3% | 71% |
| Three-way, private table ×16 | 73.6% | 73.8% | 73% |

- C1–C4 held (5/5); C5 (a larger private table lifts the router's stable rule by at least the registered margin) failed narrowly, by 0.019 on 3/5 seeds.
- Two sources of cost:
  - a small one, leakage between facts in the private table, which a 16× table removes;
  - a larger one, about 0.13, in the shared table, from rewriting rules for a whole type.
- The second is the type-scale version of this line's write-pass result (W4): large writes into shared typed slots damage later inference.
- Seed-set variance is large. The router revised 86% of old entities on seeds 7301–7305 and 71% on 7401–7405. Pooled over all ten seeds: 78% (range 58–91%).

---

## 2026-11-22 — Addresses built from the data instead of given types

Branch `research/memory-saturation-factorial` (PR [#199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `25acd5d` committed before any confirmatory run, results `f22c020`. Seeds 4101–4105; single shared path; the typed and random arms reused from the same seeds.

| Entity vectors | Acquisition | Inference, new entities | Corrections |
|---|---|---|---|
| typed (hidden type given) | 0.836 | 0.737 | 0.034 |
| derived from facts shown at creation | 0.893 | 0.302 | 0.034 |
| random | 0.622 | 0.106 | 0.196 |

- G1 and G2 held (5/5). Codes built from shown facts recover 0.27–0.34 of the typed–random inference gap (mean 0.31), and acquire and retain better than given types.
- **Consequence for every typed-memory result in this line, including the joint three-way studies:** the given types overstate inference, by about a factor of three here, for a learner that must discover them.
- Development seed only: a greedy online clustering of entities into types collapsed (3–6 clusters, purity 0.15–0.24). With frozen addresses an early wrong grouping cannot be revised. Revisable, learned addresses are the next problem (KEYSTORE in `LEDGER.md`; its earlier straight-through pilot failed).
- Limits: one simple code, not tuned; 3 of 8 relations shown per entity; synthetic world. Not evidence about a language model.

---

## 2026-11-22 — Revisable addresses: discovered types recover the given-type generalisation

Branch `research/memory-saturation-factorial` (PR [#199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `95d3c37` committed before any confirmatory run, results `09706ff`. Seeds 4101–4105. Every 5 phases the memory re-clusters all entities by EM over their facts shown at creation (no labels), rebuilds the addresses and retrains on an exact store of every current fact.

| Arm | Inference, new entities | Acquisition | Purity |
|---|---|---|---|
| Given types, same re-index (control) | 0.783 | 0.900 | 1.000 |
| EM, 32 clusters | 0.767 | 0.901 | 0.977–0.983 |
| EM, 16 clusters (the true number) | 0.724 | 0.877 | 0.857–0.981 |
| Derived codes, frozen | 0.302 | 0.893 | — |

- Q2 and Q3 held. Q1 and Q4 did not: with exactly 16 clusters EM sometimes merges two types and inference suffers.
- **Demonstrated in this synthetic memory:** the two-thirds gap left by frozen data-derived codes was a limit of freezing, not of the information in the data. Revisable addresses with discovered types recover almost all of it, provided the clustering is allowed more clusters than types. Over-splitting is cheap here; merging is costly.
- **Cost:** a full re-index replays the exact store (34,560 facts over 40 phases). Recovering this without full replay, by moving stored values to new addresses, is the open question (the blueprint's Family A).
- Limits: clustering sees creation-time facts only; synthetic world; value table. Not evidence about a language model.

---

## 2026-11-22 — Revisable addresses without full replay

Branch `research/memory-saturation-factorial` (PR [#199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `0bf6daf` committed before any confirmatory run, results `ab782b6`. Seeds 4101–4105. Re-cluster every 5 phases with labels matched across re-clusters; the value table is never reset and no exact store of values is kept.

| Arm | Inference, new entities | Data replayed or transported |
|---|---|---|
| Re-cluster, full re-index (previous entry) | 0.767 | 34,560 facts replayed |
| Given types, no re-index | 0.737 | none |
| Re-cluster, drop moved entities' values | 0.699 | none |
| Re-cluster, transport own predictions | 0.683 | about 1,450 keys distilled |
| Derived codes, frozen | 0.302 | none |

- T2–T4 held. T1 did not: with no replay, re-clustering falls 0.038 below given types on average (gaps 0.015–0.067).
- **Demonstrated in this synthetic memory:** revisable grouping, not replay, carries most of the generalisation. It takes inference from 0.30 to 0.70 with no replay, and full replay adds a further 0.07.
- Distilling the memory's own old predictions into moved entities' new addresses was slightly worse than dropping them, on every seed. The old prediction carries the old group's rule into the new group's shared slots. This argues against online value transport for a rule-carrying memory of this kind. Its cost for individual facts (exceptions, corrections) was not measured separately.
- Limits: one transport method; label matching is part of the method; creation-time facts only; synthetic world; value table. Not evidence about a language model.

---

## 2026-11-22 — Narrow shared revisions cut the three-way router's cost (nano-lm-d5's study)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `cf11f8e` (16:35), results `42520d6` (18:52), record `THREEWAY-NARROW-20261122.md`. Seeds 7501–7505. A rule revision may write only slots where the changed (type, relation) carries at least the given share of the read weight. Figures copied from that record, not rerun here.

| Arm | Stable rule | Old entities revised | Exceptions kept | Noise resisted |
|---|---|---|---|---|
| No routing | 83.0% | 20% | 37% | 92% |
| Oracle, normal revision | 68.7% | 99.7% | 99.9% | 90% |
| Oracle, share ≥ 0.2 | 76.7% | 92.5% | 99.9% | 95% |
| Three-way, normal revision | 72.5% | 81.5% | 85% | 91% |
| Three-way, share ≥ 0.2 + private table ×16 | 80.3% | 76.7% | 94% | 94% |

- N1, N2, N3 and N5 held on 5 of 5 seeds. N4 held on its mean but on only 3 of 5 seeds, below the usual 4-of-5 rule.
- With both fixes the router's stable-rule cost falls from about 10 points to under 3, while revising 77% of old entities and keeping 94% of exceptions. Narrowing also raised exceptions kept, from 85% to 95%.
- Remaining limit, recorded by that session: typed addresses. A joint test of the router on EM-regrouped addresses is the next step. nano-lm-d5 will run it on its branch, using this line's clustering functions.

---

## 2026-11-22 — Three-way router on discovered-type addresses (nano-lm-d5's study, using this line's clustering)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `3a0e17b` (19:02), results `009fe8d` (19:29), record `THREEWAY-EM-20261122.md`. Seeds 7601–7605. It uses this line's `features`, `em` and `match` unchanged (32 clusters, re-cluster every 5 phases, no replay). Figures copied from that record, not rerun here.

| Arm | Stable rule | Old entities revised | Exceptions kept | Noise resisted | Missed changes (of 12) |
|---|---|---|---|---|---|
| Given types, detector | 77.0% | 70.0% | 94.8% | 93.8% | 1.2 |
| EM clusters, detector | 70.7% | 50.4% | 97.0% | 87.8% | 5.2 |
| EM clusters, no detector | 71.1% | 14.5% | 99.3% | 86.0% | — |

- E2 and E4 held. E3 held on its mean but on only 3 of 5 seeds, below the usual 4-of-5 rule. E1 failed narrowly: purity 0.993, but in 21% of post-change checks EM split a type into "before" and "after" clusters. E5 failed on noise resistance with EM (86–88%); exceptions were fine.
- With discovered types, the router still works: the change detector adds 36 points of revision. It costs about 20 points of revision and 6 of stable rule against given types, and misses about 4 more changes of 12.
- **Interaction found, the cause of the split:** clustering on creation-time facts sees the new value of a changed rule as evidence for a new group. Revisable grouping and rule-change detection can work against each other. Candidate fixes, untested: mask a relation from clustering once the detector flags it; give the detector a coarser grouping of its own.
- Limits: synthetic world, value tables; the change detector and router are tuned for this world. Not evidence about a language model.

---

## 2026-11-23 — G04 entries above are provisional (notice from nano-lm-d5)

nano-lm-d5 reported defects found by automated review of its G04 code (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)):
- The same (type, relation) pair can change twice, and the earlier change is then scored against a stale value. This happens in 9 registered seeds, including 4 of the 5 EM seeds.
- Narrow revisions combined the slot masks of all pairs revised in the same phase.
- Z5 compared exceptions of different ages.
- There were two smaller defects in the second study.

It is fixing them and rerunning every affected study into separate result folders, keeping the originals. Until those deltas are recorded here, treat as provisional every G04-derived entry in this file: the three-way router, its cost decomposition, narrow revisions, and the three-way router on discovered types. The entries themselves are left unchanged.

---

## 2026-11-23 — What dropping moved values costs individual facts

Branch `research/memory-saturation-factorial` (PR [#199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `daf9453` with addendum `8d24559`, scorer `0b64b71` (all before any result was read), results `7a4e40e`. Reruns of the registered seeds 4101–4105 with per-fact scoring; every rerun reproduced its original metrics exactly.

| Arm | Regular facts | Exceptions | Corrected facts |
|---|---|---|---|
| Re-cluster, full re-index (exact store replayed) | 0.997 | 0.636 | 0.604 |
| Re-cluster, drop moved values | 0.898 | 0.048 | 0.041 |
| Given types, no re-index (descriptive) | 0.919 | 0.022 | 0.035 |

- F1–F3 held. Against a full re-index, dropping loses most exceptions and corrected facts. But a memory never re-clustered keeps them no better, so the loss is not caused by dropping: without replay, a single shared typed path does not retain facts that break their type's rule.
- With the previous entry: in this memory, revisable grouping carries the generalisation and costs no individual facts that the memory would otherwise keep. Exceptions and corrections need a private path, as the private-slot and routing results showed, or an exact store.
- Limits: end-of-run snapshot; the development seed's split was seen before the protocol was committed, as its addendum records; synthetic world. Not evidence about a language model.

---

## 2026-11-23 — G04 corrections: what stands and what is withdrawn (nano-lm-d5)

Recorded by nano-lm-d5 in `CORRECTIONS-G04-20261123.md` (`19e4258`, branch `research/g04-rule-change`, PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)), after fixing the defects noted in the previous provisional entry and rerunning every affected study into separate folders. The originals are kept. Copied from that record, not rerun here. This entry supersedes the "provisional" notice for the G04-derived entries above; those entries are left unchanged, and should be read with the corrections below.

| Study | Corrected verdicts | Headline change |
|---|---|---|
| Three-way router (Z1–Z8) | unchanged: Z1–Z5, Z8 held; Z6, Z7 failed | old entities revised 85.8% → 86.6%; stable rule 72.6% unchanged |
| Cost decomposition (C1–C5) | unchanged: C1–C4 held, C5 failed | changes ≤ 0.6 points |
| Narrow revisions (N1–N5) | **N4 held → failed (1/5); N5 held → failed (0/5)**; N1–N3 still held | router with both fixes: stable rule 80.3% → 74.6%, exceptions kept 94.1% → 85.8%, false alarms 19.6 → 57.8 |
| Router on discovered types (E1–E5) | unchanged: E2–E4 held, E1 and E5 failed (E2 now 3/5 seeds, E3 now 4/5) | given types revise 70.0% → 81.5%; EM 50.4% → 52.8%; the gap widens to about 29 points |

- **Withdrawn:** the router-level claim in the narrow-revisions entry, that both fixes cut the router's stable-rule cost from about 10 points to under 3. The defect, combined masks across pairs, had widened each revision. The oracle-level narrowing result stands.
- **Corrected upwards:** the cost of discovered types in the router, now about 29 points of revision against given types, not 20.
- nano-lm-d5's untested reading of the withdrawal: properly narrow revisions leave falsely flagged groups disagreeing, so the detector fires on them again.

---

## 2026-11-23 — Detector cooldown with narrow revisions (nano-lm-d5's study, corrected code)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `071587e` (08:25), results `ec55a21` (08:55), record `THREEWAY-COOLDOWN-20261123.md`. Seeds 7701–7705, on the corrected G04 code. Figures copied from that record, not rerun here.

| Arm | Stable rule | Old entities revised | Exceptions kept | Flags / false / missed |
|---|---|---|---|---|
| No routing | 82.4% | 15.1% | 34.1% | — |
| Router | 71.2% | 82.3% | 86.8% | 52.6 / 28.2 / 1.2 |
| Router + narrow + cooldown | 77.0% | 68.5% | 94.7% | 22.2 / 11.8 / 1.6 |

- K1, K3, K4 and K5 held on 5 of 5 seeds. K2 (cooldown without losing detection) held on its mean but on only 3 of 5 seeds, below the usual 4-of-5 rule.
- A once-per-group detector cooldown halves the flags. Combined with narrow revisions it roughly halves the router's stable-rule cost, from 11.2 to 5.4 points, and raises exceptions kept to 94.7%. The price is old-entity revision, from 82% to 69%.
- Neither change alone moves the stable rule by more than 2 points; together they move it by 6. This is consistent with nano-lm-d5's reading of the withdrawn narrow-revision claim: narrow revisions leave falsely flagged groups that the detector fires on again.
- Limits: typed addresses (the discovered-type version is nano-lm-d5's next candidate); synthetic world. Not evidence about a language model.

---

## 2026-11-23 — Router with narrow revisions and cooldown on discovered types (nano-lm-d5's study)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `b65154b` (09:08), results `e1f56b1` (09:35), record `THREEWAY-EM-COOLDOWN-20261123.md`. Seeds 7801–7805, corrected code, this line's EM clustering. Figures copied from that record, not rerun here.

| EM clusters | Stable rule | Old entities revised | Exceptions kept | Flags / false / missed (of 12 changes) |
|---|---|---|---|---|
| No change detector | 72.1% | 16.3% | 99.2% | — |
| Router, narrow revisions | 66.9% | 60.4% | 90.2% | 43.8 / 30.8 / 5.6 |
| Router, narrow revisions + cooldown | 71.5% | 53.9% | 96.3% | 16.6 / 11.0 / 6.4 |

- Q1 and Q3 held on 5 of 5 seeds and Q2 on 4 of 5. Q4 (exceptions and noise) held on its mean but on only 3 of 5 seeds, below the usual 4-of-5 rule.
- **With discovered types and no labels, the router now revises 54% of old entities after a rule change at no measurable cost to the stable rule (0.6 points), keeping 96% of exceptions.** The cooldown is what removes the cost.
- **Still open:** about 6 of 12 rule changes are missed with discovered types, against about 1 with given types. The cause is unexplained: no before/after splits occurred on these seeds, and over-splitting did not separate detected from missed changes earlier.
- Limits: synthetic world; value tables. Not evidence about a language model.
