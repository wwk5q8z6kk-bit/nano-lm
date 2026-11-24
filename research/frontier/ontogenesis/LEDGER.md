# ONTOGENESIS research ledger

Sections written by the memory-line session (research/memory-saturation-factorial, PR #195) were moved verbatim on 2026-11-22 to [LEDGER-FIRST-PRINCIPLES.md](LEDGER-FIRST-PRINCIPLES.md), so that this file has one writer.

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

## Results from other sessions, 2026-11-22 (checked here against their committed files, not re-run)

| Item | Status | Evidence |
|---|---|---|
| In a world where rules change, a label-free "seen twice" write policy keeps exceptions (100%) and resists one-off noise (96%), but revises a changed rule no better than plain training: 52% of old entities give the new value | Demonstrated, toy (three seeds; predictions shaped on a development seed, so a replication, not a discovery) | G04, session d5, draft PR #204, `RULE-CHANGE-20261122.md`; predictions `e4dd5fd`, scorer `780b934`, both before the registered runs |
| An oracle that writes the revision into the shared decoder with addresses frozen reaches 96–100%; retraining old addresses instead does worse than plain on new entities (57% against 69%) | Demonstrated, toy; the oracle uses the world's hidden types, so it is an upper reference, not a candidate | same |
| Consequence for Question 8: a detector that sends every surprise to private storage would also miss rule changes; routing needs a third outcome, "revise the shared rule" | Inferred | same, with `results/routing_loo/` |
| **Corrected figures:** old entities revised 92.8% (was 86.9%), seen-twice 60.8% (was 58.2%), ceiling 97.1% (was 92.9%); stable rule 95.5%; false alarms 23 per seed (was 48); misses 1.7 (was 2.3). Verdicts unchanged (Y1–Y4 held, Y5 failed). *Original text:* A label-free trigger for that third outcome: group entities by the model's own answers; when two or more new entities agree on a value the group's older members do not give, revise the decoder with addresses frozen. Old entities revised: 87%, against 58% without it and 93% for the hidden-type ceiling. Costs 1.3 points on stable rules; exceptions 100%, noise resisted 97%. It misses 2–3 of 12 changes per seed, and false alarms are frequent (29–70 per seed) | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy (seeds 7201–7203; threshold tuned on a development seed and declared; Y1–Y4 held, Y5 "at most one miss" failed) | G04 second study, session d5, PR #204, `RULE-CHANGE-DETECT-20261122.md`; predictions and scorer `4159de4` before the runs; means recomputed here from `results/rule_change_detect/verdict.json` |
| **Corrected figures:** revision 86.6% (was 85.8%); everything else unchanged; verdicts unchanged. *Original text:* One router with three outcomes (revise the shared rule, write private, write shared), with the change detector run first: revises 86% of changed rules for old entities, keeps 89% of exceptions and resists 91% of noise. Each detector alone fails one outcome: private-only revises 11% and resists 50% of noise; change-only keeps 22% of exceptions. Reversed order (private first) revises only 43% and misses 6.2 of 12 changes. Exceptions inside falsely flagged groups are kept at only 20% | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy (seeds 7301–7305; gate tuned on a development seed; typed addresses give the memory the types; Z1–Z5 and Z8 held, Z6 and Z7 failed) | Three-way study, sessions d5 and first-principles, PR #204, `THREEWAY-20261122.md`; registered `77e2131` with the reversed-order arm and the reverse-failure measure; table checked here against `results/threeway/verdict.json` |
| **Corrected figures:** unchanged. *Original text:* Costs and limits of that router: inference on stable rules falls from 0.83 (no routing) to 0.73. The labelled oracle loses as much (0.70), and the cost follows total write volume, not where false alarms land (Z6 failed). A random private choice of the same size does almost as well (exceptions 84% against 89%; Z7 failed), because exceptions reach private storage on their second sighting anyway: the private-slot detector adds little here | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy | same |
| **Corrected figures:** unchanged (differences at most 0.006); verdicts unchanged. *Original text:* Where that stable-rule cost comes from (oracle router): (1) collisions in the private table leak into reads, about 0.03 (C1), removed by a private table 16 times larger (C2); (2) writing rule-change revisions into the shared table damages other rules, 0.128 measured on the shared table alone with the private read masked (C3), and 0.007 without rule changes (C4). So the cost is a write cost of type-wide revision, not a detector cost. A ×16 table lifts the router's stable rule by only 0.019 (C5 failed narrowly, 3 of 5 seeds) | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy (new seeds 7401–7405). Seed-set variance is large: the router revised 71% on these seeds against 86% on 7301–7305 | Session d5, PR #204, `THREEWAY-COST-20261122.md`; registered `a50fc4e` before the runs; checked here against `results/threeway_cost/verdict.json` |
| **Corrected figures:** only seeds 7301–7305 changed (revision 0.858 → 0.866), so the pooled revision is about 79–80%. *Original text:* Three-way router pooled over ten seeds (7301–7305 and 7401–7405), mean [min–max]: old entities revised 78% [58–91] (oracle 98% [91–100], no routing 17% [11–27]); exceptions kept 89% [84–92]; noise resisted 91% [88–95]; stable rule 72% [70–75] against 82% [81–85] with no routing. Revision is the unstable figure; the others are tight | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy; the headline figures for this router | Session d5, `THREEWAY-COST-20261122.md` addendum `5ceb18b`; seed-set means consistent with the two verdict tables checked above |
| **WITHDRAWN for the router.** The original result depended on the bug that OR'd revision masks across pairs, which widened revisions. Corrected: the fixed router's stable rule is 74.6% (was reported 80.3%), exceptions 85.8% (was 94.1%), false alarms 58 per seed (was 20), revision 81.3%; N4 and N5 now fail. The oracle rows stand: share 0.2 gives revision 92.0% and stable 76.8%; N1–N3 held. *Original text, no longer valid for the router:* The router's stable-rule cost mostly closed: a shared revision writes only slots where the changed rule carries at least 20% of the read weight, with a ×16 private table. Router: stable rule 80.3% against 83.0% with no routing (was 72.5% on the same seeds), old entities revised 77%, exceptions 94%, noise resisted 94%. Narrowing costs some revision on these seeds (82% → 77%). Oracle at share 0.2: +0.09 on the shared-only readout, revision 92.5% | **Corrected 2026-11-23: router result withdrawn** (`CORRECTIONS-G04-20261123.md`, `19e4258`; checked here). Demonstrated, toy (seeds 7501–7505; N1–N5 held, N4 on 3 of 5 seeds; share and gate tuned on a development seed; typed, frozen addresses) | Session d5, PR #204, `THREEWAY-NARROW-20261122.md`; registered `cf11f8e` before the runs; checked here against `results/threeway_narrow/verdict.json` |
| **Corrected figures:** given types revise 81.5% (was 70%), stable 72.4% (was 77%); EM clusters revise 52.8% (was 50.4%), stable 67.5%, noise resisted 84.9%, misses 6.2 (was 5.2). The EM gap is about 29 points of revision, not 20. Verdicts unchanged (E2 on 3 of 5 seeds, E3 on 4 of 5). *Original text:* The same router with entity types learned from data (EM clusters) instead of given. Old entities revised 50%, against 70% with given types and 14% with no detector. Exceptions kept 97%. Stable rule 0.71 against 0.77 with given types; noise resisted 88% against 94%. The typed-address caveat is therefore a measured cost (about 20 points of revision, 6 of stable rule, 6 of noise), not a failure to transfer. A changed type split into before/after clusters in 21% of checks (E1 failed narrowly, bound 0.20); noise bound missed (E5 failed) | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy (seeds 7601–7605; E2 on 4 of 5 seeds, E3 on 3 of 5; given-type revision 70% here against 77–86% on earlier seed sets, so seed variance is large) | Sessions d5 and first-principles, PR #204, `THREEWAY-EM-20261122.md`; registered `3a0e17b` before the runs; checked here against `results/threeway_em/verdict.json` |
| **Corrected figures:** the pooled typed configuration is about 79–80% (only one seed set changed). The EM configuration revises 52.8%. *Original text:* Seed variance of the router, pooled across identical configurations only. The one configuration run on three seed sets (typed addresses, normal revision, standard private table) revised 86%, 71% and 82% of old entities, pooled 79% [58–91] over 15 seeds. **Five-seed means of this router should be read as ±10 points.** Other configurations, each on one seed set: share 0.2 79%, fixed 77%, EM clusters 50% [31–66]. A post-hoc diagnostic finds EM's extra misses mostly not explained by before/after splits (10 of 26) or by over-splitting | **Corrected 2026-11-23** (`CORRECTIONS-G04-20261123.md`, `19e4258`, files in `results_v2/`, same predictions and seeds; checked here). Demonstrated, toy (pooled); diagnostic post-hoc | Session d5, `THREEWAY-EM-20261122.md` addendum `1e949d5`; the three seed-set means agree with the verdict tables checked above (0.858, 0.711, 0.815) |
| A once-per-group cooldown on the change detector, on corrected code. Flags fall from 52.6 to 25.0 per seed without losing detection (misses 1.0 against 1.2). With narrow revision as well: stable rule 77.0%, against 71.2% for the plain router and 82.4% with no routing, so the cost falls from 11.2 to 5.4 points; exceptions kept 94.7%, noise resisted 92.7%. **Price, not predicted: old-entity revision falls from 82.3% to 68.5%** | Demonstrated, toy (seeds 7701–7705; K1–K5 held, K2 on 3 of 5 seeds; typed addresses; thresholds from a development seed; revision varies about ±10 points between seed sets) | Session d5, PR #204, `THREEWAY-COOLDOWN-20261123.md`; registered `071587e` before the runs; checked here against `results_v2/threeway_cooldown/verdict.json` |
| On EM-learned clusters, the router with narrow revision and cooldown costs no measurable stable rule: 71.5% against 72.1% with no detector. Old entities revised 53.9% (no detector 16.3%); exceptions kept 96.3% (no detector 99.2%); noise resisted 89.4% (88.4%); 6.4 of 12 changes still missed. Narrow revision without cooldown revises more (60.4%) but costs stable rule (66.9%) and exceptions (90.2%). No before/after cluster splits on these seeds. **Remaining limit: revision, about 54%** | Demonstrated, toy (seeds 7801–7805, corrected code; Q1–Q4 held, Q2 on 4 of 5 seeds, Q4 on 3 of 5; thresholds from a development seed) | Session d5, PR #204, `THREEWAY-EM-COOLDOWN-20261123.md`; registered `b65154b` before the runs; checked here against `results_v2/threeway_em_cooldown/verdict.json` |

## Update 2026-11-23 (this branch's own results; overrides earlier rows where they conflict)

Full write-ups with registered predictions are in `INVESTIGATION.md`.

### Measured

| Item | Status | Evidence |
|---|---|---|
| Muon at 101M, 0.8B tokens: rates 2e-3, 4e-3 and 8e-3 give mix 0.9540, 0.9544 and 0.9577, so the rate is flat across this range. Gain over the old default recipe at the same tokens: 0.0235 | Demonstrated (one seed). Q1: "best rate is 4e-3" failed narrowly; "gain at least 0.02" held | `results/scale2/`, Question 13 step 1 |
| 386M dense model, Muon, 7.7B tokens of corpus v2: held-out mix 0.8049 (web 0.888, textbook 0.531, math 0.871, code 0.602) | Demonstrated (one seed). Q2 held (predicted 0.79–0.83); Q3a held (beats our 211M on ARC-Easy 0.596 and PIQA 0.678); Q3b failed as worded (ARC-Easy 0.075 above Pythia-410M, outside the band on the high side) | `results/scale2/q13-verdict.json`; scorer committed before the result |
| Against SmolLM2-135M (about 260 times more training tokens, from memory), on the same harness: behind or equal on every task, ahead on none. Within about 0.02 only on tasks where context adds little (PIQA, HellaSwag, ARC-Challenge); clearly behind on ARC-Easy (0.048) and LAMBADA (accuracy 0.354 against 0.429, perplexity 32 against 19) | Observed | `results/lm/eval-reference-SmolLM2-*.json`, `INVESTIGATION.md` Question 13 result |
| SmolLM2-135M and 360M scored through the calibrated harness; where the model card gives figures, ours are close (from memory) | Observed | `results/lm/eval-reference-SmolLM2-*.json` |
| Nested experiment, per-arm learning rates (3e-4 and 3e-3 added to 1e-3). A reused whole block (4 × 2) learns two-step chains in 4 of 5 at its best rate, against 0 for the untied 8 × 1 of equal compute; a single reused layer against four untied ones is 2 against 1. A lower rate rescues no untied model | Demonstrated, toy. Registered decision "mixed"; L1 and L2 failed, L3 held | `results/nested-lr/verdict.json`, `PROTOCOL-NESTED-LR.md` |
| Training speed: the trainer reaches about 36% of a v5e chip's peak at 386M; data-parallel efficiency over 4 chips about 86%. Recomputing activations is 17% slower. The fused attention kernel needs heads of 128; at 101M with heads of 128 it is only 1.05 times faster | Measured (one chip; Question 14 run c) | `results/bench-speed/`, `results/q14/` |
| Question 14: heads of 128 with fused attention. W1 (at least 1.25 times faster) failed, so heads of 64 are kept; W2 and W3 not measured | Decided by the registered rule | `INVESTIGATION.md` Question 14 |

### Status changes

| Idea or claim | New status | Reason |
|---|---|---|
| Nested dense-loop architecture | Closed for this task: whole-block reuse helps at every rate tried; single-layer reuse not distinguishable from untied | registered test and its learning-rate control both done |
| n-gram store as a core component | Closed, as a complement and as a core component | leave-document-out gains 0.0096, 0.0037 and 0.0017 at 14M, 40M and 101M, below the registered 0.01 |
| "The single learning rate explains the reuse advantage" | Not supported | no arm did better at 3e-4 or 3e-3 than at 1e-3 |
| Fused Pallas attention with wider heads | Not adopted | 1.05 times faster, against the 1.25 required |
| Our models compared with Pythia-160M/410M | Superseded by the SmolLM2 comparison | Pythia is a low bar for education-filtered training text |

### Spend, 22–23 November (list-price upper bounds; billing not yet read; network egress not included)

| Item | Upper bound |
|---|---|
| Question 13: on-demand v5e-4, about 2.1 hours | about $10 |
| Question 13: spot v5e-4, 20.8 hours (billed at the lower spot rate) | at most $100 |
| Speed benchmark: spot v5e-1, about 15 minutes | under $1 |
| Question 14: spot v5e-4, at most about 4.6 hours (stopped responding at about 06:05; cause not established) | at most about $22 |
| Nested learning-rate runs: CPU of the Question 13 host | no extra cost |
| SmolLM2 scoring: laptop | none |

All machines and Cloud Scheduler jobs are deleted (checked 2026-11-23, 14:25 UTC).
