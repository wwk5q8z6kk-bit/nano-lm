# Discovery cycle 1: what do we actually know, and what did one cycle find?

Date: 2026-11-21. Independent investigation. Does not alter any registered protocol or running experiment. Not yet reviewed by a referee; treat literature statements as unchecked unless marked.

Labels: **[Established]**, **[Published]** (from memory, unchecked), **[Observed]** (ours), **[Derived]** (here), **[Hypothesis]**, **[Speculation]**.

---

## 1. The problem reconstructed from our own evidence

| What we observed | Where | What it rules in or out |
|---|---|---|
| A network trained on a stream of unrelated tasks kept 1.5–4% of old tasks | `plasticity.py`, 3 seeds | forgetting can be near total |
| The same kind of model, on a stream that shares rules, kept 93.5% with no replay | `cab_s.py`, 3 seeds | forgetting is not a fixed property of gradient learning; it depends on content |
| With no irregular content at all, it kept 99.9% | `forgetting_split.py`, 3 seeds | training order alone causes no forgetting in a stable world |
| Looped models learned two-step chains in 5 of 8 runs, dense in 1 of 8, across two budgets; outcome is a late jump | `PROTOCOL-LOOP*`, registered | tying may help; "dense cannot" is false |
| On real text the looped model was slightly worse | `lm_compare.py`, 1 seed | reuse is not free for language |
| A lookup table scores 100% on our first acquisition benchmark | Derived | that benchmark measures storage |
| A learner that writes by adding statistics kept everything and inferred best | `cab_s.py`, 3 seeds | exact, order-free writes are available when the model is simple |

**What we misunderstood.** We treated forgetting as the central obstacle and as something caused by gradient updates as such. Our own data say it is conditional: almost total for unrelated content, absent for regular content in a stable world. The question "how do we stop forgetting" was too coarse. The better question is "what, exactly, is forgotten, and what does it damage".

## 2. Ten questions we were not asking

1. Is forgetting a property of the learner, or of the content being irregular?
2. Does irregular content damage regular knowledge stored beside it?
3. If retention of regular knowledge is really inference, is "memory" two different things that should never share storage?
4. Which learners give the same result whatever the order of the data, and is that the same as never forgetting?
5. Is there a theorem limiting bounded-size learners that never forget?
6. Can growth of a system be measured in bits per new experience rather than items, and does that number fall with experience?
7. How does a learner tell, at the moment of writing, whether a fact is regular or an exception, before it has the rule?
8. When the world's rules themselves change, what distinguishes "the rule changed" from "this is an exception"?
9. Is the late jump in composition a search failure that supervision of intermediate steps removes, making the architecture question moot?
10. Is our unit of analysis wrong: should claims be about the whole system (model, store, re-indexing work) per unit of capability, never about the model?

## 3. Seven candidate principles

Three (T1, T3, T4) do not start from neural networks.

| | Principle | Mathematics | Strongest objection | Prior art | Would be disproved by |
|---|---|---|---|---|---|
| T1 | **Never forgetting is the same as order-independence.** | [Derived] If updates commute and the item just written is answered correctly, any item can be moved to last place without changing the state, so every item is answered correctly. Conversely recency loss requires order-dependence. | Only for a stationary target and within capacity; "acquisition" may fail as the store fills | Bayesian updating is order-free; conflict-free replicated data types. Probably stated somewhere in continual learning | A commuting learner that acquires each item yet loses old ones |
| T2 | **Bounded size, no forgetting, rich model: pick two.** | [Established theorem, applied] Under regularity conditions only exponential-family models have a sufficient statistic of fixed size (Pitman–Koopman–Darmois). An exact, order-free, fixed-size learner therefore exists only when the world is exponential-family in fixed features. Learned features need the raw data kept, or loss | The theorem is about exact sufficiency; approximate retention is not covered | The theorem is classical; its use as a limit on continual learners I have not seen, unverified | An exact fixed-size order-free learner with learned features |
| T3 | **A memory should cost the surprise, not the item.** | Store each experience coded by the current model; size grows by −log p(x \| model) bits. Total size is the running code length of the stream. Growth per item falls as the model improves; re-coding old items under a better model is consolidation | Coding and decoding cost compute; lossless is more than tasks need | Prequential description length; compression views of learning; neural compressors | Bits per new item not falling with experience in a structured world |
| T4 | **Forgetting is collateral damage from irregular content.** | In a factorised model, an address fitted to an exception is displaced from its type's address by δ. When the shared decoder later moves by ΔW, the entity's other answers change by ΔW·δ, zero for undisplaced addresses | May be specific to factorised models with per-item addresses | Atypical and mislabelled examples are the ones forgotten [Published]. Damage to regular facts sharing an address: not known to me, unverified | Routing irregular facts away leaving the loss in place |
| T5 | Composition fails in search, not representation | Late, seed-dependent jumps; dense jumped at step 25,000 | Largely published | Abrupt two-hop learning, weight tying studies | Supervising the intermediate step not removing the jump |
| T6 | Re-indexing can be amortised | [Established, data structures] Rebuilding all addresses each time the store doubles costs at most 2 rebuilds per item over its lifetime | Addresses are stale between rebuilds | Log-structured stores, array doubling | Accuracy collapsing between rebuilds |
| T7 | A procedure library grows capability at sublinear cost | Reused routines are stored once; cost per task ∝ routines invoked | Finding routines is exponential search | Library-learning program synthesis | — not tested here |

**Connections that carry weight.**
- Statistics → memory: the exponential-family theorem (T2) is a hard limit on what a small never-forgetting system can be.
- Data structures → learning: doubling schedules (T6) bound the cost of changing a representation over stored data.
- Biology: sparse separated codes for episodes and slow extraction of regularities is the same split as T4's routing. That is a correspondence of mechanism, not evidence.
- Thermodynamics: erasing a bit has a minimum energy cost, and an append-only store erases nothing. True, and irrelevant at present hardware, which runs many orders of magnitude above that limit. Not pursued.

## 4. Ranking

| Rank | Principle | Reason |
|---|---|---|
| 1 | T4 collateral damage | Sharp, causal, testable in minutes, and it reframes the central problem |
| 2 | T2 the three-way limit | Tells us what is impossible before we try to build it |
| 3 | T1 order-independence | Gives a design test: measure order sensitivity instead of retention |
| 4 | T3 surprise-sized memory | The only one that answers "more capable without growing in proportion" with a measurable number |
| 5 | T6, T5, T7 | Engineering bound; mostly published; not yet testable cheaply |

## 5. What was run this cycle

### 5.1 What a gradient learner forgets (`forgetting_split.py`, predictions committed in `f473247`)

Gradient learner, current phase only, seeds 5001–5003. Retention by the model alone at the end. [Observed]

| Exception rate | Regular facts, recent | Regular facts, 20+ phases old | Exceptions, 20+ phases old |
|---|---|---|---|
| 0%, no corrections | 100% | 99.9% | — |
| 10% | 99.8% | 92.1% | 65.3% |
| 30% | 99.7% | 92.9% | 74.1% |
| 60% | 99.8% | 90.9% | 78.7% |

- C2 (no irregular content, no forgetting): **held**, 99.9%.
- C1 (regular facts stay at 95% or more, and 30 points above exceptions): **failed.** Regular facts fell to about 92%; the gap was 19–27 points.
- O1 (order alone erodes regular facts by 10 points): **failed**, 7–9 points.
- So neither simple account is right. Without irregular content nothing is lost. With it, irregular facts are lost most **and regular facts are lost too**.
- The additive learner's model held regular facts at 100% at every age and exceptions at about 38%: it never put the exceptions in the model.

### 5.2 Where the lost regular facts are (exploratory, found after the fact)

Old regular facts on entities with no irregular fact: 100%, 99.9%, 100%. On entities that also carried an exception or correction: 78.0%, 78.9%, 80.4%. [Observed, post hoc, three seeds]

### 5.3 Intervention (`routing_test.py`, predictions committed in `adae995`, seeds 6001–6003)

| Arm | Old regular facts, clean entities | Old regular facts, entities carrying an irregular fact | Inference on unseen relations | Facts sent to exact store |
|---|---|---|---|---|
| Everything trains the model | 99.9% | 78.9% | 78.6% | 0% |
| Irregular facts sent to an exact store (oracle) | 99.9% | 99.6% | 89.2% | 16.5% |
| Self-routing (model decides) | 99.9% | 78.9% | 78.6% | 0% |

- R1 (gap reproduces): **held**, 21 points.
- R2 (oracle routing removes it): **held**, 99.6%.
- R4 (routing does not hurt inference): **held, and more.** Inference rose from 78.6% to 89.2%, at the ceiling of about 90%.
- R3 (self-routing recovers half the gap): **failed completely.** After 100 steps the model had memorised every fact, so nothing was ever flagged and the arm was identical to the first. The detector was wrong, not necessarily the idea.

## 6. Conclusions of this cycle

1. **In this world, forgetting of regular knowledge is collateral damage from irregular facts stored at the same address.** Removing them from gradient training removed the forgetting and raised generalisation by ten points. [Observed, pre-registered intervention, three seeds, one toy world, one model family]
2. **The gain from the additive learner in `cab_s.py` is probably this effect.** A gradient learner with oracle routing reached 89.2% inference against the additive learner's 86.7% (different seeds, so not a direct comparison). The earlier reading "additive writes are better" should be weakened to "keeping exceptions out of the shared model is what matters".
3. **The hard part is now named precisely: deciding at write time what is irregular.** Checking after fitting is useless, because fitting hides it.
4. Not shown: anything about deep networks, language, or worlds whose rules change. Novelty of item 1 is unverified.

## 7. Next actions, in order

1. A write-time detector that works before fitting: solve the entity's address from its other facts, and ask whether this fact is predicted (leave-one-out surprise). Pre-register, run.
2. Referee and prior-art search on T4 and on conclusion 1.
3. A world whose rules change for some types part-way, to separate "exception" from "rule changed".
4. Order-sensitivity measurement (T1) for every learner we have.

---

## 8. Referee corrections (2026-11-21). These override the text above.

The referee read the note, the three scripts and all result files, and replayed the world's random stream. Literature was found by search; titles are leads, not verified readings.

**Conclusion 1 is not supported as written.**
- **The loss of old regular facts tracks corrections, not exceptions.** It is flat across exception rates (92.1%, 92.9%, 90.9% at 10%, 30%, 60%) while corrections were a constant 16 per phase. At 60% exceptions almost every entity carries an irregular fact, yet retention is still 91%. The referee's fit attributes about 0.31 loss to corrected entities and about 0.02 to exception-only ones (nine points; the first is effectively an intercept).
- **A simpler mechanism fits.** A correction to an old entity trains its address on that one fact for 200 Adam steps with none of the entity's other facts in the batch. With Adam the address can move far, breaking the entity's other answers directly. That is ordinary single-fact fine-tuning interference and possibly an optimiser artefact. Decoder drift is not needed, and nothing in the code measured address displacement or decoder drift. **T4's formula is unsupported.**
- **The intervention did not discriminate.** With irregular facts withheld, every entity is a clean entity that is never touched again, and clean entities were already at 99.9%. The oracle arm also removed four things at once.
- **The ten-point inference gain is a different effect.** The deficit exists immediately after each phase and survives full replay, so it is not forgetting. It is label-noise removal with an oracle: fitting an address to an arbitrary value corrupts type identification. Published as low-rank-plus-sparse and noise-robust training.
- **The "no irregular content, no forgetting" control was confounded**: it set exceptions and corrections to zero together.
- The per-seed figures in section 5.2 came from an inline script that was not saved; the routing "mixed" arm reproduces the pattern on other seeds.
- R3 was uninformative rather than failed: with nothing flagged, the arm is identical to the baseline by construction.

**Conclusion 2 does not follow.** The additive learner does put exceptions in its model (33–40% model-alone accuracy on them against 3% chance), is not order-free (addresses are solved against the current decoder, with sampling), and its model-alone retention (89.3%) is below the gradient learner's (93.5%). Its real difference is that it re-solves a corrected entity's address from all that entity's facts jointly, with ridge regularisation. There is also a latent bookkeeping bug if a new relation is added to an entity already in the statistics; not triggered in this world.

**Theory.**
- T1 holds in one direction only and needs exact commutation, acquisition at every reachable state, and no conflicting keys. Corrections are last-writer-wins and do not commute, so T1 does not apply to our own world. An append-only log never forgets and is order-dependent, so "the same as" is false.
- T2 is a heuristic use of the theorem (it bounds real dimension, needs regularity conditions, and concerns sufficiency for a parameter, not retention of facts). **It is already published as a continual-learning limit** (a 2024 paper using the same theorem; a 2020 result that optimal continual learning needs perfect memory).
- T3 is published in its general form (description length of models; continual learning as compression; prequential coding).

**Prior art for the main idea.** Keeping unpredictable content out of the slow shared model is published (2023 and 2024 consolidation models; 2022 sparse over-parameterisation for label noise; 2021 noisy-stream replay; example-forgetting studies). The correction mechanism and the store-plus-router remedy are published in model editing (2022 onward), including the open problem of deciding scope. The per-entity statement is at most a small increment.

**What survives.** In this bilinear model under Adam, old regular facts are lost almost only on entities that later received an irregular write, and withholding those writes removes the loss. Separately, keeping arbitrary values out of address fitting raises inference to the ceiling. Neither is new.

**Controls requested by the referee, running:** exceptions only (`--eps 0.1 --fix 0`) and corrections only (`--eps 0 --fix 16`). Results will be appended below.

### 8.1 Result of the two controls (2026-11-21)

Gradient learner, seeds 5001–5003, no code change. Raw files in `results/forgetting_split/`. [Observed]

| Condition | Regular facts, 20+ phases old | Irregular facts, 20+ phases old |
|---|---|---|
| Exceptions only (10%), no corrections | 99.7%, 99.6%, 99.6% | 59.1%, 59.4%, 56.1% |
| Corrections only (16 per phase), no exceptions | 90.9%, 91.0%, 91.5% | 66.4%, 68.1%, 63.8% |

**The referee was right and T4 as written is refuted.** Exceptions stored beside regular facts cause no loss of the regular facts. Corrections alone reproduce the whole loss. The cause is a late write to an old entity's address, trained on one fact without that entity's other facts, not exception-bent addresses meeting decoder drift.

What stands: (1) exceptions themselves are forgotten (to about 58%) while regular facts are not; (2) a correction written by gradient steps on one fact damages that entity's other facts; (3) both are standard, published effects.
Not yet separated: whether the damage in (2) is specific to Adam or to training without the entity's other facts. The referee's proposed arm (train the correction together with the entity's stored facts) would decide it.
