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
