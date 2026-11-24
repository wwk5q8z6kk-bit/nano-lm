# G04 third study: a three-way router (draft protocol, for review before any code is run)

Joint study of the G04 session (this branch) and the memory-line session (`research/memory-saturation-factorial`, PR #199). Status: **draft**. Nothing below has been run. Predictions are written after a development-seed check and committed before any registered run.

## Question

Can one label-free router send each surprise where it belongs?
- An **exception** goes to private storage.
- **Noise** is held back.
- A **rule change** goes into the shared rule.

Can it do this in a world that contains all three? Each part works alone: repetition separates noise from exceptions (`RULE-CHANGE-20261122.md`), a group-agreement detector finds most rule changes (`RULE-CHANGE-DETECT-20261122.md`), and confident disagreement routes exceptions privately (the memory session's `memory_rule_routing.py`, R1 and R2 held on five held-out seeds). Together they can conflict: the private router sees a change's first instances as surprises and diverts them before the change detector can use them.

## World

`rule_change.py`'s world, unchanged: hidden types, rules, rule changes at phases 10, 20 and 30 on four (type, relation) pairs, exceptions seen at creation and again one phase later, noisy labels on 8 old facts per phase seen once.

## Learner

The memory session's two-table memory, copied and cited from `memory_rule_sharing.py` / `memory_rule_routing.py`:
- a shared table read through typed addresses;
- a private table read through spread addresses;
- reads summed; 200 joint steps per phase at 1e-3;
- a private-only pass of 2,000 steps.

Typed addresses give the memory the types for free (the memory session's own caveat). Here, a rule change written into the shared table reaches every entity of that type at once. The router's task is to let change evidence reach the shared table and keep exceptions out of it.

## Arms

| Arm | Exceptions | Noise | Rule changes |
|---|---|---|---|
| `none` | trained jointly | trained jointly | trained jointly |
| `private` | confident disagreement (τ = 0.1) → private pass | → private pass (cannot tell) | → private pass (holds the old rule) |
| `change_only` | trained jointly | trained jointly | detected and trained into the shared table |
| `threeway` | at creation: confident disagreement (τ = 0.1) → private pass; old-entity observation seen twice consistently → private pass | old-entity observation seen once → held | **change detector runs first**; facts in a detected change group are exempt from private routing and trained into the shared table |
| `threeway_t02` | as `threeway` with τ = 0.2 | as `threeway` | as `threeway` |
| `threeway_rev200` | as `threeway` | as `threeway` | shared revision with 200 steps instead of 2,000 |
| `threeway_random` | as `threeway`, but the new-entity facts sent private are an equal number drawn at random from the non-change facts | as `threeway` | as `threeway` |
| `threeway_privfirst` | as `threeway` | as `threeway` | the same detectors in the reverse order: private routing first, change detector on what is left |
| `oracle` | true label → private pass | true label → discarded | true label → trained into the shared table |

`threeway_privfirst` was added at the ontogenesis session's suggestion. Running the change detector first designs the interference away, so without the reversed order the study would never measure it. The registered comparison is "change-first beats private-first on old-entity revision by at least X"; X is set after the development check.

Order inside `threeway`, following the memory session's pitfall 2:
1. Run the change detector on the phase's new facts.
2. Exempt facts in flagged groups.
3. Route the remaining facts.

Cold start, its pitfall 1: confident-disagreement gating starts only once the memory's top probability exceeds τ. Observations about old entities are judged by repetition, never by the model, so they have no cold start.

Change detector, adapted to this memory: the signature is the memory's own answers on the other relations, and the threshold is as registered in `rule_change_detect.py` (2 matching new entities, 5 of 7 agreeing). Revision writes the new value into the shared table for the group's older members (shared table only, 200 steps), with their other answers held as targets.

## Measures

As `rule_change.py`: acquisition, stable inference, revision of new entities (1–5 and 6–10 phases after a change), revision of old entities, exceptions kept, noise resisted. Also: facts routed private, false change alarms, missed changes.

The reverse failure is reported separately, again at the ontogenesis session's suggestion: genuine exceptions whose entity falls inside a falsely flagged change group. The exemption could send them into the shared revision instead of private storage. Measure: exceptions in falsely flagged groups, and the fraction of them kept.

## Seeds

Development seed 1; registered seeds 7301–7305 (new; five, so both sessions use one "4 of 5" rule).

## Review (memory-line session, 2026-11-22), adopted

1. τ = 0.1 primary, 0.2 as a sensitivity row (`threeway_t02`). On its held-out seeds, 0.1 gave 0.81 of corrections at 0.035 inference cost, and 0.2 gave 0.50 at 0.008.
2. Old-entity observations: seen twice only, with no gate. New-entity facts outside a flagged change group pass the τ gate at creation, so an exception never enters the shared table. A shared write is costly and slow to undo.
3. Shared revision 2,000 steps, with 200 as a sensitivity row (`threeway_rev200`). True detections should cost nothing, since overwriting the type's value is right for a real change. False alarms at 2,000 steps write a wrong rule into a whole type, so the stable-rule accuracy on pairs hit by false alarms is reported separately from all other pairs.
4. Five registered seeds.
5. A count-matched random control for the private routing (`threeway_random`).
6. Limit, to be stated next to the result: typed addresses give this memory the types for free, so a shared revision reaches the whole type at once. That makes change propagation easier here than for a learner that has to discover types (for example the bilinear learner of the first two studies).

## Questions put to review (answered above)

1. Should `private` use τ = 0.1 or the cautious 0.2? Proposed: 0.1 as primary, 0.2 as a sensitivity row.
2. Is "seen twice" the right noise/exception rule for this memory, or should the private path also require the confidence gate?
3. Should the shared-table revision for old group members use the 2,000-step budget the memory session found necessary for corrections? It is a write against accumulated neighbour mass, which is the budget limit it found. Proposed: yes.

## Development check (seed 1 only) and the settings it fixed

1. **As first written, the change detector collapsed.** Its signature is the memory's own answers. Early on, this memory has barely learned the rule, so in 12 phases it flagged 638 groups, 637 of them false. Each triggered a 2,000-step shared write, which drove stable inference to 18%, at 33 minutes per 12 phases. With no routing, the memory itself learned fine (99.7% acquisition).
2. **Fixes:**
   - warm-up of 3 phases;
   - revision of only the changed relation, for at most 64 older members (typed addresses mean members of a type share slots);
   - a confidence gate.
3. **The gate's form came from the memory session's measured confidence scale** for this memory: median top probability 0.034 at phase 1 and 0.17 at phase 39, with right and wrong answers overlapping. On its advice, only the older members' answer on the changed relation is gated; the signature is not.

   | Setting (seed 1, 40 phases) | Old entities revised | Exceptions kept | Noise resisted | Stable rule | Flags / false | Missed |
   |---|---|---|---|---|---|---|
   | none (reference) | 21% | 31% | 93% | 83% | — | — |
   | oracle (reference) | 99.9% | 99.7% | 91% | 68% | — | — |
   | signature gated, 0.2 | 25% | 97% | 93% | 78% | 2 / 0 | 11 |
   | older members only, 0.08 | 72% | 73% | 80% | 69% | 146 / 121 | 1 |
   | **older members only, 0.12 (chosen)** | **76%** | **84%** | **91%** | **71%** | **72 / 50** | **1** |

   Chosen by revision, with noise held within 0.03 of none. The setting is tuned on seed 1 and declared as such.

## Predictions (registered seeds 7301–7305; informed by seed 1, so a replication)

A prediction holds when its bound holds on the mean over the five seeds. The protocol's 4-of-5-seeds rule is reported beside each.

- Z1 **Routing revises rule changes:** threeway revise_old at least 0.30 above none.
- Z2 **Private routing alone holds old rules:** threeway revise_old at least 0.30 above private (τ = 0.1).
- Z3 **Exceptions are kept:** threeway exception_kept at least 0.30 above none.
- Z4 **Order matters:** threeway (change detector first) beats threeway_privfirst on revise_old by at least 0.15.
- Z5 **The exemption costs exceptions:** among exceptions inside falsely flagged groups, the fraction kept is at least 0.20 below threeway's overall exception_kept.
- Z6 **False alarms cost the rule where they land:** in threeway, stable-rule accuracy on (type, relation) pairs hit by a false alarm is at least 0.10 below its accuracy on other stable pairs.
- Z7 **Which facts go private matters for exceptions:** threeway_random exception_kept at least 0.20 below threeway.
- Z8 **The revision budget matters:** threeway_rev200 revise_old at least 0.10 below threeway.

**Reading.** If Z1–Z4 hold, one label-free router handles all three kinds of surprise in this world, and the order of its detectors is what makes change revision possible. If Z5 and Z6 hold as well, the router's remaining cost is its false alarms, in exceptions and in the rule where they land. Calibrating that trade-off, not the router's structure, is then the open problem.

**Limit, stated with the result:** typed addresses give this memory the types for free (see the review section).
