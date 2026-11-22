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
