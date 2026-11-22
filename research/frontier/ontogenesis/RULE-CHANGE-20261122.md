# G04 first study: a world where rules change (2026-11-22)

`rule_change.py`. Deviations (made on development seed 1) and predictions X1–X6 committed in `e4dd5fd` before any registered run; scorer committed in `780b934` before any registered result. Seeds 7101–7103. Raw files: `results/rule_change/`; scored output: `results/rule_change/verdict.json`. [Observed]

**Question.** Earlier worlds here are stationary: every surprise is an exception or a correction, so sending surprises to private storage is right by construction. Here rules really change: at phases 10, 20 and 30, four (type, relation) pairs get a new value. The stream also carries entity exceptions (seen at creation and again one phase later) and noisy labels (seen once). How far does a label-free policy get, and how far is it from a learner that knows each surprise's kind?

**Answer.** Seeing a surprise twice is enough to tell noise from exceptions, without labels. It does nothing for rule changes. The rule is then revised no better than by the plain learner: 52% of old entities and 74% of new entities answer the new value, against 97% and 100% for the ceiling. Knowing the kind of surprise is not enough on its own either. When the revision is trained into each entity's address rather than the shared rule, it revises new entities worse than the plain learner does.

| Arm | Noise resisted | Exceptions kept | New entities after a change, 1–5 / 6–10 phases | Old entities revised | Stable rule (unchanged pairs) | Override store |
|---|---|---|---|---|---|---|
| Plain: trains everything | **10%** | 77% | 69% / 72% | 48% | 91% | 0 |
| Seen twice (label-free) | **96%** | **100%** | 74% / 78% | **52%** | 94% | 386 |
| Oracle, revision into addresses | 99% | 100% | 57% / 60% | 62% | 99% | 399 |
| Oracle, revision into the shared rule (ceiling) | 99% | 100% | **100% / 99%** | **97%** | 99% | 399 |

Means of three seeds; per-seed values are in the result files. Every arm learned new facts at 100%.

## Against the registered predictions

All six held. They were informed by the development seed, so they test replication, not discovery.
- X1, plain: noise resisted 10% (≤ 30%), exceptions 77% (≤ 85%), revision after a change 69% (between 50 and 85%).
- X2, seen twice: noise 96% and exceptions 100% (≥ 95%); revision 74%, within 0.10 of plain; old entities 52% (≤ 70%).
- X3, the ceiling: at least 96% on all four measures.
- X4, oracle with revision into addresses: old entities 62% (≤ 70%).
- X5, cost of having no change detector: 45 points on old entities (≥ 30).
- X6, stable rule kept: 91–99% in every arm (≥ 90%).

## What this shows

- **Noise and exceptions can be separated without labels, by repetition.** A consistent second observation is cheap, exact evidence that a fact is an exception. A single disagreeing observation is held rather than trained.
- **Rule changes are the open part, and they are invisible to that policy.** A change arrives as new entities showing a new value. To a policy that only judges old entities, these look like ordinary new facts. The shared rule then drifts slowly toward the new value, and old entities keep the old one: 52% of them are revised after up to 30 phases.
- **Where the revision is written matters as much as knowing that it is one.** Even the labelled oracle revises new entities worse than the plain learner (57% against 69%) when it retrains old entities' addresses. Retraining with the old addresses frozen, so the change must go into the shared decoder, revises everything (97–100%). This is the same lesson as G06b from the other direction: shared structure carries the rule, and a write lands in shared or private structure depending on what the update is allowed to move.

## Limits

Bilinear model, one synthetic world, three seeds. The ceiling needs the hidden types (to find every entity of a changed type), which no learner here has. Its form was chosen on the development seed, so it is an upper reference, not a candidate. Noise and exceptions were made distinguishable by design (one observation against two). A world where noise repeats, or exceptions are seen once, would need more than counting. Not a language-model result.

## Next discriminating test

A label-free change detector. Watch the shared rule's own errors on new entities, per relation and value cluster. When several new entities in a phase agree on a value the shared rule rejects, treat it as a candidate change. Then retrain the decoder with addresses frozen, using the model's own answers for the affected cluster as targets, plus the new value. The measure is how close it gets to the ceiling on revision, against how often it fires on exceptions or noise. The memory-line session's private-slot memory can be added as an arm once its detector is confirmed.
