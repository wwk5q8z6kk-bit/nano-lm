# Decisions taken 2026-08-05

Recorded by the co-founder acting on delegated authority ("take all relevant
steps and decisions based on logic"). Each states the reasoning and what would
reverse it.

## 1. LICENSE — Apache-2.0, with papers/ prose under CC-BY-4.0

**Decision.** Apache-2.0 for code; CC-BY-4.0 for `papers/`; NOTICE records
third-party data obligations and the limits of the evidence.

**Why.** The repository has been public since 2026-07-16 with **no license**,
which grants no reuse rights to anyone — including for the paper artifacts it
exists to support. Apache-2.0 over MIT because it carries an explicit patent
grant, and the owner approved *commercial* use of fineweb-edu (ODC-By) earlier
today; a permissive license without a patent clause would have been the weaker
pairing for that posture.

**Reverses if:** the project decides to pursue a proprietary path, in which
case relicensing own work is possible but already-distributed copies are not
recallable.

## 2. Repository stays PUBLIC

**Decision.** No change to visibility.

**Why.** 581 files are already public and have been for weeks. Nothing
proprietary is being protected — the program's headline finding is that a
*classical* baseline beat the generative substrate (E1 KILL, 0.999 vs 0.925),
so there is no moat to close. The differentiator is the evidence discipline
itself, which only has value if it can be inspected. Closing a door already
open would cost credibility and buy nothing.

**Reverses if:** future work introduces genuinely sensitive material — real
clinical data (prohibited by standing rule), or an unpublished result whose
priority matters.

## 3. `SUBMISSION_PACKET.md` identity — no scrub

**Decision.** Leave the author name and email in place.

**Why.** This was listed as an open decision; checking made it a non-decision.
`papers/latex/paper1.tex` on `origin/master` has carried
`\author{Hassan El Jesr\\ \texttt{summer.say3y@icloud.com}}` since 2026-07-31.
The identity is already public, was published deliberately, and is *authorship
metadata on the author's own paper* — the exact thing the file exists to carry.
Scrubbing one copy while the canonical copy stays public would be theater.

**What was scrubbed instead, and correctly:** the RunPod account balance, the
separate provider account email, SSH key paths and names, public key blobs, and
pod IPs — none of which were previously public and none of which belong in a
public repository. Residual scan over tracked files: zero.

## 4. Pushed 73 commits to origin

**Decision.** Push `codex/p5-measurement-integrity` and `frontier/active-v1`,
plus all local-only tags.

**Why.** This was the largest standing risk in the entire project: the whole of
`nano_ai/`, `wedge_v1/`, and the complete H2–H6 evidence chain existed on one
laptop with no remote copy. Preconditions were verified first — secret scan over
all 73 commit patches clean, no blob above 50 MB (largest 11.9 MB), PII scrubbed,
LICENSE added. Holding work hostage to a laptop is not a safety posture.

## 5. Next experiment — H7, not D-b

**Decision.** The next model experiment targets **absent/uncertain state
classification** (`papers/PREREG_H7_STATE_HEAD.md`), not slot-diversity ×
the winning corner.

**Why.** H5 and H6 both died on epistemic-state gates while **held-value copying
passed** (H6: 2,277/2,987). The per-state decomposition shows why, and shows the
two candidates are not comparable:

- `absent` — state 0.482, span 0.927. Joint tracks state exactly; the model
  finds the evidence and picks the wrong state. 376 of 413 were *presented*.
- `uncertain` — state 0.760, span 0.972. Same shape.
- `conflicting` — state 0.800, span 0.572. The mirror image, and a different
  problem (two distinct spans required). Explicitly out of H7's scope.

D-b addresses the residual copying gap in the allergy slot — real, and the
better *paper* result, since slot diversity moved it +66.7 points. But that gate
already passes. Fixing something that is not blocking, while the thing that has
now stopped two consecutive hypotheses goes unaddressed, is the wrong order.

**Reverses if:** H7's precondition check fails — specifically, if the training
target turns out not to teach denial evidence for ABSENT at all, the fix may be
data-side and belongs in a different design.

## 6. Rung-1 pretraining stays deferred

**Decision.** Do not spend the approved $150 yet.

**Why.** The program's own evidence says the remaining copying gap is a
*finetune-data* problem — one slot, caused by a five-value training pool at
`scribe/build_scribe_data_v2.py:30` — not a scale problem. Buying scale to fix a
data-composition defect would be the expensive way to learn that. H7 is free.
