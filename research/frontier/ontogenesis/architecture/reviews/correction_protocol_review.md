# Independent review: correction measurement protocol

**Review date:** 2026-11-22. **Reviewer:** independent trainable-reference agent.

**Conclusion:** no material defect found in the reviewed measurement contracts. The implementation is suitable as a bounded measurement foundation; this review does not establish model quality, an architecture advantage, or successful integration with a real continual-learning run.

## Reviewed sources and executed evidence

| Artifact | SHA-256 at review |
|---|---|
| `experiments/correction_protocol.py` | `f7339939059c8337a2257c9064c3073522b63fb5744cb6a464e7423c15b86087` |
| `experiments/test_correction_protocol.py` | `0722befa7924833e2918584ac6d54c46c6054f1c9a52d8cf1e34547eede34932` |

The reviewer read both complete files and independently executed the supplied suite: **26 tests passed**, with no failures or errors. No implementation or author-test files were changed during review.

The additional executable `checks/review_correction_oracle.py` preserves an independently written event-scan oracle. It uses NumPy's generator with seed **192734**, 160 synthetic histories for each of the two censor policies, and five historical cutoffs per history. Every history contains two observed keys on one subject, multiple assignments in later phases, repeated keys within a phase, and a three-value domain. Random missing immediate observations, omitted phase snapshots, and explicit abstentions exercise distinct missing-data paths. The final phase always has a snapshot so earlier historical cutoffs can be checked.

The oracle compares all fields in three event cohorts, one immediate-acquisition summary, and four phase-end horizons (0, 1, 2, 5). **320 histories × 5 cutoffs × 3 cohorts × 5 summaries = 24,000 matching rows.** The oracle locates superseding events with a forward scan of event identity and value; it does not reuse the implementation's successor or summary helpers. It uses the public manifest and log APIs, so this is independent checking of cohort calculations rather than a separate implementation of every input validator. The output includes the NumPy version and SHA-256 hashes of the implementation, author tests, and oracle for reproducibility.

Run from the package root:

```bash
python -B experiments/test_correction_protocol.py
python -B checks/review_correction_oracle.py
```

Both commands print their results and do not write package artifacts by default. A different NumPy version may produce a different deterministic history stream; the executable receipt records that version.

## Contract assessment

| Contract | Finding | Evidence and practical meaning |
|---|---|---|
| World/replay separation | Consistent | The manifest is generated independently of replay consumption. Named replay streams do not mutate the frozen world. This repairs the previously identified shared-generator confound at the interface level. |
| Causal learner view | Consistent within its stated interface | Delivered views contain only delivered events and currently observed target revisions. The full coordinator manifest is deliberately excluded from that view. |
| Same-key revisions | Consistent | Revision identity is preserved. Under `new_revision`, a subsequent same-key assignment censors the older event even if its value is unchanged. Under `value_change`, same-value revisions remain eligible until the first different value. |
| Same-subject writes | Consistent | Immediate acquisition is measured after an individual assignment. Horizon zero is the end of that phase, after later writes may have caused interference. These endpoints are not conflated. |
| Historical summaries | Consistent | Superseding events beyond the historical cutoff do not alter current eligibility. A known supersession before the requested horizon is reported as censoring even if that horizon is not yet due. |
| Primary denominator | Consistent | Failed initial acquisition remains in the primary eligible denominator. Missing initial acquisition is recorded separately and prevents an unsupported acquired-only rate. |
| Missing phase observations | Consistent | Missing eligible endpoint samples make primary accuracy undefined. `None` predictions are explicit measured abstentions, counted as incorrect rather than missing. |
| Solver outcomes | Consistent within the specified status convention | Status 0 is only a candidate; status 2 is reported infeasibility; other failure statuses remain unknown failures. A solver return code alone does not validate a deployed edit. |
| Actual deployed displacement | Consistent conditional on supplied observations | A supplied actual vector is checked against the original float64 bounds, preserving thresholds through casting and addition. The rounding regression distinguishes the requested increment from `after - before`. |
| Linear and nonlinear outcomes | Appropriately separated | Linear residual checks and actual-logit margin/argmax checks are distinct. Satisfying a linear candidate audit does not assert a successful nonlinear model edit. |
| Provenance and output preservation | Consistent on reviewed public construction paths | Constraint rows carry target-source provenance; receipt hashes bind their supplied values. Output paths use exclusive creation, and default execution is read-only. |

## Limits retained by this review

The instrument cannot verify that a caller actually queried a learner at the declared time, supplied an honest post-update vector, or withheld the coordinator's future manifest. Those are execution and integration responsibilities. The module documents its learner view as an interface discipline, not a Python security boundary. The numerical audit checks the supplied values with float64 residual arithmetic and explicit tolerances; it is not a formal certificate for arbitrary runtime arithmetic or nonlinear model behavior.

Event observations, particularly repeated same-key or same-value revisions, are correlated. The reported counts are event-level summaries, not independent statistical samples. The acquired-only endpoint remains secondary, and same-value versus semantic-change strata should remain visible in downstream reports.

This review includes no paid compute, GPU work, external dataset, language benchmark, or assessment of a learned model's retention. Integration with the native model, any training trajectory, and any future bridge have separate review scope and cannot inherit this conclusion automatically.
