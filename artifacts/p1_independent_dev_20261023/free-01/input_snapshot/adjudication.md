# Pre-inference adjudication

Both context-isolated agent annotators agreed on all 120 states (100%; Cohen's
kappa 1.0). Complete evidence sets agreed on 115/120 fields (95.83%). This is
agreement between AI agents following one rubric, not human dual-IAA or
independence of model training. The coordinator has implementation context.
Neither annotator nor the coordinator has seen model outputs for these cases.

The five evidence disagreements concern fresh-05 MED, fresh-11 MED, fresh-12
MED, fresh-16 ALG, and fresh-24 MED. Use annotation A for those fields: preserve
additional uncertainty/correction observations and use the shortest complete
affirmed clause in fresh-24. Exact alternatives and reasons are recorded in
adjudication.json; both original annotation files remain unchanged.

One further evidence adjudication affects unanimous fresh-09 MED: retain only
the two incompatible positive/negative reports as gold C evidence. The later
uncertainty confirms nonresolution but is not a contradictory side. Otherwise
a metric accepting any two of three spans could credit a positive-plus-uncertain
pair without the denial. The original three-span annotations remain available.

No states changed, no cases were excluded, and no annotation was altered to
fit the checker. All 24 cases remain in every denominator. In particular,
natural denials remain A outside the product contract's fixed vocabulary.

Interpretive limits established before inference: fresh-23 ALG is U under the
rubric's qualified-knowledge rule; fresh-24 MED and ALG are S based on clear
local affirmations to clinician questions, with patient-only quotes. Those
quotes cannot literally contain the drug names supplied by the clinician.
Medication S values retain the temporal qualifiers selected by both raters.
Consequently strict normalized-value/exact-span mismatch is not automatically
a false medical assertion. State, content, evidence and final-note emission
must be reported separately. The set contains only one unresolved C field and
four M fields, so conflict and missing-state estimates are especially limited.
