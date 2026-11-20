# Independent results and report review

A separate review agent audited both persisted model runs after inference,
then compared REPORT.md with the primary comparisons, encounters, provenance,
annotation records, posthoc diagnostics, and integrity records. This is an
AI-agent review, not human or clinician validation.

The review confirmed primary and subgroup arithmetic, annotation agreement,
transport-error counts, timings, recorded hashes, and posthoc figures. It
confirmed that the constrained checked score is entirely uncertainty prevalence,
that all sources trigger global context uncertainty, and that three removed
assertions have correct normalized content but differing evidence boundaries.
The posthoc label analysis reproduced byte for byte without model calls.

Two report findings were corrected before completion:

- All 120 checked fields are uncertain, but only 98 changed state; 22 were
  already uncertain. The report and current-state narrative now state this.
- The finishing active-state, documentation, and diff checks were not initially
  indexed in verification.json. Their successful results are now recorded.

The report also uses “hash-verified preserved” rather than “immutable” for
ordinary filesystem snapshots. The reviewer identified no other material
reporting findings. No primary output, locked annotation, or inference code was
changed during the results review.
