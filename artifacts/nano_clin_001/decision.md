# NANO-CLIN-001 — decision record

**Decision:** `D-NANO-2026-08-25` · **Run date:** 2026-08-25
**Data:** synthetic non-PHI fixtures only · **Model calls:** 0 · **Paid compute:** none
**Recheck:** `.venv/bin/python scripts/run_nano_clin_001.py`

## What was built

The smallest complete vertical slice that exercises real contracts end to end:

```
SourceArtifact -> EvidenceSpanV2 -> ClinicalAssertion -> ClinicalEvent
-> EvidenceLedger -> PatientStateSnapshot -> encounter note -> VerificationReceipt
```

Contracts live in `nano/contracts.py` and **extend** `fabric/schemas.py`
conventions (frozen dataclasses, content-addressed `_cid` ids, validation in
`__post_init__`) rather than starting a second schema system, per §5/§14.
`fabric._cid` is imported, not reimplemented.

## Measured result

| fixture | baseline A provenance | candidate B provenance | conflicts | gaps |
|---|---|---|---|---|
| `fx_basic` | 0.00 | 1.00 | 0/0 | 0/0 |
| `fx_conflicting` | 0.00 | 1.00 | 1/1 | 0/0 |
| `fx_uncertain` | 0.00 | 1.00 | 0/0 | 2/2 |

Conflict recall 1/1, gap recall 2/2, zero false positives on either.

**The provenance number is near-tautological and must be read as such.**
Candidate B constructs claims *from* spans, so of course each cites one. What
the comparison actually demonstrates is narrower and worth stating plainly: the
baseline path has no mechanism by which a claim *could* cite evidence. That is
an architectural difference, not a quality difference, and it does not yet
establish that B produces better notes.

## The difference that is not tautological

On `fx_uncertain`, the same source line renders as:

- **Baseline A:** `You had no adverse reaction documented to penicillin.`
- **Candidate B:** `NOT FOUND IN RECORD (not equivalent to absent): You had no
  adverse reaction documented to penicillin.`

The baseline sentence reads, in a clinical note, as *no penicillin allergy*.
The source says only that none was **documented**. That is the
not-found-versus-absent failure with a safety consequence, and it is produced by
the naive path by construction. Same for the colonoscopy line.

Candidate B also preserves `[time approximate as stated]` rather than resolving
"maybe around 2021" to a date — enforced in the type, since `TemporalExtent`
raises if `APPROXIMATE` precision carries a full date.

## Defects found in this slice (not fixed here)

Recording these because the run surfaced them and an unrecorded defect is how
the next session re-derives it:

1. **Interrogatives are rendered as assertions.** `Do you recall why?` emits
   `Documented: Do you recall why?`. A question is not a clinical assertion; it
   should be segmented and dropped or typed separately. Inflates assertion count.
2. **`I do not remember` is classified as a denial.** The negation matcher fires
   on `do not`, but epistemic non-recall is not clinical negation. Mislabels
   epistemic status, and in a larger fixture would corrupt the negation metric.

Both are extractor defects, not contract or architecture defects — the ledger,
state projection and verification behave correctly on their outputs. Neither
changes the measured comparison above, because both affect A and B identically.

## Fixed conflict-detector defect

The first implementation flagged **any** two distinct years as a date
disagreement, which reported `fx_uncertain` (metoprolol started 2019, stopped
2021) as conflicting. Those are sequential events, not contradictory claims.
The detector now requires the years to attach to a **shared clinical concept**.
Pinned by `test_sequential_dates_are_not_reported_as_a_conflict`.

## Invariants verified (37 tests, `nano/test_nano_clin_001.py`)

Every content claim cites evidence · absence statuses may stand without evidence
· approximate time cannot carry an exact date · locator requires exactly one
modality family · evidence must be patient-scoped · original wording preserved ·
patient report never promoted to clinician confirmation · inference never
rendered as documented · conflicts surfaced and never silently resolved ·
not-found distinguished from absent · negation preserved · reprocessing
deterministic · state rebuildable from ledger · new evidence appends without
overwriting · no cross-patient contamination.

## Safety boundaries honoured

No PHI · no live data · no diagnosis · no treatment recommendation · no weight
updates · no paid compute · no repository rename · no tags moved · no branches
merged · no prior archived result relabelled as evidence for this architecture.

## What the evidence supports

- The contracts are implementable and their invariants are enforceable in types.
- An evidence-ledger path can carry provenance end to end where the direct path
  structurally cannot.
- Conflict and gap detection work on these fixtures at 1/1 and 2/2.

## What remains proposed

Everything else. Three fixtures is not a benchmark; no model is in either path;
no clinician has reviewed an output; entity resolution, episodes, trajectories,
multi-document reconciliation and incremental invalidation are unimplemented.
No promotion or kill rule is preregistered — per §11 that comes after a baseline
exists, and this is that baseline.

## The single next scientific question

> Does the evidence-ledger path still improve provenance coverage and
> unsupported-claim rate when a **generative model** is placed behind both paths,
> or is the advantage an artifact of the rule-based renderer used here?

That is the experiment that makes this architecture claim falsifiable, and it is
LCRB-1 on the benchmark ladder.
