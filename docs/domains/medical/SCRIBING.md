# P1 — Master Scribing (Medical)

## Target pipeline

```text
audio / transcript
→ immutable source
→ turns and speakers
→ clinical events
→ entities and values
→ state
→ temporality · uncertainty · contradictions
→ evidence spans
→ verified encounter record
→ note plan
→ coherent note
→ claim decomposition
→ independent verification
→ present | abstain | review
```

## Truth object

**Primary:** structured, evidence-grounded encounter representation.  
**Secondary:** free-form note (rendering of the record).

## Competence areas (exit gate dimensions)

```text
exact values · medications · allergies · complaints · duration · severity
negation · uncertainty · temporality · speaker · experiencer
contradictions · problem/assessment/plan attribution
section structure · coherence · redundancy · unsupported content
evidence provenance · review burden · critical-error severity
clinician edit effort · time to final note
```

## Mastery criteria

Do **not** claim P1 mastered until:

1. Satisfactory automatic metrics on held-out **external** medical dialogue evaluation (no PHI in repo)
2. **Blinded human evaluation** supports usefulness and safety
3. Owner signs P1 exit record

Synthetic/mock benchmark success alone is insufficient.

## Relation to historical scribe work

`scribe/` stages (v1/v2, G, A, C, S) and Paper α inform design — see [FAILURE_TO_ARCHITECTURE.md](../../FAILURE_TO_ARCHITECTURE.md).

E1 KILL applies to the **old closed generative-substrate claim** on the old task — not to abandoning P1 scribing as the product frontier.

## Immediate engineering focus

1. Encounter representation schema (entity/event/evidence refs)
2. Span/evidence bottleneck
3. Verified record → note realization under verification
