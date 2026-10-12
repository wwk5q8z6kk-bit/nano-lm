# Nano AI core

`nano_ai` is the stable transcript-to-structure boundary for Nano. It is
framework-independent and contains no app, service, audio, or deployment code.

```text
NanoInput(item_id, transcript)
  -> NanoSolver.infer(...)
  -> NanoOutput(exactly five ordered fields)
  -> transcript-valid evidence or explicit abstention state
```

The contract version is `nano.scribe.v0`. Its fields are chief complaint,
duration, severity, medication, and allergy. A field is `supported`, `absent`,
`missing`, `uncertain`, or `conflicting`. Positively evidenced absence is a
presentable result; missing, uncertain, and conflicting fields abstain.

The package provides:

- strict contract parsing and patient-span validation;
- a solver-neutral, failure-safe inference runner;
- separate strict content, normalized content, annotation-aligned grounding,
  failure, latency, and resource reporting;
- a closed-world deterministic reference and a grounded legacy-summary adapter;
- independent raw-proposal and verification-decision measurements for the
  legacy summary pipeline;
- lazy, hash-gated historical checkpoint adapters whose identity includes the
  complete artifact, architecture, runtime, prompt, decoding, grounding, and
  explicit device backend (CPU by default); and
- self-contained, integrity-locked conformance fixtures with optional
  repository-provenance verification.

Fabric remains a downstream regression harness. Historical R★ partitions are
recorded as training/validation/regression provenance and are not relabeled as
a fresh test. The disjoint `fresh-v0` partition is sealed at
[`benchmarks/data/fresh_v0/manifest.json`](benchmarks/data/fresh_v0/manifest.json),
and the matched result envelopes and comparison are under
[`benchmarks/data/fresh_v0/results/`](benchmarks/data/fresh_v0/results/).

That run measured 1100/1100 fields for the generator-aware deterministic
diagnostic, 804/1100 (73.09%) for Nano, and 898/1100 (81.64%) for scale. Nano had
25 output-format failures and dropped from 86.00% on seen values to 60.18% on
held values. Scale used 3.176x the parameters and 2.082x p50 / 2.811x p95
latency. Both trained models scored 0/20 on challenge-only missing targets. The
verifier reduced false-presented fields to zero, but native abstention remained
zero. This is a one-run synthetic, closed-world, generator-matched engineering
benchmark—not clinical evidence or an open-world generalization result.

Three bounded native-output interventions have now been evaluated and rejected
on the synthetic development family. H1's generative state/span grammar became the
frozen baseline for H2. H2 replaced autoregressive value generation with direct
state and pointer heads and evaluated two seeds across three epochs each. The
best H2 checkpoint reached 61.82% raw overall, 64.75% held-value, and 88.40%
missing-target accuracy with no decode failures, but reached only 0% absence,
24.80% conflict, and 21.60% uncertain accuracy and produced 1,707 raw
wrong-presented fields. It therefore failed the frozen raw-model quality gate.

After transcript verification, that checkpoint reached 65.74% overall with zero
false-presented fields. This is verifier safety, not evidence that the learned
proposal states were safe: the verifier did not supply model values and cannot
turn a failed raw-model gate into promotion eligibility. Latency was not
measured because quality failed, and `fresh-v1` remained sealed and unread.

H3 tested `nano_evidence_query_pointer_v1`: Nano v0.1's exact
3,148,608-parameter trunk plus a 137,861-parameter head with field/slot evidence
queries, one state classifier shared across fields, and full-context bilinear
boundaries. All six checkpoints were perfect on training-only calibration. The
frozen tie-break selected `seed-20260805-epoch-1`, which scored 56.42% overall,
72.55% held value, 85.20% missing, 45.52% absence, 37.60% conflict, and 38.80%
uncertain on known development, with no decode failures and 1,264
wrong-presented fields. It failed uncalibrated admission, so threshold,
verifier, latency, and `fresh-v1` stages were not run.

H4 kept the H3 model, objective, optimizer, compute envelope, evaluator, and
gates fixed while replacing the narrow generated training/calibration family
with a broader, leakage-audited version. The training-only winner scored
3,027/4,000 on calibration but only 2,248/5,000 (44.96%) on known development,
including 1,136/2,987 held values, 33/413 absence cases, and 2,141
wrong-presented fields. It regressed from H3 on every gated semantic category.
H4 therefore failed uncalibrated admission and stopped before threshold,
verifier, latency, and `fresh-v1`, as required. This rejects the exact H4
data-only repair; the next work is row-level mechanism localization, not model
scaling. An independent later test is still required for any generalization
claim.

The packaged smoke suite is an engineering conformance check, not evidence that
Nano is generally capable. Malformed model output, invalid diagnostics,
checkpoint mismatch, and solver-identity drift are inference failures; they are
never converted into valid-looking abstentions.

Run the conformance suite with:

```bash
python3 -m pytest -q nano_ai/tests
```
