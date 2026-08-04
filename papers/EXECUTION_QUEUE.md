# Execution queue

The queue advances Nano as an AI core. Work is ordered by the shortest path to a
coherent, measurable intelligence: freeze the task, unify inference and
evaluation, benchmark the current Nano checkpoints, then use measured failures
to drive successive training cycles. Application and deployment work are
outside this queue.

| Priority | Work | Exit condition | Status |
|---:|---|---|---|
| 1 | Restore the AI boundary | Canonical docs and active labels agree that Nano is the scribe AI itself, Wedge is supporting infrastructure, and Fabric is a regression harness | **Complete** |
| 2 | Freeze the Nano v0 AI contract | Versioned fixtures and schemas define transcript input, five output fields, evidence spans, field states, abstention, partition roles, and metrics; app, audio, and deployment concerns are excluded | **Complete** |
| 3 | Unify inference and evaluation | One solver-neutral research surface runs the Nano contract and emits deterministic, machine-checkable outputs for baseline and candidate solvers | **Complete** |
| 4 | Seal benchmark inputs and benchmark current Nano | A disjoint, previously unmeasured test partition is frozen; exact artifact hashes identify every checkpoint; current trained checkpoints and the strongest deterministic reference are compared with field accuracy, evidence validity, coverage, abstention, robustness, latency, and resource cost reported separately | **Complete** |
| 5 | Select the next training target | A repeated, high-value Nano failure is localized and mapped to one bounded hypothesis about data, curriculum, objective, adaptation, architecture, inference, grounding, or verification | **Complete** |
| 6 | Run the first evidence-led Nano improvement cycles | H1 and H2 are trained and evaluated under frozen recipes; each is integrated or rejected without weakening its gate, and its result is retained | **Complete — H1 and H2 rejected** |
| 7 | Run the H3 mechanism repair | The evidence-query intervention targets H2's measured mechanisms and yields a quality-gate decision without reading `fresh-v1` prematurely | **Complete — H3 rejected** |
| 8 | Run the H4 data-distribution repair | A versioned, leakage-audited training family tests whether H3's calibration-to-development collapse is caused by narrow synthetic coverage while model, objective, compute, and gates remain fixed | **Complete — H4 rejected** |
| 9 | Freeze and run H5 fixed-budget replay | One preregistered H3/H4 mixture changes fit composition only; architecture, objective, total worlds, optimizer steps, seeds, evaluator, quality floors, and the existing H4 training-only calibration partition remain fixed before one development decision | **Complete — H5 rejected** |
| 10 | Implement and freeze H6 state-conditioned boundaries | One zero-initialized query-residual change lets Nano's detached soft semantic-state posterior select start/end boundary offsets while all H5 data, compute, loss, selection, evaluator, and gates remain fixed | **Next** |
| 11 | Integrate, refreeze, and repeat | A winning change enters Nano's common inference surface, passes acceptance and Fabric regressions, becomes the new frozen baseline, and leaves a ranked residual list for the next cycle | **Pending an eligible candidate** |
| 12 | Test external validity periodically | Preregistered, privacy-safe evaluations test stabilized Nano baselines on more representative transcripts with independent labels and explicit domain limits | **Deferred until the core loop is stable** |

## Nano v0 intelligence boundary

```text
supplied conversation transcript
  -> five-field structured representation
  -> evidence span and support state for every asserted value
  -> explicit missing / uncertain / conflicting states
  -> abstention when support is insufficient
```

The initial fields are chief complaint, duration, severity, medication, and
allergy. This is a compact capability boundary, not an application workflow.
Audio capture, speech recognition, UI, service orchestration, deployment, and
clinical operations do not enter this queue.

The current trained Nano checkpoints and strongest deterministic extractor were
measured under the same contract. The deterministic system is a performance
floor, diagnostic, and possible verifier—not Nano's replacement. Existing
scribe and Fabric fixtures remain reusable where their semantics match the
frozen contract. Wedge evidence-envelope, privacy, provenance, and verification
mechanisms should be adapted only when they produce a testable improvement to
Nano; the document Q&A surface is not the target intelligence.

Priorities 2 and 3 are implemented in `nano_ai`. The v0 manifest preserves the
R★ train/dev/eval artifacts as historical training, validation, and regression
provenance; it explicitly does not relabel the already measured eval split as a
fresh test. Solver failures remain separate from legitimate field abstentions,
and fixture truth is never passed through `NanoInput`.

Priorities 4 and 5 are complete. On the sealed `fresh-v0` matched run, the
generator-aware deterministic diagnostic scored 1100/1100 fields. Nano scored
804/1100 (73.09%), with 25 output-format failures and 86.00% seen versus 60.18%
held-value accuracy. Scale scored 898/1100 (81.64%), but required 3.176x the
parameters and 2.082x p50 / 2.811x p95 latency. Both trained models scored 0/20
on challenge-only missing targets. Verification reduced false-presented fields
to zero, but native abstention was zero. These are synthetic, closed-world,
generator-matched engineering measurements, not clinical or open-world claims.

## Completed Priority 6: H1 and H2

`H1_NATIVE_STATE_COPY` trained the generative native state/span grammar and was
rejected on the sealed development family. Its frozen report (SHA-256
`04fecee1339719d08d6ab8f2f5228a9d64dbe05ef02ca0baffa888a99a70571d`)
became H2's comparison rather than a promoted Nano checkpoint: 50.04% overall,
28.26% held-value, 0% missing-target, 93.70% absence, 95.20% conflict, and
92.00% uncertain accuracy, with a 5.30% inference-failure rate.

`H2_NATIVE_POINTER_SPAN` preserved Nano's causal trunk and replaced generative
field output with direct state and pointer supervision. Two seeds were trained
for three epochs each on a secure RunPod RTX 5090. The best checkpoint,
`seed-20260806-epoch-2` (SHA-256
`57755a445c7de1f2774b0667cbdfd689b4f9b73654934cc498ba4a72a522c8a2`),
produced these development results:

| Measurement | Raw model | After verifier | H1 comparison |
|---|---:|---:|---:|
| Overall | 61.82% | 65.74% | 50.04% |
| Held value | 64.75% | 64.75% | 28.26% |
| Missing target | 88.40% | 88.40% | 0% |
| Absence | 0% | 0% | 93.70% |
| Conflict target | 24.80% | 24.80% | 95.20% |
| Uncertain target | 21.60% | 100% | 92.00% |
| Inference failure | 0% | 0% | 5.30% |
| Wrong/false presented | 1,707 fields | 0 fields | 0 fields after verification |

The post-verification uncertain score and zero false-presented count reflect
verifier rejection/fallback, not correct raw state prediction. H2 passed raw
overall, held-value, missing-target, and failure thresholds but failed raw
absence, conflict, uncertain, and zero-wrong-presented thresholds. Because
promotion required both raw and final gates, H2 was rejected.

The tracked result summary is SHA-256
`f61bb7f0f3401dfbaff8a5ab7e987d313a4811442f790b1d03f0883b403806cc`;
the content-addressed row-level evaluation remains local at SHA-256
`641c08f826a5669220cd7fda8c52fbd2a682a352c8be36a97f93a099ecfe3833`.
All source, checkpoints, reports, logs, and evaluations were verified locally
and in an independent backup before the RunPod pod was terminated. The closeout
records no remaining pod. No network volume remains attached and no provider
compute or storage charge continues.

H2 stopped at its quality gate. Latency was not measured, and `fresh-v1`
remained sealed and unread.

## Completed Priority 7: H3 mechanism repair

H3 implemented the frozen `nano_evidence_query_pointer_v1` intervention: the
exact 3,148,608-parameter Nano v0.1 trunk plus a 137,861-parameter shared-state,
full-context evidence-query head. Mechanics, runtime identity, two seeds, three
epochs, training-only selection, and the uncalibrated quality stop were all
enforced as specified.

Every checkpoint scored 4,000/4,000 on its training-only calibration partition.
The frozen tie-break selected `seed-20260805-epoch-1`. Its known-development
result was 2,821/5,000 overall, 2,167/2,987 held values, 213/250 missing,
188/413 absence, 94/250 conflict, and 97/250 uncertain, with zero decode
failures and 1,264 wrong-presented fields. Only held-value retention and the
failure-count gate passed.

The uncalibrated admission failure stopped the run before the confidence
threshold, verifier, latency, or `fresh-v1`. H3 is rejected. It improved state
balance and held-value transfer relative to H2 but did not achieve safe
development quality. The tracked summary is SHA-256
`cbd23c6a5799179b487119e9bee6e181dd328e691ee5388b12d03689f538ec82`;
the content-addressed row-level evaluation is SHA-256
`df6896855980172aabd03149affffca8352c2ef9732fb860a79b3f50d854c831`.
All six checkpoints and both reports were verified locally and backed up before
the RunPod pod was deleted.

## Completed Priority 8: H4 data-distribution repair

The decision-relevant H3 signal was the split collapse: perfect training-only
calibration did not transfer to the already-inspected development family. H4
therefore changed data, not model size, while keeping H3's
architecture, tokenizer, objective, optimizer, seeds, three-epoch budget,
runtime class, evaluator, selection ordering, verifier policy, and quality
floors fixed.

The generator audit explains why H3's calibration score was misleading. Its
fit and calibration worlds had different IDs, but every calibration dialogue
template and every field-specific supported/conflict value family was already
represented in fit. H4 therefore keeps the exact exposure budget -- 2,800 fit
worlds and 200 calibration worlds, yielding 11,200 and 800 records, batch 32,
three epochs, and 1,050 optimizer steps per seed -- while making calibration a
real transfer split. Entire dialogue-template and open-value families must be
held out from fit, not merely recombined under new world IDs.

H4 expanded lexical, dialogue-template, open-value, and state-realization
coverage while enforcing fit/calibration world, value-family, and
template-family separation. It retained normal, missing, uncertain, and
conflicting variants. It did not vary the five-field dialogue structure,
evidence order or distance, add distractor turns, or create a long-context
intervention; those dimensions remain untested. Development was used only for
the final quality decision and supplied no copied phrases, values, thresholds,
or hyperparameters.

H4 retains H3's staged stop: uncalibrated semantic and retention admission
comes first; only an admitted candidate may apply a training-owned confidence
threshold and then the verifier. Both calibrated raw and verifier-final results
must reach 3,041/5,000 overall, 1,905/2,987 held values, 219/250 missing,
383/413 absence, 236/250 conflict, and 228/250 uncertain, with at most 10/1,000
failures and zero wrong- or false-presented fields. Latency follows quality;
`fresh-v1` remains sealed until both pass.

H4 ran the preregistered data-only intervention on a secure RunPod RTX 5090.
Both fixed seeds completed three epochs, all six checkpoints and both training
reports were authenticated locally, and the training-only selection chose
`seed-20260806-epoch-2` (SHA-256
`6408524c43b6ada8249aeb83e440b6aa0f64512006219663be4105f6d586e13f`).
Its calibration result was 3,027/4,000 overall with macro joint accuracy
0.8132. The known-development result was:

| Measurement | H4 raw | Frozen gate | H3 raw |
|---|---:|---:|---:|
| Overall | 2,248/5,000 (44.96%) | at least 3,041 | 56.42% |
| Held value | 1,136/2,987 (38.03%) | at least 1,905 | 72.55% |
| Missing target | 134/250 (53.60%) | at least 219 | 85.20% |
| Absence | 33/413 (7.99%) | at least 383 | 45.52% |
| Conflict target | 57/250 (22.80%) | at least 236 | 37.60% |
| Uncertain target | 84/250 (33.60%) | at least 228 | 38.80% |
| Inference failure | 0/1,000 | at most 10 | 0/1,000 |
| Wrong presented | 2,141 fields | reported at admission | 1,264 fields |

Only the inference-failure gate passed. H4 therefore failed uncalibrated
admission and correctly stopped before applying its training-owned threshold,
running verifier-final quality, measuring latency, or opening `fresh-v1`. The
independently partitioned H4 surface/value family did not repair H3's transfer
failure and degraded every gated
semantic category on this known synthetic development set. This rejects the
exact H4 data-only intervention; it does not establish that more representative
data is useless or that model scale is the missing variable.

The tracked result summary is SHA-256
`6dee4cb1999ebab75ce7a19b2f8c07ff0b36be3af72606987c300b837e31f473`;
the content-addressed development evaluation is SHA-256
`b8f8b350f4ad772b06a292c5e156d097e549f62e15be09c3c55c608574c0ed82`.
The complete result archive and every member in its checksum manifest were
verified locally before the RunPod worker was deleted. The provider now reports
no pods, and `fresh-v1` remains sealed and unread.

The error profile localizes the next decision. Duration and severity retained
high joint accuracy (91.30% and 88.50%), while medication and allergy fell to
15.70% and 3.30%. State accuracy could also remain high when exact evidence was
wrong: medication state accuracy was 83.20%, and all 413 absent fields had the
right state despite only 33 exact joint results. These observations make field
conditioning, state/evidence coupling, and exact-span boundary supervision
bounded hypotheses for diagnosis, not proven causes. No further training run is
authorized until a row-level H3/H4 error audit distinguishes those mechanisms
and preregisters one smallest sufficient intervention.

## Completed Priority 9: H5 balanced replay

H5 changed only fit composition to a deterministic 50:50 H3/H4 replay mixture.
The training-only selector chose `seed-20260805-epoch-3` (SHA-256
`04ba7b4d0dc876ca3d8de7fe7d809ca16796e5bf55249ad93ba1dd3557c394fe`).
One frozen known-development evaluation produced:

| Measurement | H5 raw | Frozen gate | Result |
|---|---:|---:|---|
| Overall | 3,909/5,000 (78.18%) | at least 3,041 | Pass |
| Held value | 2,220/2,987 (74.32%) | at least 2,167 | Pass |
| Missing target | 250/250 (100%) | at least 219 | Pass |
| Absence | 280/413 (67.80%) | at least 383 | **Fail** |
| Conflict target | 149/250 (59.60%) | at least 236 | **Fail** |
| Uncertain target | 162/250 (64.80%) | at least 228 | **Fail** |
| Inference failure | 0/1,000 | at most 10 | Pass |
| Largest per-field modal state | 922/1,000 | at most 949 | Pass |

H5 is rejected under its frozen admission rule. Threshold, verifier, latency,
and `fresh-v1` were therefore not assessed. Replay restored H3 retention and
substantially improved H4, but it did not meet the semantic-state floors; this
rejects the exact mixture, not replay generally or Nano. The checksum-bound
result is summarized in `artifacts/nano_h5/RESULT_SUMMARY.json`, and the RunPod
worker was deleted after all result members were verified locally.

## Supporting evidence available to Nano

Wedge has scoped component evidence for fail-closed claims, exact document
scoping, freshness-gated recall, and isolated review artifacts. In a frozen
10-task agent-applied repository pilot, a bounded deterministic change reduced
reviewed over-abstention from six to three while preserving four correct
abstentions.

That pilot had no manual baseline and was not representative private-corpus,
independent human, clinician, time-saved, or scientific evidence. No
standalone Wedge milestone is active. The representative-document study
stays parked unless a measured Nano failure creates a mechanism-transfer
hypothesis with a defined Nano acceptance gate. Existing Wedge evidence remains
available to inform those interventions; it does not gate core work or redefine
Nano.

## Operating rules

- The smallest sufficient solver wins; extraction tasks always include a
  classical baseline.
- Consequential values require transcript evidence or abstention.
- AI capability measurements, supporting-component evaluations, and scientific
  claims remain separate.
- An experiment must name the Nano training or engineering decision it can
  change before it runs.
- An LM must beat the strongest cheaper matched baseline by
  $\Delta U > 0.05$ without weakening grounding or abstention controls.
- A result that remains only in an experiment, Wedge, or Fabric has informed
  Nano but has not enhanced Nano until it is integrated and revalidated.
- Negative results narrow a solver hypothesis; they do not erase the scribe-AI
  objective.

## Next: Priority 10 — H6 state-conditioned boundaries

H5 produced 727 state-correct/span-wrong fields and 270
span-correct/state-wrong fields. That aggregate mismatch is descriptive, not
causal. It motivates a small diagnostic of state/evidence coupling but does not
promise that the intervention will improve Nano.

The initial hidden-fusion concept was rejected before freeze because it mixed a
normalization change with a new pointer-to-state-head gradient path. That would
have confounded the result and prevented exact H5 functional identity at
initialization.

The final H6 intervention changes only the existing boundary queries. It adds a
learned offset tensor shaped `[5 semantic states, 2 boundaries, 64 query
dimensions]`, exactly 640 parameters. For each field, the detached soft state
posterior selects separate start and end offsets, which enter as residuals on
the existing projected boundary queries. The offsets are zero-initialized, so
under the same seed H6 at step zero is functionally identical to a freshly
initialized H5; it does not resume a trained H5 checkpoint. Gold state is not
teacher-forced. H6 has 3,287,109 total parameters, including a 138,501-parameter
head.

H6 must reuse byte-exact H5 fit and calibration data, the 50:50 mixture, frozen
anchor and tokenizer, prompt, loss formulas, optimizer, seeds, three epochs,
1,050 steps per seed, training-only selection, evaluator, verifier, and every
H5 gate. It may not add data, change replay ratios, reweight losses, scale the
trunk, or relax gates. The known-development stage stops at its first failed raw
gate; only a full raw pass can admit threshold, verifier, latency, and sealed
confirmation stages.
