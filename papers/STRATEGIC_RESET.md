# Strategic reset

**Status:** Canonical AI research and engineering strategy

**Active build target:** Nano, a compact local scribe AI

## Mission

Nano is the trainable intelligence that turns a supplied conversation transcript
into a grounded structured representation. It should infer what was said, bind
factual values to evidence, expose uncertainty and conflict, and abstain rather
than invent.

`nano-lm` builds and evaluates that AI. The research around Nano exists to guide
its successive training and engineering improvements. It does not build an
end-user app, service, user interface, commercial workflow, or general-agent
platform.

The operating thesis is:

> Use the smallest sufficient solver, verify every consequential output, and
> escalate only when necessary.

“Small” includes memory, latency, energy, compute, privacy exposure, dependency
surface, and maintainability—not only parameter count. Nano may use symbolic,
statistical, neural, or hybrid methods. Architecture is earned by measured
capability, not assumed from the project name.

## AI capability contract

Nano's first coherent intelligence boundary is:

1. accept a supplied text conversation transcript;
2. infer a structured record with the initial five fields: chief complaint,
   duration, severity, medication, and allergy;
3. bind each factual value to one or more transcript spans;
4. distinguish supported, positively evidenced absence, missing, conflicting,
   and uncertain states;
5. present supported values or explicit absence and abstain when the evidence
   does not justify either;
6. run locally within a compact, reproducible compute envelope; and
7. be evaluated on held-out accuracy, grounding, coverage, abstention,
   robustness, latency, and resource cost.

The current evidence comes from synthetic clinic dialogue. It is an instrument
for developing the AI, not evidence of clinical validity or open-world
generalization.

Audio capture, speech transcription, UI, workflow automation, deployment,
distribution, billing, and clinical operations are outside the active target.
A research API or command may exist to execute and evaluate Nano, but shipping
an application is not a project milestone. Human judgments may evaluate the AI;
a human-review product workflow is not what this repository is building.

## Research-to-training loop

Research is subordinate to improving Nano. Every experiment should close this
loop:

1. measure a specific Nano capability failure;
2. localize it to data, representation, training, architecture, inference,
   grounding, verification, or abstention;
3. define the smallest change that tests the explanation;
4. train, adapt, or modify Nano under a frozen recipe;
5. compare against the current Nano checkpoint and strongest relevant baseline
   on held-out data;
6. integrate the change only when the evidence supports it; and
7. carry both positive and negative results into the next improvement step.

Here, “training” includes pretraining, fine-tuning, adaptation, curriculum and
data changes, and objective changes. Deterministic or hybrid engineering is also
a valid intervention when the evidence favors it, but every intervention is
judged by whether it measurably improves Nano.

Paper α localizes a held-out copying failure. E1 and E4 set comparison floors
and reject specific tested solver choices. Fabric tests verification behavior.
Wedge probes evidence mechanisms. None is a separate destination: each exists
to teach us how to build a better Nano.

## Component hierarchy

| Component | Role |
|---|---|
| **Nano** | The scribe AI core: inference, grounding, verification, uncertainty, and abstention |
| **NanoScribe** | Historical label used by scribe-task artifacts; not a separate application |
| [`scribe/`](../scribe/AUDIT.md) | Synthetic transcript-to-record AI prototype and evaluation trail |
| [`wedge_v1/`](../wedge_v1/README.md) | Supporting evidence, retrieval, scoping, and validation infrastructure; not Nano itself |
| [`fabric/`](../fabric/README.md) | Closed-world propose→verify→abstain regression harness; not a complete AI architecture |
| Paper α and `trajectory/` | Scientific evidence and negative-result record |

Wedge remains a useful test bed for evidence binding, contradiction handling,
abstention, private evaluation, and re-verifiable local state. Reuse mechanisms
only when they improve Nano under the AI capability contract. The Wedge
document workflow neither defines Nano nor sets its roadmap.

## Scientific constraints

| Evidence | Strategic consequence |
|---|---|
| **Paper α** measured held-out exact-copying failures in tested small-LM scribe systems | Preserve the instrument and field-localized result; do not claim clinical validity or universal model failure |
| **E1 KILL**: classical M1 beat the official generative M0 under frozen E1 utility on the closed scribe task | Keep the scribe-AI objective; reject the tested generative solver as the default for that task and require classical baselines |
| **E4 KILL**: classical beat generative-plus-verification under frozen utility on tested R★ v1 | A redesigned regime is a new hypothesis, not a reinterpretation of the failed one |
| **E3** was an agent-applied rubric audit | Never describe it as independent human or clinician validation |
| **Fabric** reached zero presented error under one closed synthetic verifier relation | Preserve it as a regression harness; do not promote it into the complete Nano architecture or an open-world guarantee |
| **Wedge v1** has scoped engineering and agent-applied study results | Treat them as supporting-component evidence, not representative Nano capability validation or a new AI identity |
| **Nano `fresh-v0`**: deterministic 1100/1100; Nano 804/1100 (73.09%); scale 898/1100 (81.64%) | Treat deterministic as a generator-aware diagnostic ceiling. Scaling cost 3.176x parameters, 2.082x p50, and 2.811x p95 while both trained models missed all 20 challenge-only missing targets. This is one synthetic closed-world matched run, not open-world or clinical evidence |
| **Nano H1**: the native generative state/span intervention was rejected on sealed development | Retain H1 as the frozen comparison for subsequent native-output work; do not promote its checkpoint or reinterpret its failure as a rejection of Nano |
| **Nano H2**: the best direct state/pointer checkpoint improved raw overall, held-value, and missing-target accuracy but failed absence, conflict, uncertain, and raw wrong-presented gates | Reject the exact H2 architecture and recipe. Preserve the extraction gain as a mechanism clue, keep raw model and verifier behavior separate, and require the next repair to address transferable state semantics and full-context evidence selection |
| **Nano H3**: the shared-state, full-context evidence-query head achieved perfect training-only calibration but only 56.42% overall on known development, with 72.55% held-value, 85.20% missing, 45.52% absence, 37.60% conflict, and 38.80% uncertain accuracy | Reject the exact H3 architecture-plus-training-family intervention. The large calibration-to-development collapse makes data-distribution transfer the next bounded target; do not scale the model or GPU without contrary evidence |
| **Nano H4**: independently surfaced training data regressed every gated semantic category relative to H3 | Reject the exact H4 data-only intervention. Preserve its complementary field behavior as diagnostic evidence, not proof that data or architecture is the cause |
| **Nano H5**: fixed 50:50 H3/H4 replay reached 78.18% overall, 74.32% held-value, and 100% missing-target accuracy but failed absence (67.80%), conflict (59.60%), and uncertainty (64.80%) floors | Reject the exact replay mixture. Preserve the retention recovery and state/span mismatch as clues; test one representation mechanism without changing data, scale, or gates |

Negative results narrow solver and architecture choices. They do not erase the
AI objective. The [Evidence Ledger](EVIDENCE_LEDGER.md) remains authoritative
for scientific status.

Historical E4 artifacts use “product scope” and “product track” for the frozen
Wedge/R★ evaluation. That provenance wording remains unchanged and refers only
to that tested component track. It does not define Nano. The E4 KILL stops the
tested generative hypothesis under that regime, not the scribe-AI program.

## Current state

| Area | State |
|---|---|
| AI identity | Restored: Nano is the scribe intelligence; Wedge and Fabric are supporting infrastructure |
| AI implementation | Nano v0 now exposes one strict, solver-neutral contract, inference runner, fixture layer, and evaluator in `nano_ai`; historical solvers are isolated behind adapters |
| Solver evidence | Classical-first is required by E1 on the closed extraction task; generation has not earned a default role |
| Verification | Fabric provides scoped regression behavior; Wedge provides mechanisms that must be adapted and tested rather than assumed compatible |
| Generalization evidence | Controlled synthetic results exist; representative open-world and independent human evaluation remain open |
| Training state | The sealed `fresh-v0` matched run reproduced a held-value gap, exact-output format failures, and a missing-state failure. H1 through H5 were then executed under bounded development contracts and rejected. H5 restored overall and held-value retention but still failed absence, conflict, and uncertainty, with 997 fields showing exactly one of state or span correct |
| Next build | Implement and preregister H6 as a bounded diagnostic: add only a zero-initialized `[5 semantic states, 2 boundaries, 64 query dimensions]` offset tensor whose start/end query residuals are selected by the detached soft state posterior; keep H5 data, objective, optimizer, compute, selection, evaluator, verifier, and gates fixed |

`fresh-v0` is a consumed historical benchmark and may be used only as a
post-selection regression. H1 established the development comparison at 50.04%
overall, 28.26% held-value, 0% missing-target, 93.70% absence, 95.20% conflict,
and 92.00% uncertain accuracy, with a 5.30% inference-failure rate and zero
false-presented fields after verification.

H2 trained a 3,157,273-parameter direct state/pointer model at two seeds and
three epochs per seed. Its best checkpoint (`seed-20260806-epoch-2`, SHA-256
`57755a445c7de1f2774b0667cbdfd689b4f9b73654934cc498ba4a72a522c8a2`)
reached 61.82% raw overall, 64.75% raw held-value, and 88.40% raw missing-target
accuracy with no decode failures. It simultaneously reached 0% raw absence,
24.80% raw conflict, and 21.60% raw uncertain accuracy and emitted 1,707 raw
wrong-presented fields. The verifier raised final overall accuracy to 65.74%
and reduced final false-presented fields to zero, but absence remained 0% and
conflict remained 24.80%. Because the frozen gate required both raw and final
quality, H2 was rejected.

This is a bounded synthetic development result used for model selection, not an
independent test or generalization result. Training loss approached zero while
development loss worsened after the first epoch in both seeds, and the best raw
model defaulted heavily to `supported` under the development lexical/template
shift. The evidence implicates state-transfer and evidence-selection behavior;
it does not establish that pointer mechanisms generally fail or that a larger
GPU/model would repair the mechanism.

H3 tested `nano_evidence_query_pointer_v1`: Nano v0.1's exact
3,148,608-parameter causal trunk plus a 137,861-parameter evidence-query head,
for 3,286,469 parameters total. It used five learned field identities, two
full-sequence evidence slots per field, one shared state classifier, and
full-context bilinear boundaries. Two seeds and three epochs per seed were run
under the frozen H2 data, objective, and optimizer contract.

Every checkpoint scored 4,000/4,000 on training-only calibration, so the frozen
tie-break selected `seed-20260805-epoch-1` (SHA-256
`6477888bc8058a3dd747d292c3593385d8fb8148672161f5389f9685762ab477`).
On known development it scored 2,821/5,000 overall, 2,167/2,987 held values,
213/250 missing, 188/413 absence, 94/250 conflict, and 97/250 uncertain, with
zero decode failures and 1,264 wrong-presented fields. It passed only the
held-value and failure-count admission gates. The protocol therefore stopped
before threshold application, verifier evaluation, latency, and `fresh-v1`.

Relative to H2 raw behavior, H3 raised held-value accuracy by 7.80 percentage
points, absence by 45.52 points, conflict by 12.80 points, and uncertainty by
17.20 points while reducing wrong-presented fields by 443. Overall accuracy
fell 5.40 points and the safety floors remained far away. H3 is a useful bounded
negative result: shared full-context queries improved balance but did not make
the narrow generated family transfer.

H4 tested a separately versioned surface-transfer data family while preserving
H3's model, tokenizer, loss, optimizer, seeds, runtime class, evaluator,
acceptance floors, 2,800-world fit budget, and 1,050-step-per-seed budget. Its
11,200 fit and 800 calibration records broadened lexical, dialogue-template,
open-value, and state-realization families and held complete template and
open-value families out of fit. Exact H4 did not change the five-field dialogue
structure, evidence order or distance, distractor turns, or long-context
regime; those dimensions remain untested.

Training-only selection chose `seed-20260806-epoch-2` (SHA-256
`6408524c43b6ada8249aeb83e440b6aa0f64512006219663be4105f6d586e13f`).
On known development it scored 44.96% overall, 38.03% held-value, 53.60%
missing, 7.99% absence, 22.80% conflict, and 33.60% uncertain accuracy, with
zero decode failures and 2,141 wrong-presented fields. It failed every semantic
gate and stopped before threshold, verifier, latency, or `fresh-v1`.

The row-level H3/H4 comparison was complementary rather than uniformly worse:
H4 raised severity joint exact from 17.20% to 88.50% but lowered allergy from
68.60% to 3.30% and medication from 62.80% to 15.70%. H4 predicted every
allergy state as `absent`; medication preserved 83.20% state accuracy while
only 19.40% of evidence spans were exact. These are post-hoc clues, not causal
proof. H5 therefore tested exact 50:50 replay from the frozen H3 and H4 fit
pools while preserving architecture, objective, exposure, and gates.
Training-only selection chose `seed-20260805-epoch-3`. On known development H5
scored 78.18% overall, 74.32% held value, 100% missing, 67.80% absence, 59.60%
conflict, and 64.80% uncertain, with zero decode failures and 1,013
wrong-presented fields. It passed overall, retention, missing, failure, and
state-balance gates but failed absence, conflict, and uncertainty. The frozen
stop correctly skipped threshold, verifier, latency, and `fresh-v1`.

H5 is a bounded negative result, not a reason to abandon Nano or silently tune
the replay ratio. Its aggregate errors include 727 state-correct/span-wrong and
270 span-correct/state-wrong fields. Those counts motivate a bounded diagnostic;
they neither establish a causal state/evidence mechanism nor promise that H6
will improve it. The initial hidden-fusion concept was rejected before freeze
because it mixed a normalization change with a new pointer-to-state-head
gradient path, confounding interpretation and preventing exact H5 functional
identity at initialization.

The final H6 intervention changes only the existing boundary queries. It adds a
learned offset tensor shaped `[5 semantic states, 2 boundaries, 64 query
dimensions]`, exactly 640 parameters. The detached soft state posterior selects
separate start and end offsets, which enter as residuals on the existing
projected boundary queries. The offsets are zero-initialized, so under the same
seed H6 at step zero is functionally identical to a freshly initialized H5; it
does not resume a trained H5 checkpoint, and gold state is not teacher-forced.
H6 has 3,287,109 total parameters, including a 138,501-parameter head. H5's
data, compute, losses, selection, evaluator, verifier, and gates remain
unchanged.

## AI development principles

- Optimize the intelligence, not an application shell.
- Use research to choose each successive Nano training and engineering step.
- Start with the strongest honest classical baseline.
- Require transcript evidence for each asserted factual value.
- Distinguish missing information, inference failure, contradiction, and
  uncertainty.
- Prefer abstention to unsupported fluency.
- Keep sensitive evaluation inputs local and out of tracked artifacts.
- Reuse Wedge and Fabric mechanisms only when Nano acceptance tests prove they
  help.
- Preserve preregistrations, results, limitations, and negative findings.
- Tie every training, data, model, or architecture change to a repeated measured
  AI failure and a frozen before/after comparison.

## Decision rule

Start work when it targets a defined Nano capability, has a measurable result,
uses the smallest adequate baseline, defines grounding and abstention behavior,
and can be verified without weakening scientific provenance.

Stop or defer work when it mainly expands apps, interfaces, deployment,
research-document features, general-agent scope, benchmark surface area, or
model complexity without a demonstrated Nano intelligence need.

## Canonical documents

- [Strategic reset](STRATEGIC_RESET.md): Nano's AI capability contract and strategy.
- [Execution Queue](EXECUTION_QUEUE.md): current AI-core work.
- [Wedge v1](WEDGE_V1.md): supporting component-validation contract.
- [Evidence Ledger](EVIDENCE_LEDGER.md): supported scientific claims and limits.
- [Decision Gates](DECISION_GATES.md): admission and promotion criteria.
- [Empirical Foundation](EMPIRICAL_FOUNDATION.md): Paper α evidence lock.
