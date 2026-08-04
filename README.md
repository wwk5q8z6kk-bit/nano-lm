# nano-lm

**Nano is the AI itself:** a small, trainable, local-first, verification-gated
scribe intelligence. Given a supplied conversation transcript, it infers a
structured record, grounds factual fields in what was said, and abstains when
evidence is missing, conflicting, or uncertain.

`nano-lm` is the research and engineering repository for building and measuring
that intelligence. It is not an app, service, user interface, or commercial
scribe product. The name also does not require a language model in every path.
The operating rule is:

> Use the smallest sufficient solver, verify every consequential output, and
> escalate only when measured failures justify it.

## Research-to-training loop

The research built around Nano has one purpose: make the scribe AI better at
each step. Paper α, the scribe experiments, classical baselines, Fabric, and
Wedge are evidence sources and development instruments—not alternative project
identities.

```text
measure a Nano failure
    -> localize the cause
    -> choose a bounded change to data, training, architecture, or verification
    -> train or adapt Nano
    -> evaluate on frozen held-out tests
    -> integrate the improvement or retain the negative result
    -> use the evidence to choose the next step
```

Classical solvers establish the floor, expose residuals, and may provide
verification scaffolds. They do not replace the objective of strengthening
Nano's trained intelligence.

## AI boundary

The currently demonstrated scope uses supplied synthetic clinic-dialogue
transcripts and a five-field structured representation: chief complaint,
duration, severity, medication, and allergy. The active work is to make the AI
core coherent, reproducible, robust, and measurably better within that boundary.

```text
supplied transcript
    -> normalize the conversation
    -> infer candidate field values with the smallest sufficient solver
    -> bind every factual value to transcript evidence
    -> verify support, conflict, and uncertainty
    -> emit value + evidence + status, or abstain
```

Each learned or generative change earns integration when it improves matched
held-out utility over the strongest cheaper baseline without weakening
grounding or abstention. A failed change becomes evidence for the next training
decision, not a reason to abandon Nano.

Audio capture, speech transcription, UI, workflow orchestration, deployment,
distribution, billing, and clinical operations are not active project targets.
Research commands and evaluation harnesses may expose the AI for testing; they
are not a product roadmap. Human review may be used to evaluate outputs, but the
thing being built is the AI core.

## Project hierarchy

| Name | Role |
|---|---|
| **Nano** | The compact scribe AI core: inference, grounding, verification, and abstention |
| **NanoScribe** | A historical name used by scribe-task artifacts; not a separate app or architecture |
| [`nano_ai/`](nano_ai/README.md) | The versioned Nano v0 contract, solver boundary, adapters, fixtures, and evaluator |
| [`scribe/`](scribe/AUDIT.md) | The existing synthetic transcript-to-record AI prototype and evaluation trail |
| [`wedge_v1/`](wedge_v1/README.md) | Supporting document-evidence and validation infrastructure; not Nano's identity or inference target |
| [`fabric/`](fabric/README.md) | A closed-world verification regression harness; not the complete AI |
| [`papers/`](papers/README.md) and [`trajectory/`](trajectory/REPRODUCIBILITY.md) | The scientific record, preregistrations, results, and limitations |

## Current state

- Nano v0 now has a stable, solver-neutral AI-core surface in `nano_ai`: strict
  transcript input, five evidence-bound field outputs, explicit state and
  abstention semantics, structured inference failures, and separate evaluation
  measurements. Historical solvers enter through adapters rather than defining
  the contract.
- Paper α and the scribe experiments establish a reproducible synthetic
  instrument and its limitations; they do not establish clinical validity or
  open-world generalization.
- E1 found that a classical/rules method beat the tested generative reference on
  the frozen closed scribe task. That rejects the tested solver choice under that
  utility; it does **not** reject the goal of building the scribe AI.
- Fabric preserves propose→verify→abstain behavior as a regression harness.
- Wedge v1 implements evidence binding, exact document scoping, contradiction
  handling, abstention, and re-verifiable local state. Those are candidate
  supporting mechanisms, not a replacement direction for Nano.
- No standalone Wedge study is active. Resume Wedge work only when a measured
  Nano failure creates a transfer hypothesis for one of its mechanisms; the
  result must then pass a Nano acceptance test.
- The sealed `fresh-v0` matched benchmark is complete. The deterministic
  generator-aware diagnostic reached 1100/1100 fields. Nano reached 804/1100
  (73.09%) with 25 format failures and a seen-to-held drop from 86.00% to
  60.18%. The 10.00M scale model reached 898/1100 (81.64%), but used 3.176x the
  parameters and took 2.082x p50 and 2.811x p95 latency. Both trained models
  scored 0/20 on challenge-only missing-state targets. The verifier reduced
  false-presented fields to zero, while neither model produced a native
  abstention.
- `H1_NATIVE_STATE_COPY` was rejected on its sealed development family. It
  became the frozen comparison for H2 rather than a promoted Nano checkpoint.
- `H2_NATIVE_POINTER_SPAN` was also rejected. Its best checkpoint improved raw
  overall accuracy to 61.82%, held-value accuracy to 64.75%, and missing-target
  accuracy to 88.40%, with no decode failures. It simultaneously fell to 0%
  absence, 24.80% conflict, and 21.60% uncertain accuracy and emitted 1,707
  wrong-presented fields. The verifier raised final overall accuracy to 65.74%
  and reduced final false-presented fields to zero, but that safety came from
  verifier rejection rather than correct raw model behavior.
- `nano_evidence_query_pointer_v1` (H3) was rejected. Its training-only-selected
  checkpoint was perfect on calibration but scored 56.42% overall on known
  development: 72.55% held value, 85.20% missing, 45.52% absence, 37.60%
  conflict, and 38.80% uncertain, with 1,264 wrong-presented fields. The run
  stopped before threshold, verifier, latency, and `fresh-v1` as specified.
- H4's data-only transfer repair was rejected. Its training-only-selected
  checkpoint scored 44.96% overall on known development: 38.03% held value,
  53.60% missing, 7.99% absence, 22.80% conflict, and 33.60% uncertain, with
  zero decode failures and 2,141 wrong-presented fields. It regressed from H3
  on every gated semantic category and stopped before threshold, verifier,
  latency, and `fresh-v1`. H4 varied surface/value families but did not test
  evidence order, distractors, distance, or long context. Row-level comparison
  found complementary H3/H4 field shortcuts, which motivated one fixed-budget
  replay test rather than a larger model.
- H5's fixed 50:50 replay repair was rejected. It recovered 78.18% overall,
  74.32% held-value, and 100% missing-target accuracy, but reached only 67.80%
  absence, 59.60% conflict, and 64.80% uncertainty. The frozen stop skipped
  threshold, verifier, latency, and `fresh-v1`. The next bounded test changes
  one representation mechanism: soft state conditioning of evidence boundaries.

See [the strategic reset](papers/STRATEGIC_RESET.md) for the AI capability
contract and [the execution queue](papers/EXECUTION_QUEUE.md) for the next build
sequence.

## Evidence boundaries

| Result | What the repository supports | Boundary |
|---|---|---|
| Paper α | A reproducible held-out exact-copying failure on the tested scribe instruments, localized to open-vocabulary fields, with measured diversity, adaptation, and stack effects | Exact string match is not clinical faithfulness; parameter count was not isolated from training budget |
| E1 | Under frozen E1 utility, classical M1 scored about 0.999 versus about 0.925 for the official generative M0, producing a scoped **KILL** | Applies to the closed synthetic scribe task and tested solver; `rho` is review load, and the result does not cancel the scribe-AI objective |
| E4 | On locked R★ v1, the best tested classical system scored about 0.638 versus about -1.623 for the best tested generative-plus-verification system, producing a scoped **KILL** | Applies only to tested R★ v1; it is not a universal claim against generation |
| E3 | Normalize-then-match rescued 0/486 exact failures; one bounded agent-applied rubric pass labeled 0/100 sampled errors faithful | This was not independent human or clinician evaluation; inter-rater agreement remains unmeasured |
| Fabric | A propose→verify→abstain regression slice reached zero presented error under its closed synthetic verifier relation | Fabric is a scoped verification harness, not the complete Nano AI or an open-world reliability claim |
| Wedge v1 | The document runtime passes its checked synthetic and repository-dogfood paths; a bounded verifier fix reduced reviewed over-abstention from 6 to 3 in an agent-applied 10-task scoped pilot | These are supporting-component evaluations, not representative validation, Evidence Ledger claims, or proof of Nano capability |
| Nano `fresh-v0` | Under one sealed matched run, deterministic scored 1100/1100, Nano 804/1100, and scale 898/1100; verification prevented unsupported values from being presented | This is a synthetic, closed-world, generator-matched engineering benchmark with one run, not clinical evidence, open-world validation, or an independent generalization claim |
| Nano H2 pointer development run | Best raw checkpoint: 3091/5000 overall, 1934/2987 held values, 221/250 missing targets, 0/413 absence, 62/250 conflict, 54/250 uncertain, zero decode failures, and 1,707 wrong-presented fields; post-verification: 3287/5000 overall and zero false-presented fields | Rejected on sealed synthetic development data used for model selection; two seeds and six checkpoints only. Latency was not measured, `fresh-v1` remained sealed, and this rejects the exact H2 architecture/recipe rather than pointer mechanisms generally |
| Nano H3 evidence-query development run | Training-only-selected checkpoint: 2821/5000 overall, 2167/2987 held values, 213/250 missing, 188/413 absence, 94/250 conflict, 97/250 uncertain, zero decode failures, and 1,264 wrong-presented fields | Rejected on known synthetic development after perfect training-only calibration. Downstream threshold, verifier, latency, and `fresh-v1` stages were correctly skipped; this rejects the exact H3 architecture-plus-training-family intervention, not Nano generally |
| Nano H4 data-distribution development run | Training-only-selected checkpoint: 2248/5000 overall, 1136/2987 held values, 134/250 missing, 33/413 absence, 57/250 conflict, 84/250 uncertain, zero decode failures, and 2,141 wrong-presented fields | Rejected on known synthetic development after changing only the independently partitioned surface/value family. It regressed from H3 on every gated semantic category; downstream threshold, verifier, latency, and `fresh-v1` stages were correctly skipped. Evidence order, distractors, distance, and long context were not tested. This rejects the exact H4 data-only intervention, not broader data work or Nano generally |
| Nano H5 balanced-replay development run | Training-only-selected checkpoint: 3909/5000 overall, 2220/2987 held values, 250/250 missing, 280/413 absence, 149/250 conflict, 162/250 uncertain, zero decode failures, and 1,013 wrong-presented fields | Rejected on known synthetic development because absence, conflict, and uncertainty missed their frozen floors. Threshold, verifier, latency, and `fresh-v1` were correctly skipped. This rejects the exact 50:50 mixture, not replay generally, the evidence-query architecture generally, or Nano |

The canonical scientific record is
[the Evidence Ledger](papers/EVIDENCE_LEDGER.md). Claims must remain as narrow as
their instruments, data, utility, and verifier relation.

## Run the supporting Wedge validation runtime

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ask --corpus wedge_v1/data/corpus \
  "How long before cached entries expire?"
python3 -m wedge_v1 compare --corpus wedge_v1/data/corpus metformin
python3 -m wedge_v1 adversarial
```

These commands exercise Wedge; they do not run the Nano scribe AI. The
[Wedge runtime guide](wedge_v1/README.md) contains its private-corpus study
workflow and privacy limits.

## Compute provenance

The historical model work contains several checkpoint lineages, so compute
venue claims are run-specific. The original 3.15M from-scratch pretrain and its
post-training stages ran locally on Apple-silicon MPS
([pretrain audit](pretrain/AUDIT.md), [post-training audit](sft/AUDIT.md)). For
cloud GPU work, the documented chronology is **Kaggle first, then RunPod**: the
10M Stage S pretrain/scribe run and Stage T fine-tuning used Kaggle T4, while
later experiments used RunPod CUDA, including E1 on an RTX 3090 and the token
coverage run on an A6000 ([scale audit](scale/AUDIT.md),
[reproducibility manifest](trajectory/REPRODUCIBILITY.md),
[E1 preregistration/results](trajectory/PREREG_E1_nonlm_baseline.md),
[token-coverage preregistration/results](trajectory/PREREG_token_coverage.md)).

The rejected H2 decision run used one secure RunPod RTX 5090 pod with the
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` template. The result summary
is content-addressed at
`f61bb7f0f3401dfbaff8a5ab7e987d313a4811442f790b1d03f0883b403806cc`;
the best checkpoint at
`57755a445c7de1f2774b0667cbdfd689b4f9b73654934cc498ba4a72a522c8a2`;
and the row-level evaluation at
`641c08f826a5669220cd7fda8c52fbd2a682a352c8be36a97f93a099ecfe3833`.
No network volume was used. Before the pod was terminated, the complete source,
six checkpoints, logs, reports, and evaluation were transferred, verified, and
stored in an independent compressed backup with SHA-256
`88420f2644e25a2acdcaaffccb244bc3383a519b47622f74bb197034e37e46e6`.
Provider closeout recorded no remaining pod, so H2 is preserved locally and no
longer accruing RunPod compute or storage charges.

H3 used a separate secure RunPod RTX 5090 pod with the same template and no
network volume. The tracked summary is SHA-256
`cbd23c6a5799179b487119e9bee6e181dd328e691ee5388b12d03689f538ec82`;
the selected checkpoint is
`6477888bc8058a3dd747d292c3593385d8fb8148672161f5389f9685762ab477`;
and the row-level evaluation is
`df6896855980172aabd03149affffca8352c2ef9732fb860a79b3f50d854c831`.
All six checkpoints, both reports, logs, runtime identity, and evaluation were
verified and backed up before the pod was deleted. The result archive and its
independent backup share SHA-256
`2c2d400a5476090d96e39427e8e8e6c880f5a3535f18c94d0561d6c6d9fcda29`.
Provider closeout records no remaining pod or ongoing H3 compute/storage charge.

H4 and H5 used separate secure RunPod RTX 5090 workers with the same exact
container template. Both complete result archives and their internal manifests
were verified locally before the workers were deleted. H5 selected
`seed-20260805-epoch-3`; its development evaluation is SHA-256
`c67393962299470fc6b5026031b61617bbae85a2883105b8b8abfcdb30820c47`,
and its complete result archive is SHA-256
`bab5327e900597a083cb04631e645f2e0f500f14f01ec7db195e754b34620749`.
Provider closeout reports no remaining H5 pod or ongoing compute/storage charge.

That is venue chronology across multiple runs, not proof that one project-wide
checkpoint was sequentially tuned on Kaggle and then RunPod. Some later runs
reused frozen Kaggle-produced bases; others used distinct or regenerated bases.

## Repository map

| Path | Purpose |
|---|---|
| [`papers/STRATEGIC_RESET.md`](papers/STRATEGIC_RESET.md) | Nano AI definition and engineering strategy |
| [`papers/EXECUTION_QUEUE.md`](papers/EXECUTION_QUEUE.md) | Current AI-core build queue |
| [`nano_ai/`](nano_ai/README.md) | Versioned AI contract, inference boundary, adapters, conformance fixtures, and evaluator |
| [`scribe/`](scribe/AUDIT.md) | Synthetic scribe AI prototype and evaluations |
| [`wedge_v1/`](wedge_v1/README.md) | Supporting evidence-runtime implementation and validation studies |
| [`fabric/`](fabric/README.md) | Scoped verification regression harness |
| [`papers/WEDGE_V1.md`](papers/WEDGE_V1.md) | Wedge component-validation contract |
| [`papers/EVIDENCE_LEDGER.md`](papers/EVIDENCE_LEDGER.md) | Canonical claim-to-evidence record |
| [`papers/DECISION_GATES.md`](papers/DECISION_GATES.md) | Experiment and AI-change promotion criteria |
| [`trajectory/`](trajectory/REPRODUCIBILITY.md) | Preregistrations, instruments, results, and reproduction notes |
| [`pretrain/`](pretrain/AUDIT.md), [`sft/`](sft/AUDIT.md), [`scale/`](scale/AUDIT.md) | Historical model-building and evaluation trail |
| [`benchmarks/`](benchmarks/README.md) | Digest-bound benchmark integration sentinel |

## Verification

```bash
pip install -r requirements.txt
python3 -m pytest -q nano_ai/tests
pytest -q
```

Full training dependencies and archival reproduction details are documented in
[`trajectory/REPRODUCIBILITY.md`](trajectory/REPRODUCIBILITY.md). Large
checkpoints and tokenized shards are not stored in Git.
