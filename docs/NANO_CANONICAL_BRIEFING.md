# NANO — CANONICAL NEW-AGENT BRIEFING

**VERSION:** 2026-08-23

**STATUS:** Canonical operational briefing. Supersedes accumulated ad-hoc prompts.
Single source of truth for a fresh agent joining the Nano / NanoScribe program.

> Read it completely before changing code, launching compute, modifying datasets,
> or drawing conclusions.
>
> Do not begin by writing another strategy document.
>
> Your job is to understand the current system, verify actual repository and
> RunPod state, then continue the highest-information experiments.

---

## I. WHAT NANO IS

Nano is NOT merely: a medical scribe; a held-out-copying benchmark; a tiny
language model; a Qwen derivative; a verifier wrapper; a clinical template
engine; an agent framework; a coding model.

The long-term mission is to build the smallest useful, reliable intelligence
system capable of:

1. constructing a faithful representation of messy evidence;
2. compressing it without corruption;
3. maintaining that representation across time;
4. reasoning over it;
5. recognizing uncertainty;
6. planning;
7. using tools and taking actions;
8. adapting while preserving provenance and reliability.

The governing principle:

> BEFORE NANO EARNS THE RIGHT TO REASON OVER THE WORLD,
> IT MUST BECOME EXCEPTIONALLY GOOD AT REPRESENTING THE WORLD FAITHFULLY.

Medicine is the first demanding DomainPack. Medicine is not intended to
permanently define Nano Core.

## II. CAPABILITY LADDER

```text
P1 — SCRIBING            "What happened?"
P2 — SUMMARIZATION       "What matters?"
P3 — CHARTING            "What is the state, and how has it changed?"
P4 — SYNTHESIS
P5 — QUESTIONING / UNCERTAINTY
P6 — REASONING
P7 — PLANNING
P8 — TOOLS / ACTION
P9 — ADAPTATION
```

Macro phases: FOUNDATION I (P1) · FOUNDATION II (P2–P3) · INTELLIGENCE
EXPANSION (P4–P9).

P1 remains the current integration frontier. P2/P3 research probes are
authorized now because they can reveal whether the P1 representation creates a
future architectural dead end. P7/P8 agentic research is also authorized as a
separate research lane.

**Do NOT let later-stage agent behavior corrupt P1 truth construction.**

## III. PRIMARY P1 ARCHITECTURAL THESIS

Generated prose is NOT primary truth. The primary truth object is the
**EncounterRecord**.

```text
SOURCE
→ TURNS
→ MODEL CANDIDATES
→ EVIDENCE REQUEST / QUOTE / SPAN
→ CONSTRAINED SELECTOR
→ EXACT EvidenceSpan
→ ClinicalAtom / value
→ assertion / temporality / experiencer / certainty
→ EncounterRecord
→ verifier where semantic interpretation is required
→ later summaries / notes / state / reasoning
```

Models propose. Software constrains. Verifiers judge semantics.
**No model directly creates trusted truth.**

## IV. CORE REPRESENTATION

Conceptual types: `Source`, `Turn`, `EvidenceSpan`, `ClinicalAtom`,
`TemporalState`, `AssertionState`, `Certainty`, `Experiencer`, `Conflict`,
`UnresolvedItem`, `EncounterRecord`.

Invariants:

- source offsets must be valid
- `source[start:end] == evidence.text`
- evidence belongs to one source/turn
- speaker attribution must be consistent
- IDs/references must close correctly
- patient evidence may be valid; clinician evidence may be valid
- OTHER / UNKNOWN may exist but cannot silently become trusted truth
- ASSERTED / UNCERTAIN evidence must be represented explicitly
- UNCERTAIN must preserve uncertainty
- silence does NOT imply a negative fact; silence may create `UnresolvedItem`
- raw evidence remains immutable
- normalization must be named and deterministic
- positive raw values require grounding
- invalid structures fail closed

Clinical/domain semantics must not be smuggled into generic Core with brittle
regexes unless explicitly proven and scoped.

## V. SUPPORT RELATIONS

```text
DIRECT_EXACT · NORMALIZED · SEMANTICALLY_SUPPORTED
UNSUPPORTED · CONTRADICTED · REVIEW_REQUIRED
```

Software may establish `DIRECT_EXACT` and `NORMALIZED`. Semantic outcomes
require independent verification. A generator may NOT self-declare
`SEMANTICALLY_SUPPORTED`, `CONTRADICTED`, or `UNSUPPORTED`.

## VI. EVIDENCE TRANSPORT

- `copy_span` — exact substring from one source turn
- `relocate` — relocate an exact model-proposed quote when unique
- `snap_relocate` — superficial normalization only: Unicode, case, whitespace

Never perform semantic paraphrase relocation. Ambiguous or missing evidence
means ABSTAIN / REVIEW — not guessed evidence.

## VII. IMPORTANT EVALUATOR CORRECTION

Do NOT equate "predicted span != exact annotated span" with "wrong evidence".

Separate: `EXACT_ANNOTATION`, `SUPPORTING_SUPERSPAN`, `BOUNDARY_OVERREACH`,
`WRONG_MENTION`, `WRONG_SOURCE`, `UNSUPPORTED_EVIDENCE`, `INVALID_SPAN`.

Example — gold `"neck"`, prediction `"My neck has been hurting"` may be:
exact annotation = false, support = `DIRECT_EXACT`, boundary category =
supporting superspan. It is not necessarily a semantic failure.

Preserve exact-span agreement and character F1 as annotation/boundary metrics.
Evaluate semantic/evidence support separately.

## VIII. STRUCTURED OUTPUT IS THE PRIMARY CONTRACT

Brittle text span-port serialization creates avoidable failure. Primary model
interface is structured `CandidateAtom` JSON / constrained structured generation.

Fields: `atom_id`, `atom_type`, `raw_value`, `assertion_state`, `speaker`,
`experiencer`, `temporality`, `certainty`, `evidence_quote` / evidence request,
`abstain`, optional confidence metadata.

Preferred order: JSON-schema constrained generation → tool/function output →
strict JSON with fail-closed parser → textual format only as benchmark/control.

The structured output is NOT trusted truth. It still passes through selector,
construction, evaluation, verification.

## IX. CURRENT DATA STATE

**VERIFY ACTUAL FILES/COMMITS FIRST.**

- `p1_contract_smoke_v1` — 3-case plumbing regression suite
- `p1_screening_eval_v1` — frozen 128-case evaluation suite
- `p1_distill_train_v1` — disjoint training/distillation corpus

A leakage validator exists. A previously contaminated artifact named roughly
`screening_p1_distill.json` was deleted after discovering that evaluation
artifacts must never become training examples.

Maintain strict `TRAIN` / `DEV` / `INTERNAL_TEST` / `FROZEN_SCREENING_EVAL`.

Leakage checks must consider: case IDs, source hashes, normalized-source hashes,
dialogue hashes, evidence tuples, held-out values, surface forms, template/value
combinations, near duplicates.

**Never train on frozen evaluation examples.**

## X. CURRENT QWEN RESULT

- C1: 32 cases, 100% coverage, 0 malformed
- C2: 128 cases, ~0.836 coverage, 0 malformed

A prior report included ambiguous metric normalization such as
`assertion_state_correct > 1`. All metric reporting must expose **numerator,
denominator, rate** and separately label encounter-level averages.

Do not optimize from coverage alone.

## XI. HISTORICAL SCIENTIFIC FINDINGS

- open-vocabulary slots failed much more than closed slots
- closed fields often had approximately zero held-out gap
- scale within the old scratch stack did not automatically solve copying
- slot diversity produced a very large controlled improvement in one experiment
- a pointer head alone did not close the gap
- pretrained transfer radically changed the problem
- Qwen2.5-1.5B historically achieved very strong transfer on the old task
- SmolLM/Qwen scale comparisons were not monotonic
- historical results were usually one base/seed and must not be converted into
  universal scaling laws

> DO NOT ASSUME SIZE ALONE SOLVES EXACT OOD EVIDENCE EMISSION.

## XII. CURRENT NATIVE NANO STATUS

Native Nano is mandatory: our architecture, our code, random initialization,
our weights, trained from scratch.

Historical Nano: ~3.15M decoder, `d_model ≈ 192`, `seq ≈ 512`, RMSNorm, SwiGLU,
RoPE, GQA, tied embeddings, small byte-BPE vocabulary. Historical training was
~32.8M tokens — not the previously misstated ~200M.

The current Native program moved beyond that historical architecture.

## XIII. NATIVE ROUND 1 — CURRENT RESULT

**VERIFY ARTIFACTS.** Round 1 is `COMPLETE_RANKED`. Eight 30M-class runs
completed/recovered. Recovered metric artifacts exist under
`artifacts/native_checkpoints/native30_*/`. Network volume `04himzqxbm`
contains ~90 MB `latest.pt` per run.

Round-1 top mechanisms: **1. evidence_bottleneck · 2. span_port**. Round 2 is
UNBLOCKED.

**DO NOT rerun the full 8-way 30M tournament.** That scientific question has been
answered sufficiently to justify promotion.

## XIV. NATIVE ROUND 2 — PRIMARY NEXT MODEL EXPERIMENT

Promote `EVIDENCE_BOTTLENECK` and `SPAN_PORT` to ~100M parameters. Use fresh
random initialization unless weight-transfer itself is intentionally tested.

Default controlled design: `evidence_bottleneck` seed0/seed1 and `span_port`
seed0/seed1 at 100M.

Hold constant where scientifically appropriate: training data, token budget,
optimizer, scheduler, sequence length, evaluation intervals, precision, loss
weighting except mechanism-specific terms.

Measure: optimization stability, loss, span accuracy, evidence support,
assertion, temporality, experiencer, certainty, open-vocabulary held-out
behavior, coverage, abstention, spurious atoms, sample efficiency, tokens/sec,
GPU utilization, cost.

Use early evaluation. Kill clearly dominated runs. Do not train all arms equally
just for symmetry.

## XV. POSSIBLE NATIVE HYBRID

Do NOT automatically combine the Round-1 winners. First determine their
capability profiles. If `evidence_bottleneck` improves support/state/selectivity
while `span_port` improves evidence transport/copy accuracy, then test
`HYBRID = evidence_bottleneck + span_port` — first one seed, adding another only
if the hybrid shows credible advantage.

The scientific question: **ARE THE TWO MECHANISMS COMPLEMENTARY?**

## XVI. NATIVE AUXILIARY OBJECTIVES

Longer-term evidence-aware Native Nano may expose explicit heads/objectives for:
span start, span end, evidence ranking/copy, atom type, assertion, temporality,
experiencer, certainty, conflict, abstention/calibration, structured output.

```text
L = w_lm*L_lm + w_span*L_span + w_atom*L_atom + w_assert*L_assert
  + w_temp*L_temporality + w_exp*L_experiencer + w_cert*L_certainty
  + w_conflict*L_conflict + w_abstain*L_abstain
```

Every weight must be explicit and versioned. Do not blindly add every head.
Ablate mechanisms.

## XVII. MODEL PROGRAM — NO SINGLE TEACHER

Nano must NOT have one universal teacher. Every donor competes for specific
capabilities.

```text
TRACK A  managed capability/reference models
TRACK B  strong practical/trainable control
TRACK C  large specialist student
TRACK D  Native Nano
TRACK E  independent verifier
TRACK F  agent/tool-policy donor collective
```

> NO MODEL IS "THE TEACHER" IN GENERAL.
> A MODEL MAY BE THE BEST TEACHER FOR A SPECIFIC CAPABILITY.

## XVIII. COMPOSER POLICY

Cursor Composer is a useful research inspiration for tool policy, long-horizon
execution, search, recovery, effort calibration, action sequencing, and stopping
behavior.

**BUT:** Do not harvest/scrape Composer trajectories or Suggestions into Nano
training data. Do not perform model extraction. Do not use Cursor Suggestions to
train Nano as a competing model. Current Cursor terms (updated 2026-08-13)
prohibit this absent separate authorization.

Composer may be used conceptually to ask "What behaviors make an excellent
agent?" — then reproduce those capabilities using legally usable/open systems.

## XIX. OPEN AGENTIC DONOR COLLECTIVE

**A. QWEN3-CODER-NEXT** (`Qwen/Qwen3-Coder-Next` + FP8 variant) — open-weight,
Apache 2.0, 80B total / ~3B active, 256K context, designed for coding agents,
strong tool calling, long-horizon behavior, execution-failure recovery, vLLM
integration, `qwen3_coder` tool parser.
*Donor roles:* tool selection, tool argument generation, error recovery,
long-horizon action policy, IDE/tool-environment adaptation, sparse-MoE/hybrid
architecture inspiration.

**B. MINIMAX-M2.1** — open-weight, modified MIT. Explicitly optimized for
coding, tool use, instruction following, long-horizon planning.
*Donor roles:* planning, persistent task execution, multi-tool workflows,
long-horizon control.

**C. GLM-4.5-AIR** — MIT, 106B total / ~12B active, designed for intelligent
agents, thinking + non-thinking modes, tool use.
*Donor roles:* reason-or-act decisions, general agent control, tool/no-tool
decision, reasoning/action transitions.

**D. GPT-OSS-120B** — Apache 2.0. Agentic workflows, tool use, structured
outputs, reasoning, adjustable reasoning effort.
*Donor roles:* structured actions, reasoning-effort calibration, schema
fidelity, reference judgments.

**E. OPTIONAL HIGH-END REFERENCES** — DeepSeek/Kimi/current frontier open models
when economical and legally appropriate. Secondary references, not automatic
teachers.

## XX. CAPABILITY DONOR MATRIX

Do not ask "Which model wins overall?" Ask "Which model demonstrates the best
policy for this capability?"

Capabilities: tool-needed vs no-tool-needed, tool selection, tool arguments,
evidence retrieval, search query formulation, observation interpretation,
failure recovery, replanning, verification invocation, abstention, stop
decision, long-horizon memory/state, reasoning effort, planning, structured
action output, uncertainty recognition, cost efficiency.

Each capability may have a different donor.

## XXI. NANO AGENT CANARY

Build `NANO_AGENT_CANARY_V1` — 32–64 deterministic / verifiable tasks.

Task families: tool required · tool not required · choose among similar tools ·
invalid/nonexistent tool · wrong arguments · tool timeout · tool error ·
ambiguous observation · multiple sequential tools · verification required ·
information insufficient · abstention required · premature stopping trap ·
repeated-tool trap · state update after tool observation · failure recovery ·
replanning · tool result contradicts prior assumption.

The task environment must have known/automatically verifiable outcomes where
possible.

## XXII. AGENT CANARY MODELS

Compare where practical: Qwen3-Coder-Next-FP8, MiniMax-M2.1, GLM-4.5-Air / FP8,
GPT-OSS-120B, strong current Qwen control, Native Nano agent-policy prototype
later.

Do not self-host every enormous model immediately. Use existing managed
endpoints, cheap providers, or bounded self-hosting depending on economics.

## XXIII. AGENT METRICS

Measure: task success, tool-selection accuracy, argument validity,
unnecessary-tool-call rate, invalid-tool rate, recovery success, observation
utilization, replanning accuracy, steps to resolution, verification-call
precision, verification-call recall, abstention correctness, stop correctness,
state fidelity, latency, tokens, GPU/runtime cost.

**Do not collapse everything into one score.**

## XXIV. NANO TOOL ENVIRONMENT

Build a Nano-native training/evaluation tool environment. Potential tools:
`read_source`, `select_evidence`, `search_record`, `retrieve_prior_state`,
`resolve_entity`, `compare_claim`, `update_state`, `query_knowledge`,
`run_verifier`, `request_review`, `calculate`, `abstain`, `stop`.

Later DomainPacks may add medical knowledge tools, guideline lookup, drug
interaction tools, external structured APIs.

Training, evaluation and deployment should converge toward the SAME tool
semantics. Do not train arbitrary fake tool syntax and then deploy different
APIs.

## XXV. TRAJECTORY COMPILER

```text
TASK
→ multiple donor rollouts
→ normalize states/actions
→ align comparable decisions
→ evaluate each transition
→ select best verified action/trajectory components
→ produce Nano-native training examples
```

**Do NOT assume the best final-answer model took the best path.** Model A may
have the best tool selection, Model B the best recovery, Model C the best
reasoning state, and deterministic software the best stop decision. Nano may
learn verified components from each. The compiled training trajectory can
therefore outperform any one donor trajectory.

## XXVI. THREE FORMS OF CAPABILITY TRANSFER

1. **BEHAVIOR TRANSFER** — state → action (call tool, do not call, verify,
   retry, abstain, stop)
2. **OUTCOME TRANSFER** — state → desired structured result
3. **MECHANISM TRANSFER** — no donor outputs required; study/reimplement
   published ideas: environment-aligned RL, executable rewards, tool-specific
   feedback, adaptive curriculum, MoE routing, hybrid attention,
   reasoning-effort control, recovery training.

Mechanism transfer is especially important for Native Nano.

## XXVII. SELECTIVE CAPABILITY ACQUISITION LOOP

`DISCOVER` → `DECOMPOSE` → `REIMPLEMENT` → `GENERATE` → `VERIFY` → `TRAIN` →
`ABLATE` → `GENERALIZE` → `COMPRESS` → `RETAIN OR REJECT`.

No measurable generalizing benefit: REJECT. Reliable benefit: PROMOTE.

## XXVIII. ADAPTIVE CURRICULUM

Training difficulty should increase as competence increases:

```text
one obvious claim → two similar mentions → wrong-speaker distractor
→ uncertain vs asserted → family vs patient → historical vs current
→ conflicting evidence → multiple encounters → tool required
→ multiple possible tools → failing tool → ambiguous observation
→ verifier required → insufficient evidence → abstention
→ long-horizon state evolution
```

When a task family becomes too easy, generate verified harder variants. Do not
merely add more copies of solved cases.

## XXIX. TARGETED LOCAL FEEDBACK

Do not rely only on sequence-level success/failure. Identify the exact bad
transition: wrong tool, wrong argument, wrong evidence, wrong speaker, wrong
state, missed contradiction, unnecessary action, premature stop, failed
abstention. Train corrective supervision at that state/action decision — much
more sample-efficient than rewarding the entire rollout.

## XXX. STRONG PRACTICAL CONTROL

Current strong P1 control: **Qwen3.8-27B**. Preferred inference checkpoint:
`Qwen/Qwen3.8-27B-FP8`.

RunPod researched configuration: vLLM, 48GB PRO tier where currently compatible,
`MAX_MODEL_LEN ~262144`, FP8 KV cache, GPU memory utilization ~0.90, `qwen3`
reasoning parser, `qwen3_coder` tool parser.

Always resolve live Hub versions/configuration before launch. Do not hard-code
stale RunPod IDs.

## XXXI. LARGE STUDENT PROGRAM

Purpose: determine whether frontier/open-teacher capabilities can be specialized
into a smaller practical model before attempting to reproduce everything in tiny
Native Nano.

Candidate class ~27B–80B. Student-A may preserve Qwen2.5-32B as
historical/trainable reference. Student-B should preferably be a
stronger/current trainable candidate. Qwen3.8 itself is a plausible specialist
student if current Axolotl/training support works cleanly.

Do not select models by prestige. Select by baseline quality, adaptation
support, license, structured output, GPU fit, sample efficiency, cost.

## XXXII. CURRENT STUDENT GATE

Student adaptation / QLoRA remains gated on measuring the actual capability gap,
including assertion-state delta, against strong reference/control models.

Resolve this quantitatively. Create `student_gap_v1.json`. Compare Qwen3.8,
Qwen3-32B-AWQ managed reference, Student-A/B where available. Report transport,
support, assertion, temporal, experiencer, omission, spurious, and abstention
deltas.

Do not train simply because a student checkpoint exists.

## XXXIII. STUDENT ADAPTATION

If a real capability gap exists, RunPod Axolotl Fine-Tuning Serverless is the
primary first attempt. Run a 20–50 step compatibility canary proving: model
loads, LoRA/QLoRA initializes, loss finite, optimizer updates, adapter saves,
artifact persists, adapter reload works.

Then run a bounded adaptation. Potential controlled arms: conservative QLoRA,
higher-rank QLoRA, standard LoRA/SFT. DPO / preference methods only if genuine
preference data exists — do not manufacture fake preference labels.

Evaluate immediately after adaptation. Optimize quality delta / dollar and
quality delta / minute — not training loss alone.

## XXXIV. VERIFIER PROGRAM

Verifier is independent from generators. Input: claim / ClinicalAtom + evidence.
Output: `SEMANTICALLY_SUPPORTED` / `CONTRADICTED` / `REVIEW_REQUIRED`.

The prior 48-case deterministic set reportedly scored 1.0 and is too easy.
Before learned verifier training, expand to ~500–2,000 hard cases including:
wrong mention, wrong source, same-source near miss, supporting superspan,
partial support, negation scope, uncertainty, temporality mismatch, experiencer
mismatch, family history, future vs completed event, multi-span conflict,
plausible unsupported inference.

Run deterministic verifier and lexical verifier. Only train a learned verifier
if residual failures exist.

**Primary risk metric: FALSE ACCEPTANCE OF UNSUPPORTED CLAIMS** — not generic
accuracy.

## XXXV. P2 RESEARCH

Question: can a verified EncounterRecord be compressed without corruption?
Measure salience, omission, unsupported additions, uncertainty retention,
provenance retention, redundancy reduction. P2 remains exploratory until P1 is
strong enough for integration.

## XXXVI. P3 RESEARCH

Question: can EncounterRecord support longitudinal state? Measure entity
continuity, event identity, temporal ordering, supersession, resolution,
persistent problems, medication change, conflicts, historical/current
distinction.

Use P3 failures to determine whether P1 representation needs extension. Do not
preemptively redesign P1 for hypothetical P3 needs.

## XXXVII. DOMAIN-GENERALITY

Create small controlled probes outside medicine: technical incident reports,
research claims/evidence, customer-support interactions, general evidence/state
extraction.

Question: are Nano Core primitives genuinely general, or has Core become an
accidental clinical template engine?

## XXXVIII. RUNPOD EXECUTION ARCHITECTURE

Use RunPod at the highest abstraction that preserves the experiment.

1. **PUBLIC ENDPOINTS** — managed reference models
2. **HUB SERVERLESS — vLLM / SGLang** — standard HF inference, student serving,
   adapter evaluation
3. **AXOLOTL SERVERLESS** — LoRA, QLoRA, SFT, possibly preference tuning
4. **OFFICIAL PYTORCH / AUTORESEARCH PODS** — Native Nano, custom architectures,
   custom losses
5. **PARAMETER GOLF TEMPLATE** — fast-training infrastructure donor, bounded
   small-model experiment environment
6. **CUSTOM POD** — only when maintained RunPod surfaces cannot express the job
7. **INSTANT CLUSTER** — only after a measured single-node scaling bottleneck

This layered design is preferable to treating raw Pods as the default.

## XXXIX. LIVE RUNPOD DISCOVERY

Never trust stale IDs. Before major work query:

```bash
runpodctl user
runpodctl gpu list
runpodctl gpu list --include-unavailable
runpodctl datacenter list
runpodctl pod list
runpodctl serverless list --include-template --include-workers
runpodctl hub list ; runpodctl hub search ... ; runpodctl hub get ...
runpodctl template search ... ; runpodctl template get ...
```

Store: stable owner/name locator, live resolved ID, release version, resolved
timestamp. Do not hard-code old screenshot/template IDs.

## XL. RUNPOD COST CONTROL

Owner authorization previously established an experimental cap. However, the
**LIVE WALLET IS THE PHYSICAL CEILING**.

```text
effective_remaining = min(remaining owner-authorized budget, live funded wallet)
```

Query `clientBalance`, `currentSpendPerHr`, `spendLimit`. Do not rely on old
balance numbers. A research snapshot observed roughly $163.13, but that is
historical by the time you read this. **QUERY LIVE STATE.**

## XLI. NO EMPTY RESOURCE INVARIANT

The campaign previously launched an empty 4090 Pod. Never repeat this.

No paid resource may exist without: `experiment_id`, git SHA, exact command,
data revision, artifact destination, GPU type/count, hourly rate, max runtime,
max projected spend, termination condition.

Within ~5 minutes of launch verify: process exists, GPU memory allocated, GPU
utilization plausible, logs advancing, metrics advancing, artifacts writing.
Otherwise **TERMINATE**.

## XLII. SERVERLESS LIFECYCLE

```text
create → submit batch immediately → collect → queue empty
→ inProgress = 0 → workersMin = 0 → verify scale-down
```

If `workersMin=0` does not stick: **DELETE ENDPOINT.** The campaign already
experienced an endpoint that enforced `workersMin=1` and had to be deleted.
Never pay idle inference burn while analyzing results.

## XLIII. POD LIFECYCLE

When a Pod's scientific question is answered: persist checkpoint/results, verify
external copy, terminate Pod. Do not preserve a GPU merely because its local
filesystem contains useful files.

Persistent storage holds checkpoints, adapters, training datasets, metrics,
prediction dumps, experiment manifests. Git should not contain giant artifacts.

## XLIV. CURRENT RUNPOD RESOURCE STATE

Latest reported open resources included serverless `vllm-6gd1gw3pm5ka3d`
(possible idle-burn concern) and pod `zo1lq44stf9pqm` (~$1.39/hour, SSH
confirmed). Another agent/thread was reportedly handling resource
cleanup/native-state checks.

**DO NOT blindly act on these stale IDs.** Query live RunPod state first. If
these resources no longer exist, record that and move on. If they exist without
attached experiments, terminate safely. Do not duplicate work another active
agent is currently performing.

## XLV. NATIVE COMPUTE TOPOLOGY

For 100M models do not assume B200 is always best. Compare live: A100, H100,
L40S / RTX Pro, B200, B300. Optimize both experiments/dollar and
experiments/wall-clock minute.

For small models, multiple independent arms are usually more informative than
multi-GPU distributed training. **Do NOT data-parallelize a 30M/100M model
merely because GPUs exist.**

## XLVI. AUTORESEARCH

Use only after the baseline Native trainer is proven. Freeze outside the search
agent: evaluation data, metric definitions, leakage rules, architecture family
where appropriate.

Allow bounded search over LR, batch, warmup, loss weights, width/depth, MLP
ratio, GQA, dropout, sequence length, optimizer.

Loop: config → train → held-out dev evaluation → retain/reject → one bounded
mutation. Do not permit the search process to modify frozen test data or success
criteria.

## XLVII. PARAMETER GOLF

Parameter Golf is NOT Nano's scientific objective. Use it as an efficiency
donor. Inspect fast data pipelines, `torch.compile` techniques, kernels, mixed
precision, optimizer implementations, checkpoint overhead, training-loop design,
short-run experimental discipline.

Evaluate: does the technique increase valid Nano experiments / GPU-hour without
degrading scientific comparability? Port only measured improvements.

## XLVIII. SCIENTIFIC METHOD

```text
OBSERVATION → competing explanations → cheapest discriminating experiment
→ result → belief update → architecture/data/software response → regression test
```

Example — observed: support high, exact span low. Possible A: valid wider
evidence. Possible B: wrong mention. Experiment: boundary taxonomy breakdown.
Do not train "better exact spans" until A/B is resolved.

## XLIX. FAILURE TAXONOMY

Classify failures at least as: candidate discovery · raw-value generation · copy
failure · text-contract formatting · structured-contract formatting · quote
paraphrase · boundary precision · wrong mention · wrong source · ambiguity ·
speaker · experiencer · assertion · negation · uncertainty · temporality ·
conflict · omission · spurious atom · malformed structure · mechanical support ·
semantic support · verifier failure · calibration · tokenization · capacity ·
pretraining · objective · data · architecture · evaluator bug · infrastructure
bug · tool selection · tool arguments · tool failure recovery · planning ·
premature stopping · unnecessary tool call.

**Do not summarize failure as "model isn't good enough."**

## L. METRIC FAMILIES

- **TRANSPORT** — exact annotation, supporting superspan, boundary overreach,
  character F1, wrong mention, wrong source, invalid span, ambiguity
- **SUPPORT** — DIRECT_EXACT, NORMALIZED, SEMANTICALLY_SUPPORTED, UNSUPPORTED,
  CONTRADICTED, REVIEW_REQUIRED
- **STATE** — assertion, negation, uncertainty, conflict, speaker, experiencer,
  temporality, certainty
- **SELECTIVITY** — coverage, correct abstention, unnecessary abstention, risk at
  coverage
- **OUTPUT** — omission, spurious atoms, malformed, critical errors
- **AGENCY** — tool choice, arguments, recovery, steps, verification, stop, state
  fidelity
- **SYSTEM** — latency, TTFT, tokens/sec, GPU utilization, VRAM, worker-seconds,
  actual dollars

Never collapse all performance into one scalar without retaining decomposition.

## LI. DISAGREEMENT ENGINE

Build a canonical cross-model case matrix comparing: best managed reference,
Qwen3.8, Qwen3-Coder-Next agent donor, MiniMax agent donor, GLM agent donor,
GPT-OSS, large student, Native Nano, verifier.

For each case classify: all correct · reference only correct · student only
correct · Native only correct · same claim/different evidence · same
evidence/different state · different abstention · tool-policy disagreement ·
recovery disagreement · stop disagreement.

Disagreement is high-value training/research data.

## LII. DATA GENERATION STRATEGY

Do NOT generate millions of generic synthetic examples upfront. Use current
failures, model disagreements, verified boundary cases, state contrasts, and
tool-policy mistakes to generate targeted curricula.

Teacher-data pipeline: proposal → schema validation → evidence validation →
deterministic verification → semantic verification if required → deduplication →
leakage check → provenance → training acceptance.

Labels: `SYNTHETIC_GOLD`, `TEACHER_VERIFIED`, `TEACHER_UNVERIFIED`, `REJECTED`.
Only trusted classes enter high-confidence training.

## LIII. CURRENT IMMEDIATE EXECUTION DAG

Do these in parallel where independent.

- **CONTROL LANE** — verify repository state; commit reproducible execution tree;
  run tests; query RunPod; eliminate idle burn; maintain cost ledger.
- **NATIVE LANE** — promote evidence_bottleneck to ~100M seeds 0/1; promote
  span_port to ~100M seeds 0/1; successive halving; test hybrid only if
  complementarity appears.
- **STUDENT LANE** — compute actual gap; choose best student; Axolotl
  compatibility canary if gap warrants; bounded adaptation if canary succeeds.
- **AGENTIC DONOR LANE** — build `NANO_AGENT_CANARY_V1`; benchmark
  Qwen3-Coder-Next first; add GPT-OSS; add GLM/MiniMax when economically
  practical; identify per-capability donors.
- **VERIFIER LANE** — harden verifier dataset; deterministic baseline; learned
  verifier only if residual problem exists.
- **P2/P3 LANE** — continue cheap representation probes.
- **DATA LANE** — mine disagreements; generate verified targeted cases; maintain
  leakage separation.
- **SYSTEMS LANE** — test autoresearch after Native baseline; inspect Parameter
  Golf efficiency ideas; benchmark only what can change throughput decisions.

## LIV. IMMEDIATE PRIORITY ORDER

```text
Priority 0: resource safety / reproducibility
Priority 1: Native 100M Round 2
Priority 2: student capability-gap measurement
Priority 3: agentic donor canary
Priority 4: Native hybrid if justified
Priority 5: student QLoRA if justified
Priority 6: verifier hardening
Priority 7: P2/P3 probes
Priority 8: autoresearch / Parameter Golf optimization
```

Do not invert this hierarchy merely because one infrastructure path is easier.

## LV. ROUND-2 NATIVE DECISION TREE

- IF `evidence_bottleneck` dominates `span_port` → promote evidence_bottleneck
- IF `span_port` dominates → promote span_port
- IF each dominates different capability axes → test hybrid
- IF seed variance exceeds architecture difference → increase seeds before
  concluding
- IF both 100M variants fail to improve meaningfully over 30M → do not move to
  300M blindly; investigate data, training tokens, objectives, optimization,
  representation
- IF 100M strongly improves → consider 300M winner only if expected information
  gain justifies cost

## LVI. STUDENT DECISION TREE

- IF strong practical control nearly matches operational reference →
  large-student adaptation has low priority
- IF substantial semantic/state gap remains → adaptation is justified
- IF gap is mostly malformed syntax → do not train; solve structurally
- IF gap is mostly annotation boundaries → do not blindly train minimal spans
- IF gap is evidence selection → target evidence curriculum
- IF gap is state prediction → target state objectives/data
- IF QLoRA canary fails → use raw PyTorch/Axolotl Pod or choose better-supported
  student

## LVII. AGENT-DONOR DECISION TREE

- IF Qwen3-Coder-Next dominates tool policy → promote as primary tool-policy donor
- IF MiniMax dominates planning → use MiniMax only for planning curriculum
- IF GLM dominates tool/no-tool decisions → use GLM for that capability
- IF GPT-OSS dominates structured action/reasoning-effort calibration → use
  GPT-OSS for those capabilities
- IF deterministic software beats models on a decision → use software, not
  another neural component

No requirement exists for one donor to win everything.

## LVIII. SOFTWARE VS MODEL

Always ask: **CAN SOFTWARE GUARANTEE THIS CHEAPER AND MORE RELIABLY?**

Good software candidates: exact offset validation, schema validation, evidence
relocation, normalization, unique-match selection, reference closure, resource
scheduling, cost limits, tool schema validation, deterministic stop conditions,
artifact persistence.

Use learned modeling for ambiguity/generalization. Use verification for semantic
judgments.

## LIX. GIT POLICY

One canonical integration owner. Use fresh branches/worktrees for parallel
implementation. Do not wholesale merge historical branches. Do not force-push
master. Do not commit giant checkpoints/logs.

Git holds code, tests, configs, small manifests, small summaries, artifact
hashes/pointers. Persistent storage holds weights, adapters, large datasets,
prediction dumps, logs.

Merge reusable, tested improvements. Do not merge every exploratory experiment.

## LX. AGENT OPERATING STYLE

You are an execution agent. Do not repeatedly ask "Should I continue?",
"Should I launch the GPU?", "Would you like me to run the next test?"

Within the authorized campaign, make routine decisions from evidence, budget,
utilization, information gain, and dependencies.

Checkpoint reports are informational. **Continue independent work after
reporting.** Do not say "I'll report when the worker finishes" while leaving
independent work idle.

## LXI. FORBIDDEN DISTRACTIONS

Do not: restart documentation reconciliation · build continual-learning
infrastructure during this campaign · build transcript memory · update AGENTS.md
for meta-work · rerun solved historical experiments · train on frozen evaluation
· treat teacher output as gold · harvest Composer outputs · self-host
multi-terabyte frontier models without a specific justified experiment · launch
empty Pods · leave Serverless workers idle · train large students before
measuring their real gap · scale Native merely because more parameters are
available · train a learned verifier on an easy solved dataset · replace Nano's
mission with coding-agent imitation · allow later-stage tools/agents to bypass
evidence truth construction.

## LXII. REQUIRED NEXT CHECKPOINT

1. What is current live RunPod wallet/spend rate?
2. Are any billable idle resources left?
3. What exact commit SHA is being executed?
4. Did `evidence_bottleneck` survive promotion to 100M?
5. Did `span_port` survive promotion to 100M?
6. What is seed variance?
7. Are their strengths complementary?
8. Is a hybrid warranted?
9. Does 30M → 100M produce meaningful improvement?
10. What is the actual student capability gap?
11. Is QLoRA justified?
12. Which agentic donor wins tool-policy canary categories?
13. Does Qwen3-Coder-Next materially outperform generic Qwen on tool/recovery
    behavior?
14. Is a learned verifier actually necessary?
15. What do P2/P3 probes imply about the representation?
16. What did each experiment cost?
17. Which experiment has the highest expected information gain next?

## LXIII. REQUIRED FINAL PROGRAM QUESTIONS

- What is the smallest architecture that can reliably represent evidence?
- What inductive biases matter more than scale?
- What capability disappears first as models shrink?
- Can explicit span/state heads outperform pure next-token learning?
- Can strong agentic policy be transferred into a compact system?
- Can tool decisions become explicit low-cost policy heads rather than long
  textual reasoning?
- Can Nano combine deterministic software + compact learned representations +
  selective generation + independent verification + adaptive tool use to
  outperform a much larger generic model on reliability/cost for its target work?

## LXIV. NORTH STAR

Do not attempt to make Nano imitate one large model.

> NANO SHOULD BECOME A SYSTEM THAT SELECTIVELY ACQUIRES THE BEST CAPABILITIES
> FROM MANY SOURCES WHILE REMAINING SMALL, GROUNDED, VERIFIABLE, AND OWNED.

Potential donors: open language models, agentic coding models, reasoning models,
retrievers, verifiers, classical algorithms, databases, search systems, state
machines, planning algorithms, RL, deterministic software. Every donor must
prove its contribution through controlled evaluation.

The desired endpoint is not "small Composer" or "small Qwen" or "small GPT". It
is **NANO** — a compact architecture whose capabilities were deliberately
selected, experimentally validated, compressed, and composed.

## LXV. FIRST ACTIONS FOR A NEW AGENT

1. Verify `origin/master` and campaign branch state.
2. Inspect all current Native Round-1 artifacts and confirm `COMPLETE_RANKED`.
3. Verify the `evidence_bottleneck` / `span_port` ranking from raw metrics.
4. Query live RunPod account, Pods, endpoints, workers and storage.
5. Terminate any resource with no attached valid experiment.
6. Commit the exact reproducible campaign tree if currently dirty.
7. Run the complete local test suite.
8. Audit metric denominators.
9. Prepare four fresh-init Native 100M Round-2 runs.
10. Select the best live compute topology by experiments/$ AND experiments/min.
11. Launch Round 2.
12. In parallel, compute `student_gap_v1`.
13. In parallel, implement `NANO_AGENT_CANARY_V1`.
14. First agentic donor to benchmark: **Qwen3-Coder-Next-FP8**.
15. Add GPT-OSS reference.
16. Add GLM-4.5-Air / MiniMax-M2.1 when cost/deployment permits.
17. Build per-capability donor ranking rather than one model leaderboard.
18. Expand verifier hard set.
19. Keep P2/P3 research probes moving cheaply.
20. Process results continuously rather than waiting for entire batches.
21. Kill dominated experiments early.
22. Generate targeted training data from actual disagreements.
23. Run Axolotl canary only if student adaptation gate passes.
24. Test Native hybrid only if evidence demonstrates complementarity.
25. Use autoresearch only after clean 100M baselines exist.
26. Mine Parameter Golf only for measurable training-system improvements.
27. Persist artifacts before destroying compute.
28. Leave no paid resource idle.
29. Report the next checkpoint with evidence, costs, and decisions.
30. Continue execution automatically.

> DO NOT RETURN ANOTHER HIGH-LEVEL PLAN.
> VERIFY STATE.
> EXECUTE THE HIGHEST-INFORMATION NEXT WORK.
