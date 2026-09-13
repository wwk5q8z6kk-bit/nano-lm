# Software-Engineering Capability — Evaluation Program (Proposal)

**Layer 3 — research portfolio companion. Non-evidential. Not an authorization.**
**Drafted:** 2026-08-27 · **Status:** design proposal, received by owner directive 2026-08-27
**Parent:** [`LABORATORY_CONSTITUTION.md`](LABORATORY_CONSTITUTION.md) · [`RESEARCH_PORTFOLIO.md`](RESEARCH_PORTFOLIO.md) (Program P)

> **SCOPE DECLARATION.** This proposes an evaluation program for a *software-engineering* capability
> that has no footprint in this repository today. The declared frontier is `P1_SCRIBING`
> (`docs/ACTIVE_NOW.md`). This document does not reprioritize P1/P2/P3 and does not register
> anything. `PROGRAM1 = NOT_AUTHORIZED` and `PROGRAM0 = INFRA_COMPLETE_STOP_EXPANSION` remain in
> force (`benchmarks/BENCHMARK_CONSTITUTION.md`). Activating any tier below requires owner
> authorization and, for Tiers 1–3, Program 1 authorization plus compute authorization.

> **OWNER DIRECTIVE, 2026-08-27.** Received as a *design proposal*, not authorization to expand the
> benchmark registry or the execution queue. Sequencing preserved: do not turn the catalog into an
> execution queue; do not register public suites merely because they appear here; close the
> infrastructure/identity gaps first; define agent-run identity and budget/environment pins before
> collecting agentic measurements; develop native process-intelligence evaluations as the research
> contribution; keep public suites as later diagnostic/validation layers subject to authorization and
> compute constraints.
>
> **Work delivered under that directive:** (a) the identity infrastructure required to make future
> evaluation scientifically identifiable and reproducible — §2.1; (b) the Class A and Class B
> process-intelligence instruments named by item 5 of the sequencing above — §4.1, §4.2. All are
> instrument definitions with tests: no corpus, no runner, no registry row, no execution, no promoted
> gate. The three-layer principle is frozen verbatim in [`../docs/EVALUATION_FRAMEWORK.md`](../docs/EVALUATION_FRAMEWORK.md).
>
> **Owner decision, 2026-09-13 — fixture-first emitter.** Sequence: freeze the seven commits; build a
> sanitized fixture corpus; build the transcript → action-log emitter against it; test `action_class`
> aggressively; run the suite; measure the 62 real transcripts only as a labelled *instrument
> validation*; audit the logs by hand; decide on contract revision; then merge. Boundary: "we have a
> corpus" must not quietly become "we have benchmark evidence". Delivered — see §4.4.
>
> **Open and owner-gated:** Program 1 authorization for Tiers 1–3. Class C (expert annotation)
> remains unbuilt by design. The v1.1 contract revision (§4.4) is specified and awaits its own
> review.

---

## 1. What is accepted without argument

The critique of the benchmark catalog is correct on its central point: the catalog mixes static model
knowledge, isolated reasoning, code generation, repository engineering, agent scaffolding, tool
execution, safety, operational performance, and product usefulness into one list, and a system can
score well across most of it while being a poor engineer. The proposed remedies that this document
adopts wholesale, without restatement:

- splitting "coding" into function-level generation / execution reasoning / repository understanding /
  competitive programming / issue resolution / greenfield construction / maintenance-and-evolution;
- adding the missing families: novel-task generalization, contamination-resistant rolling evaluation,
  deep research and information acquisition, computer-use and OS agents, terminal and environment
  operation;
- treating engineering *process* (investigation, hypothesis revision, change-impact prediction,
  verification sufficiency, escalation) as a first-class evaluation family rather than an implied
  by-product of outcome scores;
- reporting a capability vector with hard gates instead of a weighted benchmark average.

The rest of this document is the part that is *not* a restatement: what breaks when this is mapped
onto the machinery this repository already has.

---

## 2. Three corrections, grounded in existing repo machinery

### 2.1 The pin contract cannot identify an agentic run — this is a blocking defect

The critique's sharpest claim is in its §3E: the evaluated unit is

```text
model + agent policy + tools + environment + budget
```

not a model checkpoint. That claim collides directly with the identity contract already adopted.
`benchmarks/BENCHMARK_CONSTITUTION.md` § *Pins (identity)* binds every countable run to:

| Pinned today | Covers |
|---|---|
| `lm-eval` package version + git commit | harness |
| task name, task version, task YAML hash | task |
| dataset / instrument hash, record count, schema version | instrument |
| prompt/template hash, metric implementation hash, filter-pipeline hash | scoring |
| model or solver manifest hash | model |
| code git commit | repo state |
| content-addressed `run_id` | run |

There is **no slot for agent policy, no slot for the tool set, and no slot for the budget.** Two runs
of the same checkpoint under different scaffolds, different tool inventories, or different step caps
produce the same pin set and different scores. **The pin set as specified cannot distinguish runs
that differ only in agent policy, tool inventory, or budget** — so Property 1 (*Identity*) and
Property 2 (*Reproducibility*) of the Program 0 six properties are underdetermined for agentic
suites, as a matter of what the contract enumerates rather than of rigor preference.

**Verified at code level, not only in the prose contract.** `compute_run_id`
(`benchmarks/adapters/lm_eval/schemas.py:54-67`) takes exactly four inputs — benchmark manifest,
model/solver manifest, config, code commit. The one adjacent component that *is* computed,
`environment_hash`, is built at `benchmarks/adapters/lm_eval/runner.py:217` and stored on the run
manifest at `:302`, but is **not** passed to `compute_run_id` at `:219-224`. Environment is
therefore recorded and deliberately not identity-bearing.

Anticipating the obvious rebuttal — "`config_sha256` is in `run_id`, so put scaffold and budget in
`config`": identity binding here is by **explicit enumeration**, and the treatment of
`environment_hash` shows what the contract does with an adjacent component it declines to bind.
Nothing requires agent policy, tool inventory, permissions, budget, or retry policy to appear
anywhere, and an incidental catch via an unspecified `config` blob is not a pin.

**Consequence for sequencing: no agentic suite may be registered before the pin contract is
extended.** This is cheap to fix and expensive to retrofit, because scores collected under an
ambiguous pin set are not comparable to scores collected after.

**Status: closed.** Under the owner directive of 2026-08-27 ("the next engineering work should be
limited to whatever infrastructure is required to make that future evaluation scientifically
identifiable and reproducible"), the identity contract has been built:

| Artifact | Role |
|---|---|
| `benchmarks/AGENT_RUN_IDENTITY.schema.json` | the eight-component contract, machine-checkable |
| `benchmarks/agent_identity.py` | component pins + `compute_agent_run_id`, fail-closed |
| `benchmarks/tests/test_agent_identity.py` | one distinguishability test per component |

The eight identity-bearing components, matching the owner's enumeration: model/checkpoint · agent
policy/scaffold · tool set **and permissions** · environment/container state · task specification ·
budget **and retry policy** · evaluator/harness version · code/configuration commit.

Three design points carry the argument above into code. `compute_run_id` is **not** modified — its
output is the on-disk directory name for every existing run, so a fifth input would rewrite
historical identities; `compute_agent_run_id` is additive and carries the legacy id as
`base_run_id` lineage. `retry_policy` and `repeats` sit inside identity rather than beside the
result, because unreported retries are an integrity failure and pinning the policy is the only
structural defense; `repeats` is required because pass^k is undefined without it (§5.2). Permissions
are identity-bearing alongside the tool inventory, because the same tools read-only versus writable
are a different system.

Fail-closed is deliberate: absent, malformed, and placeholder (null-digest) pins raise rather than
produce a well-formed id. The honest limit is that identity is only as strong as the manifests
beneath it — these checks catch unpinned components, not dishonest ones.

### 2.2 Layer 14 (benchmark validity) already exists here — do not build a parallel apparatus

The critique's §6 list — contamination, memorized patches, solution leakage, grader exploitation,
harness manipulation, environment leakage, judge bias, nondeterminism, cherry-picking, scaffold
differences, unreported retries, version drift, saturation — overlaps almost entirely with machinery
already adopted in this repo. Mapping it onto what exists is strictly better than inventing a
fourteenth layer of new process:

| Critique's integrity concern | Existing mechanism |
|---|---|
| memorized patches, solution leakage, contamination | `contamination_status` field per suite (`benchmarks/REGISTRY.yaml`); *Living target* clause requiring contamination risk < ε_b |
| nondeterminism, unreported retries | Property 2 *Reproducibility*; Gate 0 row `REPEATED_RUN_REPRODUCIBILITY` |
| grader exploitation, harness manipulation | Property 3 *Auditability* (every aggregate traces to per-item outputs); Gate 0 row `PER_ITEM_TRACEABILITY` |
| environment leakage, scaffold differences | *Pins (identity)* — insufficient as specified; closed additively by `benchmarks/AGENT_RUN_IDENTITY.schema.json`, see §2.1 |
| cherry-picked samples, hidden failures | Property 5 *Failure visibility* (FAILED/PARTIAL/VOID/CANCELLED preserved); Gate 0 row `FAILED_RUN_PRESERVATION` |
| judge-model bias | **no existing mechanism** — genuinely new, see §5 Class C |
| version drift | `harness_pin`; Gate 0 row `TAG_INTEGRITY` |
| benchmark saturation | *Living target* (public ≈ hidden ≈ dynamic ≈ OOD ≈ product within tolerance) |
| score laundering into claims | *Separation of records*: benchmark result ≠ ledger claim ≠ product authorization |

Two gaps survive that mapping: **agentic run identity** (§2.1 — now closed) and **judge-model bias**
(specified below, not built). Those two are the actual Layer-14 work. Everything else is already
specified and should be reused rather than re-derived.

The bench gates 0–8 (`truth → localize → construct → pilot → replicate → OOD → full suite → product
transfer → public claim`) are likewise reusable unchanged for agentic suites.

#### Judge-model calibration protocol (specification only — deliberately not built)

Unlike identity, judge calibration carries **no retrofit hazard**: a judge can be calibrated at any
point before its scores are used, and no judge is used until Tier 4 Class C. Building κ machinery now
would be speculative infrastructure for a consumer that does not exist, so this is a spec, to be
implemented when a judge is first wired. The `evaluator_pin` already carries a
`judge_calibration_sha256` slot so that a calibrated judge is identity-bearing when one appears.

Required calibration record, per (judge model, judge version, prompt, rubric, task family):

```text
judge_id, judge_model_version, judge_prompt_sha256, rubric_sha256, task_family
human_label_set_sha256, n_items, n_annotators, annotator_agreement (inter-human)
judge_vs_human_agreement, cohens_kappa, per_class_confusion
decision_threshold, calibration_date, expiry_condition
```

Gates before judge scores may enter any table:

1. $n \geq n_{min}$ on a **held-out** human-labeled set (not the set used to write the rubric);
2. judge–human agreement must be reported as κ, not raw agreement — raw agreement is inflated by
   class imbalance, which is severe here because most findings are negative;
3. judge–human κ must not exceed inter-human κ by more than noise; a judge that "agrees" better than
   humans agree with each other is measuring the rubric, not the phenomenon;
4. the judge must never score items it also generated or repaired;
5. recalibration required on any change to judge model version, prompt, or rubric — each is in the
   record's identity, so a change invalidates the record by construction.

Failure mode this exists to prevent: converting an annotation cost (Class C) into an unmeasured bias
by substituting an uncalibrated judge for the labels.

### 2.3 The proposed composite conflicts with a utility form already adopted

The critique proposes:

```text
Engineering Outcome × Verification Reliability × Safety Compliance × Calibration ÷ Normalized Cost
```

`docs/EVALUATION_FRAMEWORK.md` already adopts a different form, with frozen weights:

$$U = Q - w_E E - w_R R - w_L L - w_C C$$

These are not interchangeable, and the additive form is the better one to keep:

1. **Unbounded sensitivity.** A five-way product goes to zero as any factor goes to zero and is
   dominated by whichever factor is worst-measured. With small $n$, the noisiest factor drives the
   ranking.
2. **Division by cost inverts the objective near the floor.** Maximizing $X/\text{cost}$ rewards doing
   very little very cheaply unless an outcome floor is enforced separately — so the gates are not a
   supplement to the composite, they are load-bearing for it.
3. **The repo's form already frozen-weights the penalties** and has an explicit kill/keep rule (arm
   must beat the best cheaper solver by δ, default 0.05 unless preregistered).

**Recommendation:** keep $U$ as the comparison instrument; treat the multiplicative composite as a
*display* statistic only, marked provisional, never a gate, never a promotion criterion. This is
consistent with the critique's own conclusion that gates should dominate the composite — it simply
takes that conclusion one step further and declines to promote the composite at all.

---

## 3. The 14 layers against current repo footprint

Adopting the critique's taxonomy verbatim, annotated with what exists here and what blocks each layer.

| # | Layer | Repo footprint today | Blocking condition |
|---|---|---|---|
| 1 | Language and knowledge | none | Program 1 + compute auth |
| 2 | Atomic reasoning | none | Program 1 + compute auth |
| 3 | Generalization (ARC-AGI-2, rolling suites) | none | Program 1 + compute auth |
| 4 | Code primitives | none | Program 1 + compute auth |
| 5 | Repository intelligence | none | Program 1 + compute auth |
| 6 | **Engineering process** | Class A + Class B instruments built (§4.1, §4.2); **no public suite covers this layer** | a corpus; Class C annotation; *not* blocked on Program 1 |
| 7 | Tool competence (terminal, OS, browser, IDE) | none | Program 1 + sandboxed exec environment |
| 8 | Agentic execution | identity contract present (§2.1); no suite, no runs | Program 1 + compute auth |
| 9 | Verification sufficiency | partial — abstention economics in `wedge_v1/`; verifier relation in `docs/EVALUATION_FRAMEWORK.md` principle 4 | instrument design |
| 10 | Human collaboration | none | annotation protocol + human subjects |
| 11 | Safety and governance | partial — owner gates, `NOT_AUTHORIZED` markers, freeze isolation (Property 4) | agent-action policy manifest |
| 12 | Operational efficiency | partial — Pareto boards specified in `benchmarks/BENCHMARK_CONSTITUTION.md` (score/param, /FLOP, /\$, /joule, robustness, reliability-at-coverage, quality-under-latency) | measurement plumbing only |
| 13 | Real-world utility | partial — `wedge_v1/` product harness pattern | product surface |
| 14 | Benchmark validity | mostly present (§2.2); agent identity closed (§2.1) | judge calibration — specified, built when a judge is first wired |

The two rows worth acting on first are **6** and **14**, precisely because they are the two that are
*not* gated on Program 1 authorization or paid compute. Layer 6 has no public instrument to buy;
Layer 14's remaining gaps are contract work, not measurement work.

---

## 4. Layer 6 is uncovered because the metrics lack instruments, not because nobody wanted them

The critique proposes roughly sixty process metrics across seven categories: architecture discovery,
investigation quality, hypothesis-driven debugging, change-impact prediction, design judgment,
verification sufficiency, uncertainty and escalation. They are not equally buildable. The distinction
below *is* the specification — it converts a wish list into a build order.

The instrument is an **agent action log** with one record per agent action: sequence index, declared
action class, tool, normalized-argument digest, targets, result status and digest, post-state digest
for mutations, and token/cost accounting. This is the minimum instrument; nothing in Layer 6 is
measurable without it. (Named *action log*, not *trajectory log*: `trajectory/` in this repository is
the Layer-1 evidence directory, and reusing the word would collide.)

**Status: built.** `benchmarks/AGENT_ACTION_LOG.schema.json` is the record contract;
`benchmarks/process_metrics.py` implements every Class A metric below as a pure function over a log;
`benchmarks/tests/test_process_metrics.py` covers one case per metric plus the empty-log and
single-action edges. Instrument definitions only — no corpus, no runner, no emitter, no CLI, and no
threshold promoted to a gate.

Three definitional choices the prose above left underdetermined, fixed explicitly in code because
tests would otherwise silently bless whichever reading was implemented:

- **`premature_edit_rate` is per mutated file, not per run.** A run that reads and edits `a.py`, then
  blind-edits `b.py`, is premature on `b.py`; a first-action-only run-level reading scores it clean.
- **`repeated_action_rate` requires an *unchanged result digest*.** Re-running a test after an edit is
  correct behavior; only a repeat returning the same result is evidence of not learning. Comparison is
  over the normalized-argument digest, never raw argument dicts.
- **`verification_attempted` reads a declared `action_class`**, and `destructive_action_rate` and
  `scope_violation_rate` take the tool set and scope as parameters. Which tools verify or destroy is a
  property of the harness, not a universal fact; hardcoding a name list would be wrong on every
  harness but the first.

Two conventions carry through: every rate returns **undefined rather than zero** when its denominator
is zero — a run that mutated nothing has no premature-edit rate, and collapsing that to 0.0 would let
an agent that did nothing outscore one that did the work — and the reported artifact is a **profile
vector**, never a scalar.

### 4.1 Class A — derivable from the trajectory log alone (zero annotation)

Buildable today for any task, with no gold labels.

| Metric | Definition on the log |
|---|---|
| Action count / tool calls / environment steps | count of records |
| Invalid tool-call rate | records whose result is a schema or parse error ÷ total |
| Repeated-action rate | records whose (tool, normalized args) tuple recurs after an unchanged result |
| Premature-edit rate | runs whose first mutating action precedes any read of the mutated file |
| Reproduction-before-repair rate | runs where a failing-test execution precedes the first mutating action |
| Verification-attempted rate | runs with ≥1 test/build invocation after the last mutating action |
| Reversion count | mutating actions that restore a prior file state |
| Files inspected / changed | distinct paths by action class |
| Wall-clock, tokens, cost | accounting fields |
| Destructive-command rate | actions matching a declared destructive-pattern list |
| Scope violation rate | mutating actions on paths outside a declared per-task scope manifest |
| Rollback success | post-run environment state equals pre-run state on abort |

`verification-attempted` and `premature-edit` are the two highest-value members of this class: they
are structural, cheap, and target the exact failure the critique names (patch without verification;
edit without evidence).

### 4.2 Class B — derivable from the log plus gold artifacts that issue-resolution tasks already ship

Requires only what SWE-bench-style tasks already contain (a gold patch and fail-to-pass / pass-to-pass
test sets). No new annotation.

| Metric | Definition |
|---|---|
| Localization accuracy | files touched ∩ gold-patch files ÷ gold-patch files |
| Localization precision | files touched ∩ gold-patch files ÷ files touched |
| Actions to first correct file | index of first mutating action on a gold-patch file |
| Affected-test recall | tests the agent ran ∩ gold F2P∪P2P ÷ gold F2P∪P2P |
| Regression escape rate | P2P tests passing before and failing after ÷ P2P tests |
| Claimed-complete-but-failing rate | runs where the agent declares completion ∧ F2P still fails |
| Fix minimality | lines changed ÷ gold-patch lines changed |

`claimed-complete-but-failing` is the single most diagnostic number in the whole proposal: it is the
direct measure of the failure the critique opens with — an apparently valid patch, no noticed
collateral damage, no verification — and it costs nothing beyond Class A plus the gold test sets.

**Status: built.** `benchmarks/gold_metrics.py`, with `benchmarks/tests/test_gold_metrics.py`. Class B
turned out **not** to be corpus-blocked: the gold artifacts are function parameters (`GoldTask`,
`RunOutcome`), never a bundled dataset, so the instrument is definable and testable against synthetic
tasks while the corpus decision stays open. Four design points:

- **Gold results come from the grader, not the log.** `RunOutcome` is a separate input from the action
  log precisely so an agent's own claim about its tests can never become its score.
- **A missing test result is not a pass.** An F2P test with no recorded outcome counts as failing, and
  a P2P test with no post-run result counts as broken.
- **Regression escape counts only tests that were green beforehand** — a test already red at the start
  is not a regression this run caused.
- **Localization recall and precision are reported separately, never merged.** An agent that rewrites
  the repository scores perfect recall; precision is what exposes it.

`aggregate_rate` deliberately returns `(successes, n)` rather than a float: §5.1 requires safety gates
to be stated as a Wilson upper bound, which needs both terms, and returning `0.0` over twelve runs
invites exactly the reading the bound exists to prevent.

### 4.3 Class C — requires new annotation (expensive; this is the real frontier)

Each of these needs a per-task gold artifact that does not exist in any public suite, and each unit of
annotation is expert labor.

| Metric | Annotation required |
|---|---|
| Relevant-evidence recall / missed high-value evidence | per-task gold *relevance set* over files and symbols |
| Information gain per action | a defined belief state + a scoring rule over it |
| Architecture graph precision / recall | gold subsystem/boundary/dependency graph per repository |
| Hypothesis ranking quality | gold ranked hypothesis list per bug |
| Belief-update correctness | gold posterior ordering after each evidence increment |
| False root-cause confidence | gold root cause + agent-emitted confidence (see Class D) |
| Side-effect prediction precision | gold impact set per change |
| Alternative coverage / trade-off recognition | gold design-alternative set + rubric |
| Design-review preference | blinded expert pairwise preference |
| Escalation precision / recall, useful-question rate | gold escalate/continue label per decision point |
| Adversarial-test effectiveness | mutation set or injected-fault suite per task |

Two notes on cost control. First, **mutation testing substitutes for annotation** on
adversarial-test effectiveness and partially on verification sufficiency: injected faults are
generated, not labeled. Second, **judge-model scoring is the tempting shortcut here and is exactly
where §2.2's remaining gap bites** — any LLM judge used for Class C must itself be calibrated against
human labels (report agreement, e.g. Cohen's κ, on a held-out set) before its scores enter any table.
An uncalibrated judge converts an annotation cost into an unmeasured bias.

### 4.4 Class D — requires a protocol change, not just a measurement

| Metric | Protocol change |
|---|---|
| Confidence calibration, Brier / ECE | agent must emit a success probability before the outcome is known |
| Risk–coverage curve, selective accuracy | agent must emit an abstain/proceed decision with the probability |
| Escalation precision/recall | agent must emit an explicit escalate action rather than implying it in prose |

These are cheap in code and consequential in contract: they change what a compliant agent must
produce, so they must be decided before any agentic scores are collected, for the same reason as
§2.1. **Partially pre-empted:** the action-log schema already carries a per-action `confidence` field
and an `escalate` action class, because a probability emitted before the outcome is known cannot be
back-filled into a log already collected. The calibration *metrics* (ECE, Brier, risk–coverage) are
deliberately not built — unlike the fields, they carry no retrofit hazard and have no consumer yet. Note that this repo already treats abstention as first-class (`wedge_v1/` abstention economics,
Program D — Calibration in `RESEARCH_PORTFOLIO.md`), so Class D is the closest of the four to
existing work.

**Build order implied by the triage:** Class A → Class D (protocol, before any data is collected) →
Class B → Class C. Class A, the Class D protocol fields, and Class B are built; Class C is not.

### 4.4 The instrument on real trajectories — emitter, fixtures, validation, audit

**Status: built, validated, audited.** `benchmarks/transcript_emitter.py` turns a Claude Code session
transcript into a schema-valid action log plus a per-action *classification trace* (rule fired,
evidence, compound flag, notes), so the schema's weakest assumption — `action_class` is declared and
unverifiable — is auditable rather than trusted. It is developed only against checked-in fixtures
(`benchmarks/fixtures/transcripts/`, produced by `benchmarks/transcript_sanitizer.py`); the single
component that reads real transcripts is `benchmarks/validation/instrument_validation_transcripts.py`.

What the validation established, on 62 sessions and 1899 actions: every log validates; raw and
sanitized emissions agree on every classification field (so the fixtures are trustworthy); no tool is
unmapped; 180 classifications were read by a person, 3 were wrong (2 fixed generically), 20 were the
right class with a lost target. Full audit and per-bucket verdicts:
`artifacts/instrument-validation-transcripts/audit-and-contract-decision.md`.

What it established about the *instrument*, not about any agent — the three contract-level findings:

- **Compound edit-then-test calls lose an extractable mutation target: 29 of 45.** The precedence
  that puts `verify` above `mutate` in one Bash call assumed heredoc targets are unextractable; on this
  corpus 64% name an in-repo file. Neither label is right because the call is not atomic.
- **`verification_attempted` is dominated by trailing artifact writes.** Two of four "no" verdicts on
  large sessions are a report write after a real verify; 148 of 460 mutations are outside the repo.
  The metric cannot tell a source edit from a report write.
- **Investigation by execution is invisible.** 114 bare python probes are `execute`; many are
  evidence gathering. The read/mutate/verify/execute taxonomy has no place for "ran code to learn".

**Decision (step 8): the contract needs a v1.1 revision — `call_id` for ordered atomic sub-actions
per tool call, `pre_state_sha256` on mutations, and `declared_scope` as a harness declaration of
*source* paths with scoped metrics.** Specified in the audit document; deliberately not implemented
on the merged branch, because it moves every pinned count and the fixtures, tests, and audit all
encode v1. The taxonomy gap is recorded as the question Class C annotation will have to answer. Classes A, B, and D together cover investigation discipline, verification
sufficiency, regression safety, and calibration — most of the critique's engineering-process concern —
at zero annotation cost. Class C buys architecture, design judgment, and belief revision, and it is
where the budget actually goes.

---

## 5. Aggregation: gates first, composite never

### 5.1 Gates

Adopt the critique's gate list, with one change: **a gate on a rate must be stated as a confidence
bound, not an observed count.** "Destructive-action violations exceed zero" is unfalsifiable at small
$n$ — zero observed violations in 12 runs is consistent with a 20% violation rate. Use the Wilson
score upper bound (`rules/math-toolkit.md` §2), which forces the sample size to do the work:

```text
gate_pass(violations, n, ε) :=  wilson_upper(violations, n, z=1.96) ≤ ε
```

**Status: built** — `benchmarks/confidence.py`, with `benchmarks/tests/test_confidence.py`.
`wilson_upper(0, 12) ≈ 0.24`: twelve clean runs are consistent with a true violation rate near one in
four, which is the concrete reason an observed-count gate is unfalsifiable at small n. Wilson was
chosen over Wald (degenerate exactly at the boundary where safety gates live), Clopper–Pearson
(needs an incomplete beta and buys conservatism that isn't earned), and the naive ratio (no
uncertainty at all). `failure_gate_passes` returns `None` — never `True` — when n is below `n_min`,
because an undecided gate must not read as a passed one, and `runs_needed_for_zero_observed(ε)`
answers the planning question before a gate is proposed rather than after twelve runs. The module
implements the bound; it promotes no gate and picks no ε.

| Gate | Form | Note |
|---|---|---|
| Regression escape | `wilson_upper(escapes, n) ≤ ε_reg` | ε preregistered per suite |
| Destructive / scope violation | `wilson_upper(violations, n) ≤ ε_safety` | ε_safety near zero forces large n — this is intended |
| Calibration on failed tasks | ECE ≤ ε_cal on the failure subset specifically | aggregate ECE hides overconfidence-on-failure |
| Hidden-repository collapse | hidden score ≥ public score − δ_hidden | δ preregistered; this is the *Living target* clause |
| Verification skipped | verification-attempted rate ≥ τ_verify | Class A metric |
| Scaffold dependence | score must survive scaffold substitution within δ_scaffold | requires §2.1 agent pin to even state |
| Cross-seed reproducibility | pass^k ≥ τ_k over `repeats` | see §5.2 |

Any gate failure blocks promotion regardless of outcome score. This ordering is the critique's own and
is adopted unchanged.

### 5.2 Reliability: report pass^k, not only pass@k

For agentic suites the operationally meaningful question is "does it work every time", not "does it
work at least once in k tries". Report **pass^k** (all $k$ independent attempts succeed) alongside
pass@1, with `repeats` pinned per §2.1. pass@k is a capability ceiling; pass^k is the reliability
number a deployment decision needs, and the two diverge most on exactly the long-horizon tasks this
program cares about.

### 5.3 The capability vector is the primary artifact

Report thirteen components, never collapsed: knowledge · novel reasoning · code semantics ·
repository understanding · debugging · design · tool use · verification · uncertainty · human
collaboration · safety · efficiency · generalization. Each carries its own $n$, its own confidence
interval, and its own contamination status. The composite from §2.3 may appear as a display row,
marked provisional, and may never gate.

Promotion of any component metric to a gate follows the existing rule set (`rules/math-toolkit.md`
§16): $n \geq n_{min}$, false-positive and false-negative modes tested, baseline documented,
threshold justified, recheck command exists.

---

## 6. Tier realism — what each tier is actually blocked on

The critique's five tiers, annotated with the *named* blocker rather than a uniform roadmap.

| Tier | Content | Status | Blocked on |
|---|---|---|---|
| 1 — general capability floor | MMLU-Pro, GPQA, HLE subset, ARC-AGI-2, math, IFEval, HELMET | **blocked** | Program 1 authorization (`LABORATORY_CONSTITUTION.md:134` lists large suites as not authorized) + compute authorization |
| 2 — code and software primitives | LiveCodeBench, EvalPlus, BigCodeBench, CRUXEval | **blocked** | Program 1 authorization + compute authorization |
| 3 — real engineering outcomes | SWE-bench variants, Terminal-Bench, private repos, greenfield, migration | **blocked** | Program 1 + compute + sandboxed execution environment (§2.1 pin extension no longer blocking) |
| 4 — cognitive-process evaluations | Class A/B/D metrics from §4 | **Class A + Class B instruments built** (§4.1, §4.2); Class C needs annotation | a corpus to run against — owner's call. No Program 1 dependency, no paid compute, no annotation for A or B |
| 5 — product and economics | completion per dollar, acceptance without rework, escalation quality | **partially buildable** | needs a product surface; `wedge_v1/` is the existing pattern |

The asymmetry is the actionable finding: **Tier 4 is the only tier that is unblocked, and it is also
the tier the critique identifies as least covered by public work.** Those two facts point the same
direction. Tier 4 also has a sequencing constraint on the others — Class D is a protocol change, so
if Tiers 1–3 are ever authorized before Tier 4 exists, their runs will lack the confidence emissions
that make calibration measurable, and they will have to be re-run.

The honest statement of Tier 4's own prerequisite: it needs *some* task corpus to run against. It
does not need a public one. Class A metrics are computable on any agent trajectory, including
trajectories from this repository's own development history.

---

## Appendix A — proposed registry rows (NOT applied)

`benchmarks/REGISTRY.yaml` is schema-validated, consumed by `benchmarks/reports/GATE0_PROGRAM0.md`,
and governed by `PROGRAM0 = INFRA_COMPLETE_STOP_EXPANSION`. Adding suite rows would be a
machine-readable assertion that these families are registered, which is exactly the expansion Program
0 forbids. **This document adds nothing to the registry.** The rows below are what a future,
authorized Program 1 would add, recorded here so the proposal is concrete:

```yaml
# ILLUSTRATIVE ONLY — not present in benchmarks/REGISTRY.yaml
- id: suite_swe_issue_resolution   # family: repository_engineering  (Tier 3)
- id: suite_code_primitives        # family: capability             (Tier 2)
- id: suite_terminal_env           # family: tool_competence        (Tier 3)
- id: suite_computer_use           # family: tool_competence        (Tier 3)
- id: suite_process_intelligence   # family: process                (Tier 4)  ← only unblocked row
- id: suite_novel_generalization   # family: capability             (Tier 1)
```

Every one of these requires `contamination_status`, `visibility`, `protected_metrics`, and a
`classical_baseline_required` decision before it can be written. The §2.1 identity contract is now
in place, so that is no longer the blocker; Program 1 authorization is.

## Appendix B — what this document does not do

- does not amend `docs/ACTIVE_NOW.md` or `docs/CAPABILITY_LADDER.md`. It *does* add the frozen
  three-layer principle to `docs/EVALUATION_FRAMEWORK.md`, quoted verbatim from the owner directive
  of 2026-08-27 — the one authority-layer edit, and the only one;
- does not modify `benchmarks/REGISTRY.yaml`, `REGISTRY.schema.json`, or any Gate 0 report;
- does not modify `compute_run_id` or any existing run identity; the agent-identity module is
  strictly additive (`benchmarks/agent_identity.py`);
- promotes no metric to a gate — `benchmarks/process_metrics.py` is diagnostics only;
- builds an emitter and a *fixture* corpus, and runs the instrument over this project's own
  sessions as a labelled instrument validation — none of which is benchmark evidence, enters the
  evidence ledger, or bears on any model;
- does not touch `papers/EVIDENCE_LEDGER.md`, `artifacts/` manifests, `SHA256SUMS`, or protected tags;
- does not authorize training, paid compute, Program 1, or any benchmark execution;
- does not reprioritize P1 scribing, which remains the declared capability frontier;
- makes no claim about any model's performance on any benchmark named in it.
