# benchmarks/

These directories are **non-evidential research infrastructure and aspiration scaffolding**.

Their existence does **not** establish implementation completeness, benchmark success,
scientific evidence, product authorization, or leadership on any suite.

Layer-1 truth remains `papers/EVIDENCE_LEDGER.md` and the post-α freeze artifacts.

---

## What is here

| Path | What it is | Status |
|---|---|---|
| `BENCHMARK_CONSTITUTION.md` | Program 0 charter, six properties, bench gates 0–8, pin contract | governing |
| `REGISTRY.yaml` / `REGISTRY.schema.json` | suites, tasks, resource classes | one operational task (held-value sentinel); all other suites `planned` |
| `adapters/lm_eval/` | pinned harness adapter, run identity, smoke runner | operational |
| `AGENT_RUN_IDENTITY.schema.json` + `agent_identity.py` | eight-component identity for an **agentic** run | instrument definition; no runs |
| `AGENT_ACTION_LOG.schema.json` + `process_metrics.py` | agent action log + Class A engineering-process metrics | instrument definition; no corpus |
| `gold_metrics.py` | Class B metrics — log plus gold patch and test sets | instrument definition; no corpus |
| `confidence.py` | Wilson bounds for rate gates | supporting; promotes no gate |
| `transcript_sanitizer.py` | structure-preserving redaction of Claude Code transcripts | produces the fixtures |
| `fixtures/transcripts/` | 4 sanitized sessions + 1 synthetic, with README and MANIFEST | instrument fixtures, not evidence |
| `transcript_emitter.py` | transcript → action log + classification trace | v1; v1.1 revision specified |
| `validation/` | runs the v1.0 emitter over real sessions as a labelled instrument validation | reads `~/.claude/projects/` (frozen) |
| `v1_1/` | contract v1.1: schema, emitter, metrics, 15 semantic fixtures, migration comparator, the authorized side-by-side rerun | separate path; v1.0 untouched |
| `tests/test_v1_0_freeze.py` | asserts every v1.0 file is byte-identical to `18e410c` (fixture corpus pinned by digest) | the freeze, made structural |
| `reports/` | Gate 0 report | operational |
| `suites/`, `contamination/`, `generators/`, `scorecards/` | placeholders | empty |

## Running the tests

`benchmarks/` is **not** in `pytest.ini` `testpaths`, so the default suite does not
collect it. Run it explicitly:

```bash
pip install -r ../requirements-bench-test.txt   # tests only; the modules are stdlib-only
pytest benchmarks/ -q
```

CI runs exactly that (`.github/workflows/ci.yml`, step *Benchmark contracts*). Before
that step existed these tests ran in no job at all, including the Program 0 Gate 0
property tests that guard identity, reproducibility, and Layer-1 boundary isolation.

Two pytest configs disagree about this directory and it is worth knowing which wins:
`pyproject.toml` lists `benchmarks/adapters/lm_eval/tests` in its `testpaths`, but
`pytest.ini` exists and therefore takes precedence, and it does not. pytest reports
this as `WARNING: ignoring pytest config in pyproject.toml`. Reconciling the two is
left alone deliberately — the narrowing in `pytest.ini` looks intentional, and CI now
covers the gap either way.

## The two agentic instruments

Both were added as instrument definitions only. Neither registers a suite, authorizes
execution, emits a record, or promotes a metric to a gate. Design rationale, the
tier structure they belong to, and what remains owner-gated:
`papers/SWE_CAPABILITY_EVALUATION_PROPOSAL.md`.

**Run identity** (`agent_identity.py`). `adapters/lm_eval/schemas.py::compute_run_id`
binds four inputs, which cannot distinguish two runs of one checkpoint under
different scaffolds, tool permissions, or budgets. `compute_agent_run_id` is
**strictly additive** — `compute_run_id` is untouched, since its output is the on-disk
directory name for every existing run — and binds all eight components the evaluated
unit actually consists of. Absent, malformed, and placeholder pins raise rather than
produce a well-formed id.

**Process metrics** (`process_metrics.py`). Class A only: everything derivable from an
action log with no gold labels and no annotation. Two conventions matter when reading
results. Every rate returns **`None` rather than `0.0`** when its denominator is zero —
a run that mutated nothing has no premature-edit rate, and collapsing that to zero
would let an agent that did nothing outscore one that did the work. And the reported
artifact is a **profile vector**; `class_a_profile` deliberately does not produce a
scalar.

Each metric documents its false-positive and false-negative modes, and each documented
mode has a test in `tests/test_process_metrics.py` that makes it reproducible — these
metrics are **diagnostics, not gates**, and no threshold has been promoted.

**Gold metrics** (`gold_metrics.py`). Class B: the same action log plus the reference
patch and fail-to-pass / pass-to-pass test sets that issue-resolution tasks already
ship. No new annotation, and no corpus needed to define the instrument — the gold
artifacts are function parameters (`GoldTask`, `RunOutcome`), not a bundled dataset.
Test results come from the grader as a separate input, so an agent's own claim about
its tests can never become its score; a missing result counts as failing, never as a
pass. `aggregate_rate` returns `(successes, n)` rather than a float, because a safety
gate needs both terms for a Wilson bound and `0.0` over a dozen runs invites the
reading that bound exists to prevent.

**Confidence bounds** (`confidence.py`). The bound those two terms feed.
`wilson_upper(0, 12) ≈ 0.24` — twelve clean runs are consistent with a true
violation rate near one in four, which is why a gate on an observed count is
unfalsifiable at small n. `failure_gate_passes` returns `None`, never `True`, when
the sample is too small to decide: undecided is not passed. `runs_needed_for_zero_observed`
answers the planning question before a gate is proposed rather than after twelve runs.

The two schemas are structurally independent (no cross-`$ref`, no required
back-pointer) so they version separately: identity says which run this is, the action
log says what happened inside it.

## The emitter and its fixtures

`transcript_emitter.py` turns a Claude Code session transcript into a schema-valid action
log. Every action carries a **classification trace** — the rule that fired, its evidence,
whether the Bash call was compound, and notes — because `action_class` is declared at
emission and cannot be verified from the log; the trace makes it auditable instead of
trusted. Bash is classified by an explicit precedence (test runner → verify; file write →
mutate; all read-only → read; git index → execute; else execute) that the module docstring
defends and a 100-row test table pins, including the shapes real sessions use: multi-line
scripts, heredocs, control flow, wrappers, arithmetic.

It is developed **only against `fixtures/transcripts/`**, sanitized by
`transcript_sanitizer.py` (see the fixture README for what is preserved, what is
destroyed, and the raw-vs-sanitized invariant that makes the fixtures trustworthy).
`validation/instrument_validation_transcripts.py` is the one component that reads real
transcripts; it labels every output **instrument validation, not model evidence**, and
writes to the artifacts directory, not here.

Validated on 62 sessions / 1899 actions and audited by hand (180 classifications read).
The audit found three contract-level limitations and produced a v1.1 specification
(`call_id` sub-actions, `pre_state_sha256`, scoped `declared_scope`) that is deliberately
not implemented here: `artifacts/instrument-validation-transcripts/audit-and-contract-decision.md`.

## Contract v1.1 (`v1_1/`)

v1.0 is frozen permanently (owner decision, 2026-09-13). v1.1 lives in its own
package and imports v1.0's frozen segment-level machinery rather than copying it,
so a single command segment classifies identically under both contracts by
construction. Three changes, landed as three stages with their own fixtures and
tests (C2 → C3 → C1), decided in
`artifacts/instrument-validation-transcripts/v1.1-migration-spec.md`:

- **C2 `pre_state_sha256`** on mutations, from the transcript's `originalFile`;
  null on Write-create, non-ok results, Bash, and multi-target mutations, with no
  sentinel. `reversion_count` seeds history with it, so returning a file to its
  initial content is visible; no-op mutations are excluded and counted.
- **C3 `declared_scope` / `source_scope`**: harness declarations written verbatim,
  never inferred. Additive `*_in_scope` metrics; the v1.0 keys keep v1.0 meaning.
  A report written to `artifacts/` after a real verify no longer reads as
  "shipped unverified".
- **C1 `call_id`**: one Bash call decomposes into its effects in command order,
  consecutive same-class effects merged, all four classes participating. Siblings
  share the call's args, result and status; tokens and time sit on the first.
  Behavioural rates use per-call denominators; same-call siblings never count as
  repeats. The 29-of-45 hidden edits of the v1.0 audit become their own `mutate`
  sub-actions.

`v1_1/migration.py` emits both contracts side by side and must **explain every
changed value mechanically** — each explanation category has a predicate; a
delta that fails its predicate is `unexplained` and fails the gate. On the
checked-in fixtures the gate passes with zero unexplained deltas
(`artifacts/.../v1.1-migration/fixture-delta.md`). `v1_1/migration_rerun.py` is
the single authorized rerun over real sessions, labelled migration evidence.

Three spec statements were corrected by the side-by-side runs and are recorded in
the tests rather than smoothed over: verify *presence* is per call (the sub-action
count can exceed it when two test segments are separated by another effect);
`scope_violation_rate` does change on compound fixtures because split writes add
mutation-target pairs; and v1.0 mutate targets are not a subset of v1.1's, because
v1.0 pooled heredoc and inline evidence per call and over-attributed read-only
literals as mutate targets (a v1.0 defect, documented and reproduced as fixture
F15, not patched — v1.0 is frozen). The authorized 62-transcript rerun passed the
gate with zero unexplained deltas: `artifacts/.../v1.1-migration/`.
