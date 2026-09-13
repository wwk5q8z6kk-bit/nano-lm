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
| `validation/` | runs the emitter over real sessions as a labelled instrument validation | the only reader of `~/.claude/projects/` |
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
