# Wedge v1 — supporting evidence/validation runtime and test bed

Wedge v1 is a supporting evidence and validation runtime/test bed for Nano. It
answers questions over a local document folder using deterministic retrieval,
extraction, contradiction checks, evidence spans, and abstention. It is not the
Nano AI core or its scribe inference path, and the default path does not call an
external model.

## Quick start

```bash
python3 -m wedge_v1 smoke
python3 -m wedge_v1 ask --corpus wedge_v1/data/corpus "How long before cached entries expire?"
python3 -m wedge_v1 compare --corpus wedge_v1/data/corpus metformin
python3 -m wedge_v1 report verified --corpus wedge_v1/data/corpus "What is the cache TTL?"
python3 -m wedge_v1 adversarial
```

Supported input formats are Markdown, text, and PDF. PDF extraction uses the
optional `pypdf` dependency. Install it before a PDF study with:

```bash
python3 -m pip install -r wedge_v1/requirements-optional.txt
```

## Exact document scope

Document IDs are corpus-relative paths without the file extension. `--doc` is
repeatable on `ask`, `find`, `scan`, `compare`, and `report`.

```bash
python3 -m wedge_v1 ask --corpus /path/to/documents \
  --doc notes/cache "What is the cache TTL?"
python3 -m wedge_v1 compare --corpus /path/to/documents \
  --doc study_a --doc study_b metformin
```

If any requested ID is unknown, the operation returns `ABSTAIN` with zero claims
and reports `selected_doc_ids` and `missing_doc_ids`; it never falls back to the
whole corpus.

## Representative private-use study

The demo and example task pack are smoke tests only. A representative study
requires 10–50 readable documents and 10–20 genuine, unique questions, including
at least one repeat-recall task. Every question must name its exact document
scope so evaluation can never widen to the whole corpus.

Store the question pack in the ignored `wedge_v1/data/owner_tasks/` directory
or outside the repository. First inventory the corpus to discover its exact
document IDs:

```bash
python3 -m wedge_v1 study inventory \
  --corpus /path/to/documents \
  --out wedge_v1/.private/corpus-inventory.json
```

Both `--corpus` and `--out` are required. The output is a private JSON report
published atomically without overwriting an existing path; stdout is an
aggregate, content-free receipt. The report lists valid document IDs and
unsupported or unreadable entries without copying document content.

Create the task pack explicitly, then capture each genuine question without
placing private query text in shell arguments:

```bash
python3 -m wedge_v1 study init \
  --tasks wedge_v1/data/owner_tasks/questions-v1.json
python3 -m wedge_v1 study add \
  --tasks wedge_v1/data/owner_tasks/questions-v1.json \
  --id recurring-question-01 \
  --mode ask \
  --doc corpus/relative_document_id \
  --expect-status SUPPORTED \
  --expect-status CONTRADICTED \
  --manual-baseline-seconds 120 \
  --query-file /private/path/question.txt
```

After capture, rerun the inventory with optional `--tasks` to diagnose every
task's exact scope:

```bash
python3 -m wedge_v1 study inventory \
  --corpus /path/to/documents \
  --tasks wedge_v1/data/owner_tasks/questions-v1.json \
  --out wedge_v1/.private/task-scope-diagnostics.json
```

Scope diagnostics may include a non-authoritative proposal only when a task's
exact basename has one unique corpus match. After reviewing the private report,
apply a complete set of confirmed proposals to a new task-pack revision:

```bash
python3 -m wedge_v1 study repair-scopes \
  --corpus /path/to/documents \
  --tasks wedge_v1/data/owner_tasks/questions-v1.json \
  --inventory wedge_v1/.private/task-scope-diagnostics.json \
  --out wedge_v1/data/owner_tasks/questions-v2.json \
  --confirm-all-exact-basename-proposals
```

All five arguments are required. The command revalidates the report against the
current corpus and exact source-pack identity, and succeeds only when every
unknown scope reference still has one exact, case-sensitive, unique-basename
match. Confirmation is all-or-nothing: a stale, partial, ambiguous, or forged
proposal set writes no revision. The source pack is never changed. The new pack
preserves its schema, private storage class, queries, expected statuses, manual
baselines, and owner metadata while changing only confirmed document scopes. It
is published atomically as an owner-only (`0600`) file without overwriting an
existing path, and stdout remains aggregate-only and content-free.

Scope repair does not supply a missing baseline or establish readiness. Rerun
the inventory into another unused output before checking the new revision:

```bash
python3 -m wedge_v1 study inventory \
  --corpus /path/to/documents \
  --tasks wedge_v1/data/owner_tasks/questions-v2.json \
  --out wedge_v1/.private/task-scope-diagnostics-v2.json
```

The lifecycle below then begins with `study check`, which remains the authority
for corpus IDs, coverage, recall inclusion, measurement completeness, and
representative readiness.

Without `--query-file`, `study add` reads the query from standard input. `init`
never overwrites an existing path, and `add` validates the complete pack under
an inter-process lock before an atomic append. Its output means only
`CAPTURED`; corpus IDs, coverage, recall inclusion, and representative readiness
remain the authority of `study check`.

If publication succeeds but directory durability cannot be confirmed, the
command returns `INDETERMINATE`. Inspect the pack before retrying; do not assume
the task was rejected.

The canonical file is a JSON object with private storage metadata and a `tasks`
array:

```json
{
  "schema": "nano-lm.wedge_v1.study_tasks.v1",
  "storage_class": "OWNER_PRIVATE",
  "study_class": "REPRESENTATIVE_USE",
  "tasks": [
    {
      "id": "recurring-question-01",
      "mode": "ask",
      "query": "the genuine question",
      "doc_ids": ["corpus/relative_document_id"],
      "expect_status": ["SUPPORTED", "CONTRADICTED"],
      "manual_baseline_seconds": 120
    }
  ]
}
```

`mode` is `ask`, `find`, `compare`, or `recall`. A `compare` task must scope at
least two documents. A `recall` task counts as one of the 10–20 unique questions
and executes that same scoped question twice during `study run`: the first pass
must produce `REFRESHED` with one solver execution and the immediate second pass
must produce an audited `CACHE_HIT` with zero solver executions. Its saved state
stays inside the ignored study directory and is frozen with the other private
study artifacts.

`expect_status` records the acceptable runtime status or statuses; it is task
metadata, not the usefulness judgment. Use the time a manual answer would
normally take for `manual_baseline_seconds`.

Canonical packs default to `REPRESENTATIVE_USE` when `study_class` is omitted,
and every task then requires a positive manual baseline. A deliberately narrower
agent-applied run may instead declare
`"study_class": "AGENT_APPLIED_SCOPED_PILOT"`. That exact, identity-bound class
allows missing manual baselines as a warning while keeping every scope, status,
recall, audit, timing, and review gate. Its check can be `study_ready: true` but
is always `representative_ready: false`; review must use `agent_applied`, the
manual comparison remains absent, and no time-saved or representative-use claim
is supported. Unknown, malformed, or non-canonical class declarations fail
closed.

Do not append to a task pack that already backs a frozen study. Initialize a
new version instead, then use a new study directory; changing a source pack
changes its study identity and verification fails closed.

Then run the isolated lifecycle:

```bash
python3 -m wedge_v1 study check \
  --corpus /path/to/documents \
  --tasks wedge_v1/data/owner_tasks/questions-v2.json \
  --dir wedge_v1/.studies/first-use
python3 -m wedge_v1 study run \
  --corpus /path/to/documents \
  --tasks wedge_v1/data/owner_tasks/questions-v2.json \
  --dir wedge_v1/.studies/first-use
python3 -m wedge_v1 study review \
  --dir wedge_v1/.studies/first-use --reviewer owner
python3 -m wedge_v1 study summary --dir wedge_v1/.studies/first-use
```

`check` separates basic smoke readiness, study readiness, and representative
readiness. `run`
freezes corpus, task-pack, solver, card, saved-recall, and private raw-result
digests and writes no global CoE or owner artifact. Before review or summary,
verification rederives every task once with the identity-bound deterministic
solver over its exact live scope, with persistence disabled; it then re-audits
the frozen raw result and reproduces the reviewer card. For a recall task this
verification uses the underlying `ask` solver and is separate from the recorded
`REFRESHED` → `CACHE_HIT` transition and its timing.

A successful run reports `status: COMPLETE` and
`decision: REVIEW_REQUIRED`: this means the frozen run artifact is ready for
review, not that a final component decision has been made. `review` records the
outcome, structured failure class, suggested correction, reviewer kind, and
monotonic review time. `summary` contains only aggregate counts, timing,
formats, and digests—never queries, document IDs, spans, paths, or correction
text.

Use these review labels consistently:

- `USEFUL`: the supported output and evidence directly answer the task without
  a material correction.
- `PARTIALLY_USEFUL`: some content helps, but a material omission or repair is
  required.
- `NOT_USEFUL`: the output does not advance the task.
- `CORRECT_ABSTENTION`: scoped evidence is insufficient or conflicting and
  abstention is appropriate.
- `OVER_ABSTENTION`: sufficient scoped evidence existed for a useful answer.
- `WRONG_EVIDENCE`: the conclusion may be plausible, but a cited source or span
  does not support it.
- `RETRIEVAL_MISS`: relevant readable evidence existed in scope but was not
  retrieved.
- `INGESTION_FAILURE`: an in-scope source exists but could not be read or was
  extracted incorrectly.
- `CONTRADICTION_HANDLED`: the output surfaced the conflict without falsely
  resolving it.

For a failure label, enter a concrete reason and suggested correction:

```text
LABEL concrete reason | FAILURE_CLASS | suggested correction
```

For example: `WRONG_EVIDENCE cited span states a different value |
WRONG_EVIDENCE | cite the matching source sentence`. The explicit failure class
may be omitted when it is the same as the label, but the reason and correction
are required.

Each accepted judgment is saved immediately. Re-running `study review` resumes
at the next unlabeled task. To correct a mistaken judgment, clear it with an
audited undo and then resume the study review:

```bash
python3 -m wedge_v1 review \
  --state wedge_v1/.studies/first-use/review.json \
  --undo recurring-question-01 --reviewer owner
python3 -m wedge_v1 study review \
  --dir wedge_v1/.studies/first-use --reviewer owner
```

After review, `study summary` emits a final Wedge-development decision:
`INCOMPLETE`, `FIX_REPEATED_FAILURE`, or `NO_REPEATED_FAILURE`. These decisions
guide Wedge component work; they do not establish Nano AI capability,
generalization, or scientific or clinical claims. `NO_REPEATED_FAILURE`
means that the complete study
contained no failure class occurring at least twice; it does not authorize a
speculative fix, architecture change, or model experiment.

Study verification establishes agreement with the current exact inputs and
identity-bound deterministic solver. Because every local file is mutable, it is
not a historical-authenticity proof if the inputs, code, and artifacts are all
replaced coherently; that stronger claim requires an external protected anchor.

For the generic `review` command, private `--corpus`, `--tasks`, or
`--from-dogfood` inputs require an explicit `--state` path outside the
repository or under the ignored `wedge_v1/.private/` directory. Private owner,
contact, and gallery exports follow the same containment rule. `--summary` and
`--next` inspect frozen review state without rerunning the solver.

For a public smoke test:

```bash
python3 -m wedge_v1 owner-dogfood --demo \
  --out /tmp/nano-demo/results.json \
  --gallery /tmp/nano-demo/gallery.md \
  --gallery-json /tmp/nano-demo/gallery.json
```

## Verified saved recall

```bash
python3 -m wedge_v1 habit --corpus /path/to/documents \
  --doc notes/cache --save "What is the cache TTL?"
python3 -m wedge_v1 habit --corpus /path/to/documents --recall TASK_ID
```

`--save` returns a task ID. The first recall creates a verified snapshot; an
unchanged later recall returns `CACHE_HIT` only after a live evidence audit and
without another solver run. Task, exact scope, selected-source content, solver,
result, or audit changes force one refresh. A failed refresh returns safe
abstention and neither serves nor persists the stale answer. `--rerun` remains
the batch-status workflow.

All public claim surfaces, including verified reports, retain an auditable
claim/evidence envelope; invalid bindings are reduced to abstention before
presentation.

Generated Wedge evaluations are not Evidence Ledger claims unless they are
reviewed and promoted separately.
