# P1 evidence comparison

[REPORT.md](REPORT.md) reconciles preserved compact-model results, six recovered
OpenResearch prompt-clause arms, the corrected larger-model C2 comparison, and
terminal H6 evidence. It keeps each experiment's
task and denominator visible. No single score ranks these studies.

`sources/` contains exact Git blobs copied from full commits recorded in
[manifest.json](manifest.json). These are imported evidence and scoped source
interpretations, not executable model integrations. Historical plans and
authority statements inside preserved documents do not supersede current
project authority. Branch-local P3/P4 labels are study stages, not completion of
the program's P3/P4 capability milestones.

Verify and read from any checkout of this repository:

```bash
python3 scripts/reconcile_p1_comparison.py --format markdown
python3 scripts/reconcile_p1_comparison.py --verify-git --format markdown
python3 -m pytest -q scripts/test_reconcile_p1_comparison.py
```

The default command checks every copy's size, SHA-256 and Git blob hash without
requiring old branches to remain checked out or fetching any remote. Add
`--verify-git` to compare copies with the original local commit objects. If a
copy is missing and those objects exist, `--restore-sources` restores only
missing files; an existing mismatched file is an error, never overwritten.

The OpenResearch evidence comes from `work/p1-triaxial-session` commit
`2e29d396cdf8d34812bc3fb93c7c47e9675e1781`, with each prereg preserved from its
earlier source commit. Six exact slot exports include the synthetic prompts,
raw responses, parsed quotes and joint outcomes. The report uses only fresh
instances i12–i35 (384 slots); it excludes i0–i11 from inference. It reconciles
the historical paired endpoints from those stored classifications without
invoking the model or evaluator. Slots repeat within synthetic instances, so
historical slot-level p-values do not establish independent clinical evidence.

`manifest.json` also contains a sanitized retrieval audit: run IDs, source
commits, log sizes/hashes, prompt hashes, clause flags and exact agreement with
all 576 exported slots per arm. Raw CLI logs and account metadata are not
included. If the original local `<run-id>.log` files are available, repeat the
capture audit without contacting OpenResearch:

```bash
python3 scripts/reconcile_p1_comparison.py --orx-log-dir /absolute/path/to/run-logs --format markdown
```

The source `dataset_revision` string predates the 36-instance expansion; use
the pinned commits and prompt hashes to identify the recovered cohort.
Historical CONFIRMED/NOT CONFIRMED/MIGRATION labels retain their original
prereg scope. They are not program-wide verdicts or an H7 result.

The committed report is the no-H7-result snapshot. Regenerate it with:

```bash
python3 scripts/reconcile_p1_comparison.py --format markdown > artifacts/p1_comparison/REPORT.md
```

Once the separately executed, authorized H7 experiment produces a complete
result, compare that stored result without another evaluation pass:

```bash
python3 scripts/reconcile_p1_comparison.py \
  --h7-result /absolute/path/to/h7-result.json \
  --h7-sha256 EXPECTED_SHA256_FROM_RESULT_CAPTURE \
  --format markdown
```

Supply the hash recorded when capturing the result. This verifies those bytes,
the frozen screen identity, two treatment seeds, termination contract and
stored decision consistency. It does not authenticate checkpoint/training
report bytes or replay model outputs. H7 remains a separate comparison group
because its fresh five-field screen differs from the compact two-field denial
bundle and C2 instrument. A setup log, incomplete training run or PASS on a
different screen cannot populate the completed H7 slot.

The reconciliation command imports no model packages, does not open screen
records, makes no provider calls and never runs an experiment.
