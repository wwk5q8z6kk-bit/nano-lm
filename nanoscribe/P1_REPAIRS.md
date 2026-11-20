# P1 medication and field-preservation repairs

This records the earlier v2 implementation and its original verification.
The subsequent v3 scope and parse-recovery repair is documented separately in
[the scope repair record](../artifacts/p1_scope_repair_dev_20261023/REPORT.md).

Date: 2026-10-23. Development changes on `codex/p1-h7-scribing-path`, based on
`61a06524c9d1a9998f64c90d1035831b575713eb`. No new model-quality evaluation,
campaign training, paid compute, private data, frozen-screen rerun or clinical
evaluation was performed.

## Result

The opt-in context path now retains a supported complaint when a complete,
recognized medication reply is uncertain or its model proposal is wrong.
For example, `I've been having wrist soreness.` remains a reviewable complaint
alongside `I cannot remember what I took.` as uncertain medication evidence.
The original transcript and exact character offsets remain available.

Medication use and an explicit blanket denial establish a conflict regardless
of which side the model selected. All observations survive, including a third
observation omitted from a two-span model proposal. Two different medicines or
allergens can coexist; these remain uncertain under the scalar schema rather
than becoming an invented conflict. Repeated compatible observations retain
their evidence without becoming conflicts.

Field-local handling requires a recognized question and complete bounded reply.
Unknown replies, orphan turns, third-party/history context, cross-field
corrections and empty role boundaries still quarantine asserted fields across
the note. A model's label or omitted evidence cannot establish safe scope. The
checker also scans the original source when a custom solver omits a conflicting
or uncertain observation.

The constrained decoder offers medication/allergy support only from recognized
positive replies, absence only from complete explicit denials, and conflict
pairs only from incompatible observations. Role attribution accepts the same
case/spacing variants as the source checker and preserves intervening unknown
or empty boundaries. The prompt now distinguishes coexisting names from conflict.

Raw proposals remain separate from repaired output in `inference_diagnostics`.
A source-preserved conflict does not mark an incorrect raw proposal verified.
A proposed conflict pair must itself establish conflict; a compatible pair is
not credited just because a third, omitted observation conflicts with it.
For aggregate uncertainty, a complete reply quoting one of the bound same-field
observations is accepted consistently with the decoder's whole-reply `U` syntax.
Partial or wrong-field replies cannot gain verification through that mapping.

Version identities:

- Context policy: `context-source-scope-v2`, appended to context solver versions.
- Diagnostics: `nanoscribe.context-state-diagnostics.v2`.
- Constrained decoder: `typed-context-source-trie-v6`.

The default legacy adapter remains separate. Its frozen implementation is
unchanged; the shared transcript reader now preserves empty role boundaries.
Malformed raw summaries still fail closed with no draft. The binder, decoder
and checker share bounded recognition rules, so they do not constitute
independent semantic verification or establish model quality.

## Separate development cases and historical preservation

[test_p1_repairs.py](test_p1_repairs.py) adds 40 fresh synthetic regression cases
with deliberately injected raw proposals. [test_constrained.py](test_constrained.py)
adds 35 cases for admissible state/span choices and role boundaries. These are
software development fixtures, not held-out medical evaluation or independent
human annotations. They do not load the frozen pilot or experiment inputs.

Historical pilot run-08, run-09 and challenge-01 are replayed using their
hash-authenticated source snapshots in isolated subprocesses. Tests retain exact
saved-draft equality, legacy/raw/checked quality assertions and the challenge's
recorded 5/6 outcome. Snapshots pin only the recorded modules; unpinned
dependencies still resolve from the repository. No archived result is recomputed
through the new policy and relabelled as historical performance.

A before/after SHA-256 comparison verified all **699 tracked files** under
`artifacts`, `nano_ai`, `fabric`, `trajectory`, `papers`, `ledger` and
`docs/evidence` unchanged. The manifest SHA-256 was
`866aa4db65827e983266bbb7ce1936677ff70b8197c312a4f68b62c27e7f1276`.
Frozen adapters, H6/H7 evidence, recorded pilots and the existing external
checkpoint symlink were preserved. No tags were created, moved or pushed.

## Verification

Environment: Python 3.14.7, pytest 9.0.2. Runs used
`PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
Pytest used `-q -p no:cacheprovider -o addopts=''`. Exact invocations after the
environment prefix were:

```sh
python3 scripts/check_active_now.py
python3 scripts/check_docs_integrity.py
python3 fabric/test_fabric.py
python3 trajectory/test_recompute_c3.py
python3 -m pytest -q -p no:cacheprovider -o addopts='' nanoscribe
python3 nanoscribe/benchmark.py
python3 -m pytest -q -p no:cacheprovider -o addopts='' nano
python3 -m pytest -q -p no:cacheprovider -o addopts='' nano_ai/tests
python3 -m pytest -q -p no:cacheprovider -o addopts='' benchmarks/
python3 -m pytest -q -p no:cacheprovider -o addopts='' fabric trajectory wedge_v1
python3 -m pytest -q -p no:cacheprovider -o addopts='' scripts/test_reconcile_p1_comparison.py scripts/test_run_h7_kaggle.py scripts/test_run_p1_pilot.py
git diff --check
```

| Check | Result |
|---|---|
| `python3 scripts/check_active_now.py` | Pass |
| `python3 scripts/check_docs_integrity.py` | Pass |
| `python3 fabric/test_fabric.py` | 8/8 pass |
| `python3 trajectory/test_recompute_c3.py` | 7/7 pass |
| `python3 -m pytest … nanoscribe` | 252 passed |
| `python3 nanoscribe/benchmark.py` | Pass; four exact spans, zero critical errors, zero unnecessary abstentions |
| `python3 -m pytest … nano` | 316 passed |
| `python3 -m pytest … nano_ai/tests` | 499 passed, 1 skipped |
| `python3 -m pytest … benchmarks/` | 500 passed, 1 skipped |
| `python3 -m pytest … fabric trajectory wedge_v1` | 32 passed |
| `python3 -m pytest … scripts/test_reconcile_p1_comparison.py scripts/test_run_h7_kaggle.py scripts/test_run_p1_pilot.py` | 90 passed |
| `git diff --check` | Pass |

Total pytest results: **1,689 passed, 2 skipped**. Direct fabric/trajectory
checks overlap with pytest and are not added again. The two environment skips
are the unavailable installed Nano RunPod guard and optional `lm_eval` package.
The NanoScribe benchmark is a software simulation; it loads no model weights.

The [pre-change baseline](../docs/discovery/evidence-register.md#observed-execution)
recorded 1,614 passed and the same two skips. This change adds 75 regression
cases, with no remaining test failures. The archived model scores are unchanged;
the extra passing software tests are not a model-performance improvement claim.

Independent patch review found and prompted repairs for a falsely credited
compatible conflict pair, erased empty-role boundaries, inconsistent decoder
role parsing and a whole-reply uncertainty verification mismatch. Focused
regression checks now pass for all four.

Remaining work is separate model development validation, independently reviewed
annotations and broader dialogue transfer. The bounded grammar can still abstain
on unfamiliar language. These repairs do not change H6/H7 verdicts or establish
P1 mastery, clinical completeness or deployment readiness.

## Subsequent development and fresh validation — 2026-10-26

The historical results above remain unchanged. See the [latest bounded retention
repair](../artifacts/p1_retention_repair_dev_20261026/REPORT.md) for complete-clause
observations, question coverage, medication/allergy guards, independent review
and the full 1,940-pass verification record. The [fresh independently annotated
validation](../artifacts/p1_retention_validation_20261026/REPORT.md) remains a
failed transfer: free/constrained abstentions are 24/24 and 23/24, and neither
mode retains an exact-correct assertion. All 1,713 prior protected hashes match.
