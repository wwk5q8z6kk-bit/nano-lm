# lm-evaluation-harness pin

| Field | Value |
|-------|-------|
| Package | `lm-eval==0.4.12` |
| Source commit | `6d642546f4688648fced259eb3302efd36ece5af` |
| Repository | https://github.com/EleutherAI/lm-evaluation-harness |
| Release tag | `v0.4.12` |
| Python | `>=3.10` (repo requires-python) |
| License | MIT (upstream) |
| Dependency scope | **optional / benchmark-only** via `requirements-bench.txt` |
| Default runtime / CI | Optional; only needed for external harness validation |

Install:

```bash
pip install -r requirements-bench.txt
```

Validate external task:

```bash
lm-eval validate \
  --tasks nano_held_value_sentinel \
  --include_path benchmarks/adapters/lm_eval/tasks
```

**Warning:** task-version changes can make results incomparable even under the
same package version. This adapter therefore pins the package version, source
commit, task YAML digest, and instrument digest.
