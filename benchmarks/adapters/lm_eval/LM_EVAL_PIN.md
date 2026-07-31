# lm-evaluation-harness pin (Program 0)

| Field | Value |
|-------|-------|
| Package | `lm-eval==0.4.12` |
| Source commit | `6d642546f4688648fced259eb3302efd36ece5af` |
| Repository | https://github.com/EleutherAI/lm-evaluation-harness |
| Release tag | `v0.4.12` |
| Python | `>=3.10` (repo requires-python) |
| License | MIT (upstream) |
| Dependency scope | **optional / benchmark-only** via `requirements-bench.txt` |
| Default runtime / CI | **not** installed until smoke green and owner opts in |

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

**Warning (upstream):** task-version changes can make results incomparable even under the same package version. Program 0 therefore pins package version **and** source commit **and** task YAML SHA-256 **and** instrument SHA-256.
