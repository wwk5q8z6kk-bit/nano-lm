# Frontier — `accelerated-research-campaign-v2`

**Not canonical.** Authority: [`docs/ACTIVE_NOW.md`](../docs/ACTIVE_NOW.md) · [`docs/EXECUTION_PLAN.md`](../docs/EXECUTION_PLAN.md)

## Branch intent

Evolve P1 from **foundation PRs on master** (#37–#41) into a **paid multi-track campaign** with:

- production tool-calling stack (`structured` + `tool` inference paths)
- agent platform (`agent_canary`, sandboxed `coding_tools`)
- Native Nano extended training + evidence-bottleneck variants
- student C1/C2 vLLM serverless fan-out

## This branch adds (beyond `c4822b9`)

| Area | Paths |
|------|-------|
| Tool calling | `nanoscribe/tool_calling.py`, `tool_inference.py`, `tools.py` |
| Qwen3 coder adapter | `nanoscribe/inference/qwen3_coder.py` |
| Agent canary | `nanoscribe/agent_canary.py`, `scripts/agent_canary_bench.py` |
| Campaign artifacts | `artifacts/campaign/agent_canary_v1_results.json`, native checkpoints |
| Docs | `docs/subsystems/NANOSCRIBE.md`, `docs/research/ACCELERATED_CAMPAIGN.md` |

## Campaign configs (branch-local)

- `frontier/accelerated_research_campaign_v1.json` — track budgets + surfaces
- `frontier/p1_serverless_campaign_v0.json` — serverless wave
- `frontier/p1_three_track_baseline_v0.json` — harness baseline

## Before paid compute

1. `python3 scripts/campaign_control_plane.py inventory`
2. Confirm manifest under `artifacts/campaign/manifests/`
3. Read [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md)

## Merge posture

Selective port to `master` after:

- harness green on structured + tool paths
- agent canary bench reproducible
- campaign artifact summaries committed (not raw logs)
- no Evidence Core diff

## Quick verify

```bash
python3 scripts/check_active_now.py
python3 -m pytest nanoscribe/test_tool_calling.py nanoscribe/test_agent_platform_smoke.py -q
```
