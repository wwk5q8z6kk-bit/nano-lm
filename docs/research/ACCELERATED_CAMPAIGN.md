# Accelerated research campaign

P1 scribing research at **intelligence-per-dollar** velocity: parallel tracks, manifest-gated paid compute, artifact contracts.

**Operational authority (paid waves):** [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md) — supersedes ad-hoc launch prompts.

**Branch-active detail (v2):** [ACCELERATED_RESEARCH_CAMPAIGN_V2.md](../ACCELERATED_RESEARCH_CAMPAIGN_V2.md) — measured snapshot + integration posture on `frontier/accelerated-research-campaign-v2`.

**Program authority:** [ACTIVE_NOW.md](../ACTIVE_NOW.md) · [EXECUTION_PLAN.md](../EXECUTION_PLAN.md)

---

## Mental model

```text
question / hypothesis
        ↓
experiment manifest (git SHA, dataset revision, surface, budget cap)
        ↓
wallet gate (runpodctl user · campaign_remaining)
        ↓
RunPod research OS surface (API → Hub Serverless → Flash → Template → Pod)
        ↓
artifacts/campaign/*  (checkpoints, metrics, summaries)
        ↓
harness recompute → failure-to-architecture loop
```

**No manifest → no paid compute.** Queue empty → scale to zero / delete ephemeral endpoints.

---

## Parallel tracks (v1 schema)

Manifest: `frontier/accelerated_research_campaign_v1.json`

| Track | Role | Typical surface | Budget cap (v1) |
|-------|------|-----------------|-----------------|
| **A — Frontier teacher** | Capability ceiling | Managed API / Kimi canary | $25 |
| **B — Strong control** | Specialist baseline | RunPod Serverless (Qwen3.8-27B) | $15 |
| **C — Large student** | Distillation target | Raw pod / Axolotl Hub | $50 |
| **D — Native Nano** | Scratch architecture screen | Flash / raw pod | $60 |
| **E — Verifier** | Independent semantic check | Cheapest GPU | $15 |

Tracks are **not** sequential gates — they run in parallel within envelope; synthesis happens at artifact + harness layer.

### Track semantics

- **Frontier teacher** — measures what is possible on hard atoms; not default deploy.
- **Strong control** — Qwen3.8 Serverless structured/tool inference; primary regression surface.
- **Large student** — QLoRA/SFT on disagreement/disjoint train sets (`p1_distill_train_v1`); not `p1_screening_eval_v1` (frozen forever).
- **Native Nano** — evidence-head variants at 30M–100M; trains on official PyTorch / autoresearch templates, not Axolotl.
- **Verifier** — independent check; skip learned verifier when deterministic baseline ≥ ~0.95.

---

## Execution surfaces (preference order)

```text
1. Public / managed endpoint
2. Hub Serverless (vLLM, SGLang)
3. Flash
4. Pod Template
5. Raw Pod (last resort)
6. Cluster (only when authorized)
```

Details: campaign autonomous execution doc §0.

---

## Key artifacts

| Path | Purpose |
|------|---------|
| `artifacts/campaign/experiment_manifest.v1.schema.json` | Required before paid job |
| `artifacts/campaign/spend.json` | Ledger + wallet reconciliation |
| `artifacts/campaign/campaign_status.json` | Live inventory snapshot |
| `artifacts/campaign/manifests/*.json` | Per-experiment manifests |
| `artifacts/campaign/*_summary.json` | Round summaries (native, student, etc.) |
| `frontier/*.json` | Branch-local campaign + baseline configs |

---

## Control plane (no spend)

```bash
python3 scripts/campaign_control_plane.py inventory
runpodctl user    # live wallet — physical ceiling
```

---

## Fan-out and inference modes

```bash
# Structured (default) + tool calling
python3 scripts/campaign_fanout.py orchestrate --modes structured,tool --endpoint YOUR_ENDPOINT

python3 scripts/tool_call_harness.py
python3 -m pytest nanoscribe/test_tool_calling.py -q
```

See [infrastructure/TOOL_CALLING.md](../infrastructure/TOOL_CALLING.md).

---

## Agent platform (campaign v2)

Sandboxed coding-agent tools for autonomous campaign scripts:

- `nanoscribe/coding_tools.py` — `read_file`, `list_directory`, `search_code`, `apply_patch` (dry-run), `run_command` (allowlisted)
- `nanoscribe/agent_canary.py` — agent canary bench path
- `scripts/agent_canary_bench.py` — campaign automation entry

Paths confined to sandbox root; no unrestricted shell.

---

## Integration status

| Component | State | Location |
|-----------|-------|----------|
| Encounter Representation v0 | **integrated** | `nanoscribe/encounter.py` (master #37) |
| Evidence transport eval | **integrated** | `nanoscribe/test_evidence_transport.py` (#38) |
| Model adapter + baseline bridge | **integrated** | `nanoscribe/adapt.py`, `adapters.py` (#40) |
| Qwen inference + harness | **integrated** | `nanoscribe/tracks.py`, serverless (#41) |
| Tool calling + agent stack | **integrated** (this branch) | `nanoscribe/tool_*.py`, `agent_canary.py` |
| H6 / `nano_ai/` span bundles | **cross-branch** | not in this tree |

Base: `origin/master` @ `c4822b9` (see [ACTIVE_NOW.json](../ACTIVE_NOW.json) `integration_base_sha`).

---

## What this is not

- Not a license to skip prereg on **confirmatory evidential** runs
- Not PHI / private owner data on cloud
- Not “bigger checkpoint = better Nano” — optimize **faithful representation per dollar**
- Not reopening Evidence Core / ledger in campaign PRs
