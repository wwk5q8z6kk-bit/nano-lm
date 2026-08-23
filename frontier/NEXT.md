# Frontier evolution — accelerated research campaign v2

**Branch:** `frontier/accelerated-research-campaign-v2`  
**Authority for paid compute:** [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md)  
**Canonical program state:** [`docs/ACTIVE_NOW.md`](../docs/ACTIVE_NOW.md) (not superseded by this file)

**Updated:** 2026-08-23

---

## Campaign phase snapshot (COMPLETE — idle, $0/hr)

| Lane | Status | Key artifact |
|------|--------|--------------|
| Managed reference (Qwen3-32B-AWQ) | **WINNER** C2 coverage 78.7%, assertion 100% | historical wave v2 |
| Student A (Qwen2.5-32B-Instruct) | C1C2 complete; semantic gap documented | `artifacts/campaign/student_gap_v1.json` |
| Native round 1 (30M × 8) | COMPLETE_RANKED | `artifacts/campaign/native_round1_summary.json` |
| Native round 2 (100M × 4) | COMPLETE — winner `evidence_bottleneck` | `artifacts/campaign/native_round2_summary.json` |
| Native extended (200 steps) | COMPLETE_WITH_WEIGHTS — loss 0.049 | `artifacts/campaign/native_extended_summary.json` |
| Agent canary (GPT-OSS-120B) | **PARSE FIX** — 48/48 parsed, outcome 0.50 | `artifacts/campaign/agent_canary_v1_results.json` |
| Agent platform smoke | OFFLINE PASS (3/3 contract cases) | `nanoscribe/test_agent_platform_smoke.py` |
| Verifier hard set | SKIP learned train (baseline 1.0 @ 500) | `artifacts/campaign/verifier_dataset.json` |

**Wallet posture:** `NO_ACTIVE_PODS` · ~$154 live · ~$149 campaign_remaining (floor $10)

---

## What we learned (evidence, not narrative)

1. **Managed Qwen3-32B-AWQ** remains the operational P1 structured baseline on C2.
2. **Student Qwen2.5-32B** closes C1 but lags on C2 assertion — QLoRA gate unlocked, not yet run.
3. **Native evidence_bottleneck** trains (loss 32 → 0.05 over 200 steps) but **hash-LM decode is not P1-viable** — smoke eval 0% coverage, 8/8 malformed spans.
4. **Agent canary** failed silently on empty parse before content-action extraction fix; full 48-task probe now **parse_rate=1.0**, outcome_mean=0.50.
5. **RunPod hygiene** — weight pull-before-terminate, CUDA preflight, US-KS-2 > EU-RO-1 for A100 SSH stability.

---

## Evolution gate (next engineering, not more idle GPU)

Priority order for P1 integration frontier:

```text
1. ENCOUNTER REPRESENTATION v0  (docs/EXECUTION_PLAN B1)
        ↓
2. SPAN/EVIDENCE TRANSPORT on managed student path
   (bridge student structured decode — not native hash-LM yet)
        ↓
3. VERIFIED RECORD → NOTE RENDERING  (B3)
        ↓
4. OPTIONAL parallel tracks (experiment-scoped):
   - QLoRA canary on student assertion gap
   - span_port native extended comparison (when A100 stock)
   - Qwen3-Coder tool-path agent canary (teacher collective)
```

**Explicitly deferred:** native 300M promotion, hybrid native+student, learned verifier — until structured P1 decode works on at least one path.

---

## Operator next commands (no spend unless manifest)

```bash
# Integrity
python3 scripts/check_active_now.py
python3 -m pytest nanoscribe/test_agent_platform_smoke.py nanoscribe/test_agent_canary.py -q

# Control plane refresh
python3 scripts/campaign_control_plane.py inventory

# Re-run agent canary (managed endpoint, manifest required)
python3 scripts/agent_canary_bench.py --probe-tasks 48

# Native smoke eval (local weights if present; CPU ok)
python3 scripts/train_native_nano.py --cpu-smoke
```

---

## Checkpoint chain

| File | When |
|------|------|
| `checkpoint_v0.json` | Campaign bootstrap |
| `checkpoint_v1.json` | Wave 1 managed + student |
| `checkpoint_v3.json` | Native round 2 + extended launch |
| `checkpoint_v4.json` | **This evolution** — all lanes idle, parse fix landed |
