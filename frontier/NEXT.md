# Frontier branch notes

Not canonical authority. See [`docs/ACTIVE_NOW.md`](../docs/ACTIVE_NOW.md).

## Active branch

`frontier/accelerated-research-campaign-v2`

## Campaign v2

- Plan: [`docs/ACCELERATED_RESEARCH_CAMPAIGN_V2.md`](../docs/ACCELERATED_RESEARCH_CAMPAIGN_V2.md)
- Manifest: [`accelerated_research_campaign_v2.json`](accelerated_research_campaign_v2.json)
- v1 checkpoint: [`artifacts/campaign/checkpoint_v4.json`](../artifacts/campaign/checkpoint_v4.json)
- Paid compute authority: [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md)

## Next bounded tasks (local-first)

1. **B1** — encounter schema + roundtrip tests (done)
2. **Native30 revalidation** — `python3 scripts/run_native30_revalidation.py --analyze` (local MPS/CUDA; resume-safe; replaces Kaggle/RunPod for 30M screen)
3. **B2** — Span transport v2 local smoke done; C2 serverless re-run when paid compute authorized
4. **B3** — claims + note rendering v0 (done)

### B2 launch (not executed)

```bash
# Preflight: wallet + ACTIVE_NOW gate
runpodctl user
python3 scripts/check_active_now.py

# Deploy ephemeral vLLM endpoint, then:
export RUNPOD_SERVERLESS_ENDPOINT_ID=<live-endpoint-id>
python3 scripts/student_serverless_c1c2.py \
  --endpoint "$RUNPOD_SERVERLESS_ENDPOINT_ID" \
  --suites c2_screening \
  --mode tool \
  --model Qwen/Qwen2.5-32B-Instruct
```

Result artifact: `artifacts/campaign/span_transport_v2.json`

## Cross-worktree (selective port only)

Integration worktree: `/Users/mac/Projects/nano-lm-nanoscribe` (`frontier/nanoscribe-core-v1`). Port policy: [`p1_integration_manifest.json`](p1_integration_manifest.json). Do not wholesale merge `nano_ai/` or P4 CUAD training.

## v1 empirical anchors

| Metric | Value | Source |
|--------|-------|--------|
| Managed ref C2 exact gold span | 0.110 | `student_gap_v1.json` |
| Student A assertion gap vs ref | −0.320 | `student_gap_v1.json` |
| Native 100M extended smoke coverage | 4/7 (57%) | `native_smoke_eval_extended.json` |
| Native 100M extended smoke malformed | 3/8 | candidate-scored decode (was 8/8 autoregressive) |
| Verifier hard set accuracy | 1.0 @ n=500 | `checkpoint_v4.json` |
