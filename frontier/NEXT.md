# Frontier branch notes

Not canonical authority. See [`docs/ACTIVE_NOW.md`](../docs/ACTIVE_NOW.md).

## Active branch

`frontier/accelerated-research-campaign-v2`

## Campaign v2

- Plan: [`docs/ACCELERATED_RESEARCH_CAMPAIGN_V2.md`](../docs/ACCELERATED_RESEARCH_CAMPAIGN_V2.md)
- Manifest: [`accelerated_research_campaign_v2.json`](accelerated_research_campaign_v2.json)
- v1 checkpoint: [`artifacts/campaign/checkpoint_v4.json`](../artifacts/campaign/checkpoint_v4.json)
- Paid compute authority: [`artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md`](../artifacts/campaign/CAMPAIGN_AUTONOMOUS_EXECUTION.md)

## Shipped this session

- **wedge_v1 `ask --escalate-stub`** — opt-in hybrid stub after classical ABSTAIN (`WEDGE_ESCALATE_STUB=1`); default fail-closed unchanged
- **wedge_v1 `--doc` scope** — exact document filter on `ask` / `find` / `scan`; unknown/empty → `ABSTAIN` with `failure_codes`
- Tests: `wedge_v1/test_escalate_stub.py` + smoke pins

## Next bounded tasks (local-first)

1. **B2** — Span transport improvement + C2 re-run (RunPod serverless, routine budget) → `artifacts/campaign/span_transport_v2.json`
2. **B3 follow-up** — Claim decomposition on rendered notes (`nanoscribe/decompose.py` wiring to encounter v0)
3. **wedge_v1 port (selective)** — human review / habit loop from `frontier/active-v1` when owner wants product surface; do not wholesale merge `lm/` stack

## Done (campaign v2)

| ID | Task | Artifact |
|----|------|----------|
| B1 | encounter_representation_schema_v0 | `nanoscribe/schemas/encounter_v0.schema.json` |
| B3 | verified_record_to_note_rendering_v0 | `nanoscribe/render/encounter_note.py` |

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
