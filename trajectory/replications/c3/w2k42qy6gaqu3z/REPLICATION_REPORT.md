# C-3 REPLICATION REPORT — pod w2k42qy6gaqu3z

**Kind:** post-primary venue-robustness replication (owner-directed). **It does not
alter the frozen primary decision rule** (primary result at commit `0359c22`); it is a
robustness check, reported without merging into the primary estimate.

## Provenance (frozen before results — see replication_manifest.json)

- primary result commit: `0359c22`; code commit at launch: `5234367`
- pool / eval / kernel / recompute / prereg SHAs: recorded in the manifest
- **identical** pool, eval, kernel, checkpoint, tokenizer, and (bug-fixed) scorer as the
  primary — the ONLY deliberate change is the hardware venue.
- **Venue:** primary RTX 4090 → **replication RTX A6000** (RunPod secure cloud), torch
  2.8.0+cu128, CUDA 12.8. 3 FT seeds (0,1,2). Pre-launch cap-safety assertion PASSED
  (max response 47 tok < max_new 64), so no evaluator-induced truncation.
- remote↔local artifact SHA-256: **match** (clean retrieval); pod deleted, `pod list` empty.
- cost ≈ **$0.30** (~35 min × $0.49/hr, estimate). Ceiling was $0.60; auto-relaunch off.

## Result — the frozen verdict REPRODUCES on different hardware

| contrast | primary | replication | Δ | verdict (both) |
|---|---|---|---|---|
| H-transition (T-avail − T-sep) | +1.7 | −2.5 | −4.2 | **REFUTED** |
| H-boundary (B-sub − B-space) | −8.3 | −7.5 | +0.8 | **REFUTED** |
| H-length (short − long) | +25.0 | +22.5 | −2.5 | **UNRESOLVED** |
| T-full control | 100% | 100% | — | valid (not void) |
| seed-unstable cell types | 12 | 14 | — | — |

All three mechanical verdicts **MATCH**. Every contrast delta is < 5 pts — within the
run-to-run noise implied by the 25–27% seed-instability, not a hardware effect. Only
**2 / 48** cell types changed majority-flip state across venues: `desloratadine`,
`ptomaine` — both boundary types (the same seed-fragility class C-1b documented, e.g.
ragweed's 0→100 venue flip). H-stochastic remains **not supported** in the replication
(H-length still UNRESOLVED, so its all-three-REFUTED AND-gate does not fire).

## Morphological re-inflection reproduces

Error-class distributions are close, and **morphological_near_miss is the dominant
non-hit class in BOTH runs** (primary 173, replication 166); truncation is identical
(47 = 47). This strengthens the morphological re-inflection observation as a real,
hardware-robust pattern — but it **remains POST-HOC EXPLORATORY** (n=4 stems in the
primary's corpus analysis, selected after seeing the error distribution). It is NOT
presented as a causal mechanism; the registered morphology probe (M = morphological
exposure × number agreement × transition availability × boundary type) is the correct
next step, and only after this reproduction.

| error class | primary | replication |
|---|---|---|
| morphological_near_miss | 173 | 166 |
| substitution | 84 | 75 |
| truncation | 47 | 47 |
| parse_failure | 38 | 60 |
| tail_only | 26 | 26 |
| omission | 8 | 27 |
| exact_copy | 14 | 6 |
| garble | 6 | 0 |

(parse_failure/omission/exact_copy shift modestly — consistent with seed/hardware
nondeterminism at the boundary; none changes a verdict.)

## Conclusion

The C-3 mechanical verdict (H-transition REFUTED, H-boundary REFUTED, H-length
UNRESOLVED; H-stochastic not supported) is **robust to hardware venue**. The
morphological re-inflection failure pattern reproduces. No implementation or provenance
defect was uncovered, so **the frozen primary result stands unchanged**. Cross-check:
`replication_comparison.json` (verdicts_reproduce: true).
