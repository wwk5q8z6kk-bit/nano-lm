# What dropping moved values costs individual facts: results

Run 2026-11-23 under [PROTOCOL-MEMRULE-FACTSPLIT.md](PROTOCOL-MEMRULE-FACTSPLIT.md) (`daf9453`, with an addendum `8d24559` written before any result was read). Scored by `memory_rule_factsplit_score.py` (`0b64b71`, committed before any result was read). Seeds 4101–4105. Raw files and the scorer's verdict: `results/memory_rule_factsplit/`. [Observed]

Every rerun reproduced its original registered `late` metrics exactly, so no seed was excluded.

## End-of-run retention, means over five seeds

| Arm | Regular facts | Exceptions | Corrected facts |
|---|---|---|---|
| Re-cluster, full re-index (exact store replayed) | 0.997 | 0.636 | 0.604 |
| Re-cluster, drop moved values | 0.898 | 0.048 | 0.041 |
| Re-cluster, transport own predictions | 0.891 | 0.069 | 0.040 |
| Given types, no re-index (descriptive) | 0.919 | 0.022 | 0.035 |

About 6,400 regular, 660 exception and 580 corrected facts per seed.

## Against the registered predictions

- F1 (drop keeps at least 0.20 fewer exceptions than full re-index): **held**, five of five (0.040–0.059 against 0.616–0.660).
- F2 (drop within 0.10 of full re-index on regular facts): **held**, four of five (gaps 0.091–0.111).
- F3 (drop keeps no more corrected facts than full re-index): **held**, five of five.

## Reading

Dropping moved values costs individual facts heavily against a full re-index: exceptions 0.05 against 0.64, corrected facts 0.04 against 0.60. But the cost is not caused by dropping. A memory that is never re-clustered and is given the true types keeps exceptions no better (0.02), and corrected facts no better (0.04). In every arm without replay, the shared typed slots hold the type's rule, and individual facts that break it are not retained, whether or not anything moved. Only replaying an exact store keeps them, and that is storage, not the memory's own retention.

This completes the transport result (`RESULTS-MEMRULE-TRANSPORT.md`). Dropping moved values loses nothing that a single shared-path memory would otherwise have kept. The exceptions it never keeps belong in a private path, as the earlier private-slot and routing results showed, not in a better transport.

## Limits

- End-of-run snapshot only.
- The development seed's split was seen by this session before the protocol was committed by a duplicate copy; the addendum records it.
- Single shared path; synthetic world; value table. Not evidence about a language model.
- All twenty reruns finished on the first attempt.
