# Strict tournament receipt gate — 2026-11-26

**Scope:** additive, prospective verification only; the registered scorer, protocol, raw results, historical verdicts and post-audit receipt are unchanged. This wrapper is deliberately specific to the fixed 14M, three-seed, T4 tournament; do not use it as an unregistered protocol for 40M/42M experiments.

## Reconciliation

PR #203 advanced from `7d0e6ee0926f614c41ee5b6e6f5e797fb82cff15` to `43286582e2d916f3f4552dc351999e10b4832697` on Nov 25. Its new `audit_and_strict_score.py` blob `2d15b6b5b86f9b67e6aabc34c5d054396669fef5` checks completion, final steps/tokens, duplicate seeds and wall-clock significance. The historical `score_tournament.py` and `PROTOCOL.md` blobs remain unchanged. Both prior automated review threads are marked resolved. `results/tournament/strict_audit_receipt.json` blob `7f31e70395322ac16be6f939277bcc1ab3956cdf` retains historical D1 CLOSE_DEPTH_ROUTING_AT_14M and D2 REJECT_UNCONDITIONAL_CODA_REFINE; post-audit D1 qualifies value residual for a proposed 42M gate, not adoption and not a registered new result.

Remaining gap: the additive scorer accepts arbitrary nonempty seed sets, checks only the final curve entry, and rereads files between validation and scoring. The new `strict_receipt_gate.py` guards these boundaries before calling the existing additive scorer. It checks the exact seed/arm set, protocol-specific recipe/parameter/order identities, every evaluation step/token count, finite nonnegative recorded metrics, strictly increasing elapsed time, and reported zero AMP skips. It rejects ambiguous duplicate JSON keys, nonfinite constants, receipt hash mismatches and scorer-provenance mismatch. A private temporary byte snapshot binds scoring to the validated bytes. Strictly requiring these fields is a *new prospective review gate*, not a retrospective change to the experiment's registered decision rule. A gate failure demands review rather than rewriting the historical finding.

## Verified evidence

Independently inspected the three exact committed JSON reports via GitHub at the above PR #203 pin. All 15 arm-seed combinations have 6 valid evaluation rows (90 rows total). Exact seed and recipe, arm order, parameter counts, completed flags, reported zero AMP skips, token counts, finite metrics and monotonic time all passed an independent JS check. The original text bytes' SHA256 were recomputed in that inspection and match the source's post-audit manifest:

| Source (relative to this directory) | Git blob | Recomputed SHA256 |
|---|---|---|
| results/tournament/seed_1000.json | 20186fe52e857fcdd4c8229b0a0859df8e98c8b6 | 19e04ef0afbb41e9a06994d7e5ee6c9820ad29e9d603280fed5f521f67bfc7df |
| results/tournament/seed_1001.json | 084fd9f08329745970303f85d6dc64f61af13fe5 | 90d694298e5a70f6847db1f4003a41e96bbff1af029cc87face498b0cdb53ab3 |
| results/tournament/seed_1002.json | 65aa980b88a629c0e1242b768b5babb047b00cb7 | abec8ba7d79648dba781b1c705f94aa8996e3dfb0b2d8af99ef5eeb942b3b665 |

The added `results/corpus_v1_sha256_manifest.json` in PR #203 is a source-reported corpus checksum manifest (blob `06f84908bdc7ae13dda5b4c4bb807489d953dc26`). It does not by itself bind the original Kaggle runs to the corpus bytes, trainer, scorer or checkpoint. Dataset rights are not cleared by hashes. No dataset or checkpoints were downloaded here.

## Verification and usage

`strict_receipt_gate.py` (blob `838fc65efa631f0ddb8d823f9573a25603c4aa95`) and `test_strict_receipt_gate.py` (blob `867cc37a8683ac7a300f670f24b89c9c6de66827`) passed 8 unit tests in an isolated Linux Python 3.13 CPU environment using synthetic fixtures and an injected scorer. GitHub readback matched the exact locally tested file blobs. The real receipt rows and hashes were checked independently through the connector; **the Python wrapper was not run against the actual three original GitHub files here**, and no owner Mac or training runtime was accessed. Execute the full integration check on a checkout with this branch:

```sh
cd research/frontier/gate1_and_depth_routing
python -m unittest -v test_strict_receipt_gate
python strict_receipt_gate.py results/tournament/seed_1000.json results/tournament/seed_1001.json results/tournament/seed_1002.json --expected-receipt results/tournament/strict_audit_receipt.json
```

The output is a review-only report, not architecture authorization. The existing post-audit scorer can emit a qualification proposal under its amended decision logic; the wrapper does not rewrite or re-register that logic. Next cheapest decisive checks: run the above integration command with a verified checkout, independently review the wrapper, and bind original training inputs/checkpoints to immutable execution receipts where available. No paid compute, training, protocol edits, raw-result changes, PHI, owner-local access or merge.
