# V5 late addendum: newly committed store/cache results

**Fixed late cutoff:** `c597d5fefdd378c3e5c66a51e7690cdb90853287`. **Preserved core cutoff:** `0afd5629990b4e442fb3aa8f7c700b030d8faede`. **Review date:** 2026-11-22. This is a bounded addendum, not a replacement of the earlier source snapshot, claims as of that snapshot or validation receipts.

## 1. What changes now

Three additional JSON results establish that corrected in-context cache evaluations were subsequently committed at S and M, and that a store evaluation was committed at M. **The M cache result meets the numerical stopping branch registered on the approximately-40M checkpoint:** every domain gain is below 0.01 BPB. The baseline-control discrepancy and other evidence qualifications remain, so this is a qualified numerical decision, not a fully cleared independent replication or proof against all copy mechanisms. **The registered approximately-101M store decision remains pending:** the new store result is M, not L.

| Late artifact | Model size derived from source shapes | Weighted model BPB | Augmented BPB | Reported-row gain | Decision relevance |
|---|---:|---:|---:|---:|---|
| `cache-exchange-S.json` | 14,292,224 | 1.1442 | 1.1406 | +0.0036 | Supplementary S result; does not set the M gate |
| `cache-exchange-M.json` | 40,379,904 | 1.0481 | 1.0463 | +0.0018 | Applicable to the registered 40M cache rule |
| `store-exchange-M.json` | 40,379,904 | 1.0481 | 1.0495 | **−0.0014** | Store worsens the reported mix at M; cannot activate the L gate |

The previous statement “no committed causal-cache rerun at cutoff 0afd” remains historically accurate. The current combined V5 narrative must now add this separately dated evidence rather than continuing to describe the cache evidence as wholly pending. The original future-leaking cache result remains invalidated permanently.

## 2. Frozen provenance and bounded inspection

Five exact file copies are preserved with Git blob hashes, SHA-256 hashes, byte lengths and immutable source URLs in the [late source manifest](../evidence/upstream_v5_store_late/source_manifest.json): the three JSON results, `INVESTIGATION.md`, and the late `tpu/train_jax.py`. The core source manifest and its original ten store/cache files remain unchanged. The [late reconciliation](../analysis/reconcile_store_cache_late.py) verifies five late and ten prior copies before reading their data; its [saved receipt](../results/store_cache_late_reanalysis_v5.json) contains the arithmetic, scale mapping and decision classifications.

| Frozen late source | Git blob SHA |
|---|---|
| [cache-exchange-S.json](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/c597d5fefdd378c3e5c66a51e7690cdb90853287/research/frontier/ontogenesis/results/lm/cache-exchange-S.json) | `3120320aee8c133f804ee3d9a816c03601bb7bb4` |
| [cache-exchange-M.json](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/c597d5fefdd378c3e5c66a51e7690cdb90853287/research/frontier/ontogenesis/results/lm/cache-exchange-M.json) | `799c64b014c9f71fe40aa81bc1f5d5d64f3c21ab` |
| [store-exchange-M.json](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/c597d5fefdd378c3e5c66a51e7690cdb90853287/research/frontier/ontogenesis/results/lm/store-exchange-M.json) | `6a7921ca27d4a88db61935a4d780c7ee5bc46b19` |
| [INVESTIGATION.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/c597d5fefdd378c3e5c66a51e7690cdb90853287/research/frontier/ontogenesis/INVESTIGATION.md) | `447ec7e9645d919a0e76bd4b20c0cf519b0805f1` |
| [train_jax.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/c597d5fefdd378c3e5c66a51e7690cdb90853287/research/frontier/ontogenesis/tpu/train_jax.py) | `e1d34ed3e8ab8f20fa136a1d9a3ceb2f74b77c11` |

The fixed Git comparison contains four commits and ten changed paths. Its head has commit timestamp `2026-11-22T02:35:42Z`. The late source versions of `cache_exchange.py` and `store_exchange.py` were fetched and their blobs verified equal to the core versions: respectively `2eb6c34e50366154976bf0c8be844cd4a5fa72e4` and `5df9cd372c3d841479aae8478a71b7323d9d4979`. Their prior frozen copies therefore remain exact source references for the evaluator logic. The manifest records these equal identities; they are not counted as extra copies.

This audit did not load checkpoints, import JAX/model code, extract a corpus, run training, run an evaluation or inspect any head after c597. It executed only JSON/hash checks, decimal arithmetic and AST comparisons. Git timestamps and hashes establish repository provenance, not the original execution time or custody of a checkpoint.

## 3. Full domain arithmetic and controls

Positive gain means lower augmented BPB. Every displayed gain is reproduced by exact decimal subtraction of the committed rounded rows.

| Domain | S cache gain | M cache gain | M store gain |
|---|---:|---:|---:|
| Web | +0.0023 | +0.0012 | −0.0026 |
| Textbook | −0.0005 | −0.0003 | −0.0014 |
| Math | +0.0068 | +0.0036 | +0.0016 |
| Code | +0.0114 | +0.0059 | +0.0016 |
| Reported weighted mix | +0.0036 | +0.0018 | −0.0014 |

Both cache results report `causality_check_passed: true`. All their foreign-cache gains round to zero. The M store's shifted-control mixed gain and maximum fitted control weight also round to zero. These records are consistent with the corrected source, but a serialized Boolean is not an independently rerun causality test. The already executed source-function fixtures and source inspection establish the repaired timing mechanism; they do not independently reproduce these language scores.

Every late result reports `max_abs_difference_from_reported: 0.0001`. M store explicitly records the training-run baseline: web 1.1075, textbook 0.7902, math 1.1822, code 0.8750, mix 1.0481. Its new model-only rows are higher by 0.0001 on **three** rows—web, textbook and math—and equal on code and mix. This differs from the earlier S result's four discrepant rows.

The cache JSONs contain only the maximum baseline difference, not per-domain training-run baselines. Their checkpoint paths and model-only rows exactly match the corresponding S/M store records. Thus the S four-row and M three-row cache discrepancies can be reconstructed **by cross-record inference** using those store references. The receipt labels this inference explicitly; it does not invent missing cache fields. The literal baseline-equality condition is not fully met, and the scripts still do not directly enforce zero-weight equality through the mixture loss path.

The M cache's largest domain gain, code at 0.0059, lies 0.0041 below the 0.01 stopping threshold. Simple four-decimal rounding cannot erase that numerical gap. This observation is not a confidence interval or a bound on unmeasured evaluation bias. No per-example loss data, paired document uncertainty or independent checkpoint repeats are supplied.

## 4. Scale mapping and the registered gates

The SIZES table's M comment says approximately 42M, while the registration describes approximately 40M. The dimensions resolve this naming ambiguity. Counting tensors from `init`—tied vocabulary embedding, final normalization, per-block query/key normalization, Q/K/V/O and gated feed-forward matrices—gives the following static counts. The same shapes are present in both cutoffs; these are counts derived from source, not verified checkpoint inventories.

| Source size | Derived parameters | Registered shorthand |
|---|---:|---|
| S | 14,292,224 | 14M |
| M | 40,379,904 | 40M |
| L | 100,684,032 | 101M |

**Cache rule.** The preserved cache source reads its decision on the 40M checkpoint: code gain at least 0.02 requests trained copy-head arms; below 0.01 on every domain closes that explicit-copy line at the tested scale; the intermediate region requests one 14M trained arm. M has gains 0.0012, −0.0003, 0.0036 and 0.0059. The numeric stopping branch therefore applies. S's code gain of 0.0114 does not override a rule registered on M. The appropriate ledger status is **“registered numerical stopping branch met; control/provenance qualifications remain.”** This supports limiting follow-up on this exact comparator under the stated practical threshold. It does not prove that learned pointer mechanisms, different cache capacities, semantic retrieval or all external memory are useless.

**Store rule.** The decision note reads its threshold on 101M: at least 0.02 motivates more store work, 0.01–0.02 retains it as a baseline, and below 0.01 closes that n-gram-store line. The new store-M result is negative, but the registered scale is still L. No L store result is present in the preserved base tree plus the bounded late comparison. The store decision stays **pending the registered L result**, even if the M result reasonably lowers enthusiasm for this specific design. Moving the stopping scale after seeing M would be a new decision, not automatic completion of the original gate.

## 5. What the new train source changes

Commit `96c1863113dd8755f93d6a56655da955d2ab4acb` changes the training pipeline. The prior version sampled crop positions on the JAX device; the late version samples them on the host with NumPy, changes run names to include `-hs`, and adds distillation and run-management logic. The late source's docstring attributes a nonuniform training distribution to the old sampler and reports approximately 12% excess sampling over the first 37% of an 800M-token file. **Those numerical bias estimates are source-reported and were not independently verified in this audit.**

AST comparison confirms that eight relevant definitions are unchanged: `init`, `rms`, `rope_tables`, `apply_rope`, `forward`, `count`, `loss_fn`, and `evaluate`. The source update therefore does not, by itself, change the inspected model architecture or baseline evaluation formula. It does change the training recipe and future checkpoint provenance.

All three late result paths use the older `dense-S/M-lr0.001-s1000.npz` naming convention, without `-hs`. This is evidence consistent with first-version checkpoints. Because checkpoint payloads and executed recipes are absent, it is an inference from naming, not a verified training-history fact. These files provide **no result for a repaired host-sampler checkpoint**. They must not be described as a validation of the sampling repair or of the new distillation objective.

The 23 added investigation lines register a separate distillation proposal, with new baselines and source-specified thresholds. This addendum records that plan's existence only. It does not execute the proposal, accept its causal interpretation of capacity, or import its “twice the baseline-seed difference” rule as a formal significance guarantee.

## 6. Unchanged evaluation and resource qualifications

| Question | Late evidence | Remaining limit |
|---|---|---|
| Calibration exposure | Each cache fits 524,288 target positions; store-M fits 1,048,576 | Presentations can repeat source positions; raw fitting/crop identities are absent |
| Store storage | Same 1,022,224,980 key/count-array bytes | This excludes construction peaks, model, workspace and other process memory |
| Cache support | S and M both report web 0.132, textbook 0.094, math 0.325, code 0.445 | This is true-target positive-mass support, not cache accuracy or guaranteed mixture benefit |
| Store equal-prefix comparison | Exchange code is unchanged | The first two crop targets can still use pre-crop context unavailable to the model |
| Cache causal scope | Corrected explicit mode; per-crop cache resets | Document crossings and native input identity are not established by the raw artifacts |
| Foreign/shifted control | All displayed control gains round to zero | Disjointness, independence and deployability do not follow from zero rounded gain |
| Document exclusion | No new document-aware store/cache interface | Leave-one-position-out does not equal leave-document-out; joins/duplicates remain unresolved |
| BPB denominators | Same source formula and corpus-manifest dependency | Raw NLL, actual scored bytes, mixture weights and tokenizer identity are unrecovered |
| Runtime | Whole-script seconds: S cache 277, M cache 720, M store 682 | No isolated latency, hardware identity, matched resource budget or uncertainty |

The support fractions are identical across S and M, as expected for deterministic cache support on the same stated crop sampling independent of model weights. Their equality is consistent evidence, not proof that the unrecorded crop manifests and checkpoint execution histories were identical. Store and cache comparisons also differ in calibration exposure and access to external versus local context, so their gains alone do not rank them on a resource frontier.

## 7. Disposition and next evidence

Preserve the core “as of 0afd” observations and append the late evidence to the decision history. Update the current summary from “no committed cache result” to “reported corrected S/M cache results with small gains; M meets the numerical stopping branch under unresolved controls.” Retain the historical future-leak invalidation and the pending 101M store gate. No architecture branch is selected by this addendum.

Before treating this as a fully cleared evaluation decision, resolve baseline equality with identical precision/crops and a direct zero-weight test; bind actual checkpoint/data/tokenizer identities; preserve exact mixture weights and per-example NLL/byte attribution; enforce declared document/prefix boundaries; and report repeat/paired uncertainty and resource measurements. A repaired host-sampler comparison needs its own checkpoints and receipts. No such work is claimed complete here.

### Deterministic audit handoff

The late reanalysis reports five verified late copies, ten verified prior copies, fifteen reconciled late BPB rows, two unchanged exchange-script blobs and eight unchanged model/evaluation function ASTs. `registered_cache_40M_gate.all_four_domain_gains_below_0_01` is true; `source_controls_fully_satisfied` is false. `registered_store_101M_gate.decision` is `pending_registered_L_result`. These are deterministic evidence checks, not new neural experiments.

Reanalysis source SHA-256: `78c516e2eac1715d5a34a61ee0708b5597a33dc2e4dc1d6c2e4db4d1a823e158`. Saved receipt SHA-256: `fa75a633bab73d3b1e701f79d6a053867711157b0f9290b4549c4a77928d0805`.

## Final chapter-addendum review closure

The store/cache reviewer independently read the completed chapter 13 late-addendum notice and §7 against the five frozen late sources, the unchanged evaluator identities and this reconciliation. **Reviewed chapter SHA-256:** `ff2fb8795351c7d15fd51107e7b003e5ec490a94d12e3d37ad77e3be28ffd07d`.

No blocking factual or proof defect was found. The chapter reports all fifteen gains correctly, distinguishes source-derived parameter counts from checkpoint inventories, applies the approximately-40M cache rule only as a qualified numerical stopping branch, and leaves the approximately-101M store rule pending. The baseline comparisons are correctly separated into directly recorded M-store fields and inferred cache per-domain differences. Cache support is not mislabeled as accuracy; zero rounded controls are not treated as independence proofs. Sampling-version attribution remains an inference from naming, and no repaired-sampler, distillation or corpus-production outcome is claimed.

The notice explicitly preserves the earlier cutoff-bounded absence statement and the historical invalidation. Its cost and denominator caveats remain consistent with the source audit. This closure binds the completed chapter content above; the final release validator may now run with source/review edits stopped. No additional model or corpus execution was performed for this closure.
