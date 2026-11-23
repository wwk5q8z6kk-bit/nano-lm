# Contextual evaluation and scorer audit — V6


**V6 package relocation:** exact upstream source bytes are unchanged. Packaged read-only audit runners are in `analysis/reconcile_upstream_v6.py`, `analysis/reconcile_correction_v6.py` and `analysis/reconcile_context_v6.py`; their stdout is compared with saved audit results by the V6 release gate. Original review workspace commands below describe the audit session.

**Research boundary:** repository `wwk5q8z6kk-bit/nano-lm` at `3a38b5f1529a718838b5bc83a010574bfa9a4c6e`; PR207 evidence review inspected at `f05da96cb10c22982a8271f25c708bb064c7b371`. **Scope:** source and committed result inspection, bounded synthetic scorer diagnostics and a new local evaluation instrument. No checkpoint, tokenizer binary, training corpus, training run or paid compute was used. The V5 package and upstream repository were not modified.

## Executive finding

The next useful free task is now concrete: a frozen generated relational evaluation with public prompts, a private semantic oracle, paired context interventions, strict generated-answer scoring and a generic native-checkpoint adapter contract. It is an **evaluation instrument**, with no measured learned-model performance yet.

The historical benchmarks remain useful evidence under their recorded harness. They do not prove why contextual performance differs, and their existing receipts do not independently establish tokenizer alignment, exact context lengths, absence of truncation or benchmark/checkpoint identity. This review preserves their measurements while distinguishing code-contract weaknesses from demonstrated historical errors.

| Finding | Evidence class | Action |
|---|---|---|
| No dedicated generated-reasoning D protocol identified in the inspected frozen research evaluation paths | Source inventory | Add the new instrument in V6; retain historical scorers |
| Native scorer uses separately encoded context/continuation, EOS 1 and a 512-token input window; HF scorer uses joint encoding without the same prefix/window | Direct code fact | Register and audit each model's token boundary, BOS and context policy before comparisons |
| WinoGrande no-context candidates are identical | Direct code fact; synthetic reproduction | Treat 0.4957 as the first-option gold frequency, not a contextual-capacity baseline |
| Empty continuation can pass the greedy all-token test vacuously | Synthetic schema witness | Reject empty/malformed records in new instruments; do not assert existing tasks contain them |
| Q13 scorer accepts a deliberately mismatched synthetic checkpoint/benchmark identity | Synthetic robustness witness | Bind future run, checkpoint, data and benchmark hashes |
| Seven committed benchmark sizes sum to 22,348, not 30,328 | Arithmetic correction | Correct the synthesis; preserve raw results |
| ARC-Easy no-context native minus Smol135M equals 0.0046 | Arithmetic correction | Replace “within 0.004 on every task” with the actual table |
| “Longest 259” has no committed tokenizer/task length receipt in the inspected evidence | Provenance limitation | Report as an upstream assertion until item-level lengths and hashes are available |
| Corpus contamination clearance | Not established | Keep NOT_CHECKED for V6; characterize old 13-token scan narrowly |
| V6 generated instrument checks | Local measured software behavior | 31 focused checks pass after independent-review repairs; symbolic controls only |

## 1. Existing results: what is measured

The following values are the committed aggregates, not a fresh checkpoint evaluation. All accuracies are plain accuracy; character-normalized scores exist separately. The seven task sizes total **22,348**. The narrative's 30,328 does not match these files. [S5–S8]

| Task | N | Native 386M context | Native no context | Smol135M context | Smol135M no context |
|---|---:|---:|---:|---:|---:|
| HellaSwag | 10,042 | 0.3427 | 0.3023 | 0.3548 | 0.3061 |
| ARC-Easy | 2,376 | 0.5964 | 0.3333 | 0.6444 | 0.3287 |
| ARC-Challenge | 1,172 | 0.2602 | 0.2218 | 0.2807 | 0.2201 |
| PIQA | 1,838 | 0.6779 | 0.6311 | 0.6844 | 0.6273 |
| WinoGrande | 1,267 | 0.5201 | 0.4957 | 0.5296 | 0.4957 |
| OpenBookQA | 500 | 0.2180 | 0.1260 | 0.2180 | 0.1300 |
| LAMBADA | 5,153 | 0.3544 | 0.0000 | 0.4289 | 0.0000 |
| **Total records** | **22,348** | | | | |

Native LAMBADA word-level perplexity is 32.06; the reference is 19.26. The registered Q13 result and the 386M/7.70B-token/.8049-BPB synthesis are existing upstream evidence; this task did not reproduce their underlying run. [S5, S7, S10]

For ARC-Easy, native contextual gain under this harness is 0.5964−0.3333 = **0.2631**. The reference gain is 0.6444−0.3287 = **0.3157**. The difference in no-context accuracies is **0.0046**, which is greater than the literal 0.004 narrative bound. Similar no-context accuracy is evidence about these answer-prior controls. It does not identify training compute, architecture or representation quality as a unique cause of the contextual gap. Dataset sampling, prompt/tokenization policies, model training history and ordinary item-level uncertainty remain possible contributors. [S1, S5–S8]

## 2. Prompt, tokenization and target contracts

`eval_prep.py` downloads seven datasets and creates task-specific formatted strings. HellaSwag joins activity/context text; ARC and PIQA use a question/answer prefix; OpenBookQA uses the stem; WinoGrande inserts each option into the prefix and scores a shared suffix; LAMBADA separates the final word. Native context and continuation are then **encoded independently**. Dataset revisions and per-item provenance are not fully pinned, and PIQA download failure is caught by omitting the task. [S2]

The native scorer prepends EOS ID 1, concatenates token IDs and left-crops to 513 IDs, providing at most 512 model inputs and their shifted targets. The target-slice arithmetic is correct on the ordinary and long-context synthetic examples inspected. Cropping removes the initial EOS when needed, so the “every context starts with EOS” comment is not a universal invariant. Continuations exceeding the window are rejected, but empty continuations are accepted and produce an empty target slice. An all-token greedy check on that empty slice is true. That is a robustness problem; no actual empty benchmark row was observed. [S3, D1]

The HF path encodes the joined context and continuation, sets the target start from the separately encoded context length, adds no prefix to nonempty context and has no explicit 512-token crop. It does not verify that the context-only encoding is a prefix of the joined encoding. Some tokenizers can retokenize across that boundary. The current leading-space continuation templates may avoid the problem for particular models, but the tokenizer bytes and complete prepared task files were not available here to establish that. These are **conditional contract risks**, not proof that the recorded scores are wrong. [S4]

LAMBADA uses teacher-forced token log probabilities and an all-target-tokens-greedy criterion. If every target token matches greedy prediction, each previously supplied target token also matches what greedy generation would have produced; therefore that criterion is a valid exact-target check under the given prefix. It should not be dismissed merely because it is teacher forced. It still does not test unconstrained generated reasoning responses with format, stopping, unknown-answer and partial-output policies. [S3–S4]

Both paths report a character-length-normalized multiple-choice statistic. This is a separate endpoint from plain accuracy and from token-length normalization. Cross-tokenizer comparisons should preserve the chosen endpoint and disclose BOS, target boundary, numerical precision and context policy.

## 3. Context length and provenance

The latest investigation asserts that the longest benchmark example contains 259 tokens and no 512-token truncation occurs. No committed length manifest, prepared tokenized task file or tokenizer/checkpoint content hash supporting that maximum was found in this audit's exact frozen source set and inspected research evaluation tree. This review cannot verify or refute the asserted maximum. A tokenizer-specific maximum is also not automatically transferable to every HF reference. [S1–S4]

To close that gap, a future benchmark receipt needs per-item IDs, context/continuation token counts, special-token count, target offsets, crop count, prompt/tokenizer/model hashes and the exact dataset revisions. Aggregate maxima should be computed from those saved item records, with zero-truncation asserted mechanically. The same receipt should record omitted tasks and explicit failure statuses instead of quietly changing the suite.

The Q13 scorer checks completion and the final evaluation step. It mechanically implements its registered comparisons, including the constrained half-rate-rerun rule. It does not bind the benchmark's checkpoint identity to the run or validate the full source-data/configuration chain. A tiny synthetic invocation with run name A and a foreign checkpoint B benchmark was accepted and labelled A. This demonstrates a missing binding check, **not** an actual historical mismatch. [S9, D1]

## 4. Contamination interpretation

The existing scan hashes contiguous 13-token windows of each gold context-plus-answer sequence and compares them with a training token stream. Its “window touching answer” predicate means the matching window includes **at least one** continuation token, not that the whole answer was copied. A low match rate does not rule out paraphrases, source-level overlap or shorter memorized fragments. [S11–S12]

The scan also does not exclude document-boundary windows, recheck hash matches against exact bytes, identify every model's actual sampled exposure or produce a scored clean-subset comparison in the inspected output. The 10B-token scanned corpus is not identical to proof of which tokens a particular 7.70B-token run saw. The source already warns that overlap is not proof of memorization and that clean results do not exclude paraphrased contamination. Preserve that qualification.

For V6, generated disjoint names, worlds and fact sets plus split-overlap checks reduce direct reuse within this instrument. They do not prove absence from a model's training material, especially after publication. **Contamination remains NOT_CHECKED.**

## 5. V6 implementation delivered

The initial five deliverables and the separately preserved final review receipt are now present:

| Artifact | Function |
|---|---|
| `experiments/generative_reasoning_protocol_v6.json` | Frozen machine-readable task, information, metric, context and output policy |
| `experiments/generative_reasoning_protocol_v6.md` | Human-readable protocol and commands |
| `experiments/generative_reasoning_v6.py` | Deterministic generator, validation, immutable dataset output, scoring and generic adapter runner |
| `experiments/test_generative_reasoning_v6.py` | 31 focused semantic, leakage, missingness, context and receipt checks |
| `results/generative_reasoning_checks_v6.json` | Historical initial 27-check receipt, preserved unchanged |
| `results/generative_reasoning_checks_v6_final.json` | Current repaired 31-check receipt, source hashes and complete denominators |

The immutable public/oracle/manifest files are separately saved under `context_audit/generated_relations_v6/`. The recipe was written before the first generated check at **2026-11-23T21:01:36Z**, SHA256 **1d86c0b00d4c6db8974fd2bb3f739ba54809daedaa8a839109521c2184fd49b6**. This is a local freeze, not independently witnessed preregistration.

The default recipe has 16 development and 32 evaluation worlds per task family: **96 worlds, 480 records**. Each world supplies full A, full B with a different relevant value, reordered A, no context, and a missing-link case. The gold follows visible facts; private metadata does not create an unlearnable random target.

A candidate receives only `CandidateView(prompt)`. A trusted coordinator retains IDs and the private oracle. The generic runner resets state per case, obtains actual tokenizer input IDs, checks the full 32-token output allowance against context capacity, and refuses silent cropping. Real checkpoint use requires a separately implemented and validated adapter; none is loaded by these checks.

Primary scoring requires both same-query counterfactual answers to be correct. Original-target contextual gain compares A and the no-context output **against A's target**. Correct UNKNOWN abstention is reported separately. Missing outputs, format failures, backend errors, context ineligibility and output-budget exhaustion remain explicit failures in complete denominators.

Model receipts include input/output IDs, hashes, context and stop policy. Attempted prompts are counted even if generation later errors; failed-call output token costs are explicitly unknown. Receipts check consistency. They cannot independently attest which checkpoint executed or that a backend's raw text decodes from the supplied IDs. The real adapter must verify that boundary; private/public files are not an OS security sandbox.

## 6. Observed instrument checks

The final saved run passed **31 checks**, with zero errors, failures or skips. The initial 27-check receipt and its exact code/tests are preserved as historical evidence. No legacy full-repository suite was run. The exact visible-facts parser agrees with the separate structured oracle on all 480 records. A planted cross-split duplicate was rejected, as were missing-world schemas, duplicate IDs, tampered gold/hashes, false eligibility, silent truncation, nonfinite values and identity inconsistencies.

| Symbolic/control method | Development primary A/B joint | Evaluation primary A/B joint |
|---|---:|---:|
| Visible-facts symbolic oracle | 32/32 | 64/64 |
| Always UNKNOWN | 0/32 | 0/64 |
| Constant value | 0/32 | 0/64 |
| First visible value | 4/32 | 7/64 |
| Last visible value | 8/32 | 5/64 |

These figures show why shortcuts and all-condition checks matter. First/last-value strategies sometimes select the relevant fact accidentally. The deterministic oracle's perfect result establishes semantic solvability and scorer behavior; it is not learned-model competence.

The evaluation split contains 222,368 planned prompt UTF-8 bytes and 112,608 context bytes across 320 records. Development contains 111,184 and 56,304 respectively. Actual model tokenizer counts and model context eligibility remain **NOT_CHECKED**; model execution remains **NOT_RUN**.

### Independent-review repairs and unchanged dataset

The independent adversarial review found that the initial implementation could accept a coherent in-memory dataset edit under the original manifest identity, accept an EOS-labelled response that bypassed the first-newline policy, and retain aliases to mutable adapter token buffers. These were actual instrument defects, not model findings.

The repaired scorer rebinds the current canonical public/oracle/manifest bytes at each scoring call; any newline now requires the registered first-newline stop; preflight IDs are immutable snapshots, and receipt input/output arrays are copied. Four new regressions cover these paths, including buffer reuse across cases. Generator provenance remains the original generator hash, while the result separately names the current scorer. The public/oracle bytes and dataset manifest are unchanged.

Initial source SHA256: `6f79b8f606bc7fa780c9724ad199a44628dfa886e762783292f4fba19ef14c73`. Repaired source: `fb189cfdac616c3036f8bfe3e443bc39ff8bcaf746d0f09f145185d24eae2ff8`. Final receipt: `44af92e3cf4938731559091368e1d205b838b25d450a2c840cdbed2ccdd917eb`. Exact historical sources are under `context_audit/initial_instrument/`; `context_audit/instrument_revision_history.json` records the receipt chain. The independent recheck passed all 21 adversarial checks plus all 31 supplied regressions; all three findings and their four manifestations were closed. The separate [independent review](v6_generation_independent_review.md) records that result and confirms byte-for-byte regenerated public/oracle content and unchanged control metrics.

## 7. Next concrete work and limits

A real NanoLM adapter is the next dependency: bind native `load_params`/`forward`, exact tokenizer/BOS IDs, model config, greedy decoder, stop rule and checkpoint hashes; verify its raw output against decoded tokens; then execute the frozen evaluation when checkpoint access and execution authorization exist. This does not require adopting an external pretrained model as the project's base.

Do not use these instrument controls to select a winning architecture, claim natural-language generalization, close a training efficacy gap or certify corpus cleanliness. The shared grammar and short relational tasks are deliberately narrow. Publish separate development/evaluation/family metrics and preserve the original freeze and raw receipts if implementation defects require a revision.

## Exact evidence and reproduction

`context_audit/source_manifest.json` records exact Git blob SHA1, SHA256, byte length and local source path for 16 inspected files. Every fetched file's Git blob identity was recomputed from its exact bytes. The recursive tree response was nontruncated (4,594 entries). `context_audit/source_diagnostics.py` and `source_diagnostics.json` reproduce the arithmetic and source-only witnesses without importing JAX.

References below identify primary repository sources at exact commits:

- **S1:** [INVESTIGATION.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/INVESTIGATION.md)
- **S2:** [eval_prep.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_prep.py)
- **S3:** [eval_score.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_score.py)
- **S4:** [eval_score_hf.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_score_hf.py)
- **S5:** [386M benchmark](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/dense-XXL-lr0.002-s1000-tok7.7e%2B09-muon-dp4-fl0.01-hs-v2.bench.json)
- **S6:** [386M no-context benchmark](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/dense-XXL-lr0.002-s1000-tok7.7e%2B09-muon-dp4-fl0.01-hs-v2.bench-nocontext.json)
- **S7:** [Smol135M reference](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-reference-SmolLM2-135M.json)
- **S8:** [Smol135M no-context reference](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-reference-SmolLM2-135M-nocontext.json)
- **S9:** [score_q13.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/score_q13.py)
- **S10:** [Q13 verdict](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/scale2/q13-verdict.json)
- **S11:** [eval_contam.py](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/tpu/eval_contam.py)
- **S12:** [Contamination scan aggregates](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/3a38b5f1529a718838b5bc83a010574bfa9a4c6e/research/frontier/ontogenesis/results/lm/eval-contamination-v2.json)
- **S13:** [PR207 evidence review](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/f05da96cb10c22982a8271f25c708bb064c7b371/research/frontier/ontogenesis/reviews/2026-11-23-evidence-and-decisions.md)
- **D1:** [Local source diagnostics](../results/context_source_diagnostics_v6_original.json), with [reproduction script](../analysis/reconcile_context_v6.py).

