# Store exchange and causal cache review

**Research cutoff:** `0afd5629990b4e442fb3aa8f7c700b030d8faede`, repository `wwk5q8z6kk-bit/nano-lm`. **Review date:** November 21, 2026, America/New_York. **Scope:** source inspection, committed rounded-summary arithmetic, and synthetic integer counterexamples. No language-model checkpoint was loaded and no corpus was accessed or evaluated. This review adds a causal reference comparator; it does not repair or replace historical measurements.

## 1. Decision summary

The store-S measurement is recoverable as a small reported improvement with significant protocol qualifications. The corrected negative control is a coherent distribution, but its asserted lack of information is unproved. A separate, previously unrecorded crop-boundary mismatch gives the store extra causal context. The first in-context cache result is invalid because the evaluator actually inserted future transitions. The repaired cache code has no committed valid result in the inspected tree. Neither the registered 101M store decision nor the 40M cache decision can be resolved by this S result.

| Evidence | Classification at cutoff | Permitted conclusion |
|---|---|---|
| Store-S primary rows | Reported measurement, arithmetic reproduced; equal-prefix and baseline-control qualification | This frozen checkpoint plus this fitted store reports a 0.0044 weighted-BPB reduction |
| Original shuffled true-token scalar control | Invalidated control; raw JSON retained | Its 0.1752 gain cannot serve as a proper negative-control estimate |
| Replacement shifted-context control | Proper count distribution; rounded zero gain reported | Zero gain was observed in this fitted run; independence is not proved |
| First cache result described as 0.62 BPB | Source-reported invalidated result; no corresponding raw result recovered | Object-identity routing caused future leakage, independently reproduced on integers |
| Repaired causal cache | Inspected source; valid language result pending | Explicit mode removes the demonstrated identity bug |
| New local causal comparator | Executable fixture-validated implementation | Prefix/reset/timing/distribution contracts can be exercised without model training |
| Store 101M / cache 40M decision gates | Pending | S numbers cannot trigger decisions registered on other model scales |

The exact source copies, historical variants and Git blob/SHA-256 manifest are preserved in [`evidence/upstream_v5_store`](../evidence/upstream_v5_store/source_manifest.json). The read-only [reconciliation script](../analysis/reconcile_store_cache.py) verifies all ten copied files before extracting allowlisted functions. Immutable upstream source URLs appear in the manifest; local source labels below refer to that manifest.

## 2. Recovered store result and arithmetic

Source: `current/research/frontier/ontogenesis/results/lm/store-exchange-S.json`.

| Domain | Model BPB | Model + store BPB | Difference | Shifted-control BPB | Difference from training-run baseline |
|---|---:|---:|---:|---:|---:|
| Web | 1.1945 | 1.1929 | 0.0016 | 1.1945 | +0.0001 |
| Textbook | 0.8940 | 0.8887 | 0.0053 | 0.8940 | 0 |
| Math | 1.2963 | 1.2851 | 0.0112 | 1.2963 | +0.0001 |
| Code | 0.9903 | 0.9798 | 0.0105 | 0.9903 | +0.0001 |
| Reported weighted mix | 1.1442 | 1.1398 | 0.0044 | 1.1442 | +0.0001 |

Decimal subtraction reproduces every stored gain. The mixed reduction is approximately **0.3845% of the reported model BPB**. This is arithmetic on rounded rows; it supplies neither uncertainty nor a per-token replication. The same primary rows survive the control correction from commit `7351fb137985d3cfefc71b0eb96e850cab53ec81` to the cutoff. Preserving them does not demonstrate that the new control fixes every evaluation issue.

The result records **1,022,224,980 bytes** of count/key arrays, about **0.9520 GiB**, and **1,048,576 fitting target positions**. The source constructs the store from 48 concatenated one-million-token training chunks. The result names one S checkpoint with `s1000` in its path, and records 346 elapsed seconds. Neither checkpoint bytes/digest, machine identity, peak memory, isolated inference latency, raw log probabilities, nor the executed CLI are included. Treat the checkpoint seed as source/path attribution, not a newly verified checkpoint property.

Under the source defaults, evaluation samples 20 batches × 16 crops × 512 targets = 163,840 target positions per domain; overlapping crops can repeat positions. Fitting similarly samples 64 × 32 × 512 positions. These are exposure counts, not unique-token counts. The source converts mean token NLL to BPB with the whole-domain manifest's bytes-per-token ratio, then combines rounded domain BPB using manifest mixture weights. The corpus manifest is not included in this evidence set. Consequently the mixed denominator/weights and exact sampled-byte totals were **not independently reconstructed**. A future rerun should record summed NLL, scored token positions, actual scored UTF-8 byte attribution, tokenizer identity, crop offsets, domain weights and denominator convention.

## 3. Controls: what was corrected and what remains

### 3.1 Invalid scalar shuffling

The original store script shuffled each position's scalar probability of its own true token, then treated the shuffled number as the probability of the new position's target. A scalar borrowed from another target is not generally an entry in one consistent predictive distribution for the current context. The resulting 0.1752 mixed-BPB gain is therefore not a legitimate negative-control measurement. The defect is the probability construction; a gain of that magnitude is not, by itself, a universal mathematical impossibility.

The replacement looks up a genuine training-store count distribution using a context shifted by 100,003 positions, then queries the current target under that distribution. This repairs coherence. It does **not** prove independence: distant contexts may retain domain, topic, repetitive-structure or frequency information. The cyclic shift may also read future validation contexts and wrap the file. That is acceptable only as an explicitly nondeployable diagnostic whose data access is recorded, not as a causal competitor. The observed mixed control gain rounds to zero and the fitted maximum rounds to zero; exact zero weights and zero population benefit are not established.

### 3.2 The baseline control is near agreement, not literal equality

The registered condition calls for reproducing the training-run BPB to four decimals. Four rows differ by 0.0001. The script records a maximum difference when `--reported` is provided but does not enforce a tolerance. Also, the displayed `model` row uses direct model NLL, rather than an explicit zero-weight call through the mixture loss; the latter adds `1e-12` inside its logarithm. This is not an exact identity test of the entire mixture scoring path. No cause for the recorded difference can be established from rounded summaries. Required repair: record a declared numeric tolerance before rerunning, compare identical crops/precision/denominators, and directly test the zero-weight scoring implementation.

### 3.3 Equal-prefix violation in the store evaluator

`store.lookup(d, 0)` runs on the complete validation file before the evaluator slices 512-token crops. The dense model sees only its current crop. For the first target, the store can condition on two tokens preceding the crop; for the second, one. From the third target onward, its maximum three-token context is within the model prefix. At most two of the 512 target positions per crop have this particular mismatch; the impact on the reported loss cannot be inferred from that fraction.

The pinned source itself reproduces the violation on a small integer fixture:

| Item | File A | File B |
|---|---|---|
| Full validation sequence | `1,2,3,4` | `7,8,3,4` |
| Shared visible crop | `3,4` | `3,4` |
| Store probability of target `4`, full-file lookup | 1 | 0 |
| Store probability of target `4`, crop-local lookup | 0.5 | 0.5 |

The training-store fixture is `1,2,3,4,7,8,3,5`. This is **extra causal pre-crop context, not future leakage**. Fitting uses the same global-lookup/sliced-crop mismatch. A corrected evaluation must either restrict both systems to the same crop prefix or supply the same declared extra context to both. This review preserves the old result as qualified evidence; it does not subtract an invented correction from its gain.

## 4. Exposure, self-hit and document boundaries

Training-fit `loo=1` subtracts the current n-gram occurrence from its context and true-follower counts. For an occurrence included in the fitted store, this is a leave-one-position-out construction. It does not remove its other document occurrences, overlapping contexts, duplicate documents or other sampled appearances. A train-only datastore is not automatically free of same-document fitting advantage. The registered leave-document-out variant remains unimplemented in this inspected script.

Concatenating 48 separated chunks creates synthetic adjacencies at 47 chunk joins; their effects are not quantified. Flat token files expose no document identifiers to either evaluator. The cache resets by crop, but a crop can still cross a document boundary unless the upstream data contract prevents that. Establish document-aware chunking, canonical document IDs, exact split/dedup rules, and explicit boundary resets before claiming document exclusion or equal information.

The cache foreign-window diagnostic uses the next crop in the same batch, drawn from the same domain file. Such windows may overlap or belong to related/same documents. A coherent distribution from another window is not necessarily information-free. To test provenance rather than mere label scrambling, use a frozen document-disjoint derangement, record overlaps/duplicates and identify the allowed source split. Even a truly independent corpus-frequency prior can improve a miscalibrated model; zero improvement is not a theorem of independence.

The store and dense baseline share a checkpoint, but their total storage and inference work differ. Count-array bytes omit parameters, input files, fitting arrays, construction peaks, Python overhead and temporary workspace. The single elapsed time combines activities and has no matched baseline timing. The evidence therefore supports **neither a matched-cost comparison nor a deployment-efficiency result**. A future comparison must separate model training, store construction, mixture fitting, corpus exposures, persistent memory, peak memory and inference cost.

## 5. Cache invalidation and the new causal implementation

In the old cache source, mode selection used `ctx is tgt_tokens`. Training passed one list object twice. Evaluation passed two newly created but equal lists, selecting the foreign/full-window path and inserting future transitions before scoring. With tokens `0,1,2,3`, the original same-object path returns true-target probabilities `[0,0,0]`, while equal separate lists return `[1,1,1]`. The current explicit `foreign=None` path returns `[0,0,0]`. These are executed source-function fixtures, not language data.

The current script checks suffix perturbations on three training windows, eight trials each. It compares returned true-target probabilities and gate indices. That is useful, but it is narrower than verifying a complete output distribution under all prefix-equivalent sequences. No committed valid cache rerun was located in the pinned repository tree, so the source-reported 0.62 result stays invalidated and the 40M decision remains pending.

The new [causal comparator](../prototype/causal_ngram_cache.py) deliberately exposes a smaller contract:

| Contract | New reference behavior |
|---|---|
| Input scope | Observed integer tokens after an explicit document/crop reset |
| Prediction API | No target or future suffix argument; read-only prediction |
| Timing | Predict and score, then observe the target |
| Window | Retain at most 512 observed tokens by default; count only completed transitions wholly inside it |
| Backoff | Longest available suffix match, maximum order 3 by default |
| Distribution | Positive additive smoothing over the complete vocabulary |
| Empty/no match | Uniform cache distribution; native mixture weight forced to zero |
| Masked target | Excluded from loss but observed as context; padding/missing positions must not be observed |
| State identity | Immutable snapshot plus digest of config, context identity, observed count and retained prefix |
| Model role | Optional augmented comparator; no alteration of the native architectural core |

Let the retained observed prefix be \(z_{1:L}\), and select the longest matching suffix order \(n\). Counts only include starts satisfying \(j+n\le L\). With follower counts \(c_v\), \(C=\sum_v c_v\), vocabulary size \(V\) and \(\alpha>0\), the cache uses

\[
p_{\rm cache}(v)=\frac{c_v+\alpha/V}{C+\alpha}.
\]

Every term is positive and the sum is one. Prediction depends only on the retained observed prefix and fixed configuration. By induction, resetting to the same identity and observing the same prefix gives identical counts, complete distributions and snapshots, independently of any future suffix. Scoring before insertion preserves this invariant; observing a masked target is allowed because masking removes its loss, not its causal availability afterward. Convex interpolation with a positive normalized native distribution is normalized; the no-match and zero-weight cases return the native probabilities exactly. These claims require the caller to supply the **same actual prefix to the native model** and to reset at the declared boundaries. The comparator cannot authenticate external document IDs or inspect the model's inputs.

Eight grouped fixture checks pass, including 1,093 exhaustive prefixes against a separately grouped-transition oracle and 40 target/suffix variants comparing full probability vectors. Checks cover sequential/recomputed-prefix parity, snapshot immutability, reset isolation, score-before-observe, masked timing and mixture behavior. A review correction normalizes accepted real-valued smoothing to a serializable Python float and rejects unrepresentable positive values. This is an unoptimized correctness reference; it has no measured BPB, runtime advantage or learned gate.

## 6. Registered decisions and remaining work

The store decision note sets its gate on **101M parameters**: mixed-BPB gain at least 0.02 motivates further store work, 0.01–0.02 retains it only as a baseline, and below 0.01 closes this n-gram-store line at the tested setting. The cache script sets its gate on **40M parameters**: code gain at least 0.02 motivates copy-head arms; below 0.01 in every domain closes that explicit-copy line at the tested scale; the middle region requests one 14M arm. Only an S store result is committed in the inspected tree. These rules are recorded as registered source intentions, not activated decisions or independent preregistration custody.

| Gap | Status after this review | Next evidence required |
|---|---|---|
| Rounded store arithmetic | Reproduced | Raw NLL/byte/crop manifests for full reconstruction |
| Invalid scalar negative control | Mechanism diagnosed; replacement source preserved | Disjoint-source control specification and rerun |
| Historical cache future leakage | Reproduced and invalidated | Valid corrected checkpoint evaluation |
| Shared crop prefix | Counterexample reproduced; new comparator contract implemented | Corrected upstream evaluation without extra pre-crop context |
| Document exclusion / chunk joins | Open | Document-aware store construction and split manifests |
| Literal baseline identity | Open | Explained precision/aggregation difference and direct zero-weight check |
| 101M/40M gates | Open | Valid measurements at registered model sizes |
| Matched cost / uncertainty | Open | Resource accounting and paired repeat/crop uncertainty protocol |

The prior small-model store pilot and current S exchange differ in training exposure and pipeline details. A reduced reported gain does not isolate training duration as its cause. This review also supplies no basis for rejecting continuous caches, semantic kNN stores or external memory in general.

## 7. Primary research and exact reading depth

| Source | Read in this continuation | What it supports here | What it does not support |
|---|---|---|---|
| [Grave, Joulin & Usunier, *Improving Neural Language Models with a Continuous Cache*, arXiv:1612.04426v1](https://arxiv.org/pdf/1612.04426) | PDF pp. 1–3: abstract, §2 cache description, §3 cache sum/interpolation equations | Recent-history memory; stored previous-state/observed-follower pairs; interpolation as a distribution | NanoLM gains, this count comparator's efficiency, or a complete literature review |
| [Khandelwal et al., *Generalization through Memorization: Nearest Neighbor Language Models*, ICLR 2020 / arXiv:1911.00172v2](https://arxiv.org/pdf/1911.00172) | PDF pp. 1–3: abstract, §2 datastore/inference, §3 split and extra-context evaluation conventions | Train-datastore keys/targets, normalized neighbor aggregation and interpolation; explicit disclosure of evaluation context | Equal performance of n-gram and semantic retrieval, or permission to transfer its benchmark results to NanoLM |

The first paper's cache sums only previously observed followers. The second distinguishes training-datastore retrieval from recent test-document caching and states its extra-context convention. These support keeping source provenance, causal timing and declared context lengths explicit. The equal-prefix counterexample, count normalization proof and findings above are this review's own source analysis; they are not claims attributed to either paper.

## 8. Chapter 13 proof review and numerical closure

The store/cache workstream independently reviewed [chapter 13](../13_CAUSAL_MEMORY_AND_RESOURCE_FRONTIER.md) against the pinned upstream evidence, reconciliation receipt, final comparator and fixture suite. **Reviewed chapter SHA-256:** `32ebc807a6cebb2f0006747ffa9bbc93a80da69f6c2e35d5426541e5dc15a1c2`. This closure applies to that content; later material changes require an explicit review amendment.

The chapter's five store rows, 0.0044 reported mixed gain, four baseline discrepancies, cutoff-bounded absence statements, causal-versus-equal-prefix distinction and resource limitations agree with the audit. The prefix-invariance proof correctly depends on explicit resets and caller-supplied observed tokens. Its state digest identifies retained local state, not discarded history, native-prefix agreement or external document provenance. The chapter makes no measured matched-cost or language-quality claim for the new comparator.

The proof review found a numerical gap after the first comparator freeze. Exact-arithmetic convexity does not by itself protect a floating-point implementation: with smallest positive Python float \(m\), native/cache probabilities both \((m,1)\), and mixture weight one half, computing two separately rounded products produces \((0,1)\). The cache can naturally produce this input with two vocabulary items and smoothing \(2m\); it is not merely a malformed public record.

**CC-R3 correction:** reopen the freeze narrowly, compute each convex component from the smaller positive endpoint plus a nonnegative weighted difference, and validate the final output. The mixture group now contains a smallest-positive-float regression and near-endpoint weight checks. The native-comparison reviewer independently reproduced the old arithmetic failure, checked 21 vector/weight combinations and reran all eight grouped fixtures. Its [review amendment](causal_cache_review.md) closes CC-R3. No model or corpus run was needed.

| Final artifact | SHA-256 |
|---|---|
| `prototype/causal_ngram_cache.py` | `8836cf8ab645628499a6b438963d798c6c064cf832bb67df8a42a8ac32325732` |
| `prototype/test_causal_ngram_cache.py` | `b22f37dc4fc9a9355683eec6d30f7b5a9471bc4b2a89a0178e2e3174c0abd23f` |

The new [final fixture receipt](../results/causal_cache_checks_v5_final.json) is retained separately from [the earlier receipt](../results/causal_cache_checks_v5.json). Their grouped JSON summaries are byte-identical because this regression extends the existing mixture group without changing group names or counts; the final source hashes and independent review identify the expanded assertions. Both receipts report eight passing groups, 1,093 oracle prefixes and 40 suffix variants.

Two chapter clarifications also close review findings: no-match/zero-weight identity means exact preservation of the **validated Python-float representation**, and a future valid cache evaluation would create new evidence while the historical future-leaking result remains invalidated. The normalization theorem is an exact-arithmetic statement; code checks positivity and a `math.fsum` normalization error no larger than `1e-12`. These qualifications supersede broader numerical wording earlier in this review. No remaining factual or proof blocker was found in the reviewed chapter's stated scope.
