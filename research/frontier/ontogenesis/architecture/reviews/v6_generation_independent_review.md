# Independent adversarial review of V6's generated relational instrument

**Reviewer scope:** source inspection, the supplied test suite, and independent synthetic fixtures only. No learned model, checkpoint, real tokenizer, training corpus, paid resource, or external inference service was used. The prior upstream research audit remains frozen and was not changed for this task.

**Final review status:** accepted as a bounded synthetic evaluation instrument after correction. Three material consistency defects were found in the initial implementation, reproduced in four independent checks, repaired by the implementation owner, and independently retested. The corrected source passes 31 supplied regressions and all 21 independent checks. No learned-model result is established by this acceptance.

## 1. Reviewed artifacts and evidence custody

The initial files were copied byte-for-byte into `generation_independent/initial_snapshot/` before running checks. `generation_independent/initial_sources.json` records the original paths, snapshot paths, lengths and SHA-256 values.

| Artifact | Initial SHA-256 |
|---|---|
| `generative_reasoning_v6.py` | `6f79b8f606bc7fa780c9724ad199a44628dfa886e762783292f4fba19ef14c73` |
| Frozen protocol JSON | `1d86c0b00d4c6db8974fd2bb3f739ba54809daedaa8a839109521c2184fd49b6` |
| Human protocol Markdown | `f089a7ae15e3530e5f62f73011e102e2c4686ce77b325dc909bec47352d648ad` |
| Supplied 27-test suite | `db843e3c11eed5f7c301ea0a2ba8b7c6b3f56579e0d308e757c6755039a905ae` |

The supplied **27 tests pass** on that initial snapshot. The independent review script, `generation_independent/adversarial_checks.py`, runs **21 additional focused checks**, of which **four fail on the initial source**. Those four manifestations belong to three root causes below. The script imports only the snapshotted instrument and Python standard-library modules. Temporary generated data stays in temporary directories; no production instrument files are edited.

The independent result JSON is deliberately labelled `INDEPENDENT_SYNTHETIC_INSTRUMENT_REVIEW_NOT_MODEL_RESULTS`. Some fixtures construct forged model-shaped receipts to demonstrate the trust boundary. They are not model measurements and must never be promoted as checkpoint results.

## 2. Material findings requiring correction

### D-IR-01 — The scoring API does not bind mutable dataset rows back to their receipt

**Severity:** material provenance defect in the Python scoring API. The ordinary file-loader path does check file hashes, so this is narrower than a claim that all on-disk tampering succeeds.

`score(dataset, ...)` validates current row semantics and validates the run's declared dataset-manifest hash against `dataset.manifest_sha256`. It does not verify that the current in-memory public/oracle rows are the rows whose hashes the manifest records. `Dataset` and its nested rows are mutable.

**Reproduction:** deep-copy a valid dataset; change one direct-retrieval world's A value in its visible facts; regenerate that world's affected prompt strings, expected answers, A-target fields, prompt hashes and byte counts. Keep the original manifest and manifest SHA completely unchanged. The resulting dataset still satisfies all relational and counterfactual invariants. `score()` accepts it and emits a report tied to the old dataset hash.

The claimed public-file hash is `bcacfd6c2efb7aa9e5eea4d28386dd059cd13d8e6edf37d59ccb41b166d71d71`, while the altered canonical public rows hash to `e0f2a32ca4c8151c00073a34bdf80261359a40e6b603e0ccc34774e86d4a8d78`. The report nevertheless preserves the original dataset-manifest hash. This example does **not** replace every file and hash; it exposes a consistency gap after a legitimate dataset has been loaded.

**Required fix:** revalidate dataset-row bytes/counts against the stored manifest at scoring entry, and bind the manifest object itself to its claimed manifest hash. Immutability or trusted constructors can supplement that check, but a frozen top-level dataclass alone does not freeze nested lists and dictionaries. Preserve the explicit ability to score missing predictions; do not solve provenance by filtering the registered dataset to successful outputs.

### D-IR-02 — An EOS label bypasses the first-newline stopping contract

**Severity:** material score-validity defect for externally supplied model receipts.

The initial validator applies newline checks only when `stop_reason == "newline"`. With `stop_reason == "eos"`, the following raw strings are accepted and scored correct when the identifier is the target:

```text
"\nvalue_e06e3ce3143a"
"value_e06e3ce3143a\n\n"
"\nvalue_e06e3ce3143a\n"
```

The answer normalizer strips outer ASCII whitespace, so each becomes the expected identifier. Under a decoder that must stop at the **first** newline, the first and third examples should have stopped before the answer was emitted. A receipt saying EOS cannot make that observed raw-text sequence consistent with the declared policy.

**Required fix:** enforce text-level stopping consistency independently of the caller's selected stop-reason label. Require newline-stopped outputs to have the one permitted terminal newline and no earlier one; reject EOS-labelled outputs containing an earlier newline. Preserve the distinction between a malformed receipt, a generated answer with invalid format, and valid budget exhaustion. A budget-exhausted string that happens to match the target should remain incorrect.

This fix can establish consistency between raw text and declared stop policy. Without the actual tokenizer/backend, it still cannot prove that the token sequence decoded to that text or that an EOS token occurred.

### D-IR-03 — Mutable token buffers can bypass preflight equality or rewrite earlier receipts

**Severity:** material execution-receipt consistency defect for adapters that mutate or reuse buffers, including accidental mutation.

There are two independent manifestations:

| Manifestation | Synthetic reproduction | Initial behavior |
|---|---|---|
| Input alias | `encode_prompt` returns list `[7,8,9]`; `generate` pops its last element in place and returns that same list | Equality comparison sees the already-mutated object on both sides; prediction remains `generated`, receipt records `[7,8]` |
| Output-buffer reuse | First completion returns a shared output list `[4,5]`; second completion mutates that list to `[4,6]` | Both final receipts record `[4,6]`; the first completion's originally observed token IDs are lost |

Counts alone do not catch the second case because the list length remains two. Prompt hashes also cannot detect it because they bind prompt text, not the returned token-buffer snapshot.

**Required fix:** snapshot the preflight encoding before calling generation, compare returned input IDs against that immutable snapshot, and copy/freeze input and output lists when storing receipts. A later adapter mutation must not rewrite an earlier case's receipt. Add both a single-case input-mutation regression and a two-case output-reuse regression.

## 3. Findings that passed independent examination

### 3.1 Every registered denominator is retained

The instrument contains 96 base worlds and 480 records. Independent checks covered every split/family combination:

| Split | Family | Base worlds | Records | Primary joint denominator | UNKNOWN denominator |
|---|---|---:|---:|---:|---:|
| Development | Direct retrieval | 16 | 80 | 16 | 32 |
| Development | Two hop | 16 | 80 | 16 | 32 |
| Development | All | 32 | 160 | 32 | 64 |
| Evaluation | Direct retrieval | 32 | 160 | 32 | 64 |
| Evaluation | Two hop | 32 | 160 | 32 | 64 |
| Evaluation | All | 64 | 320 | 64 | 128 |

Each per-condition policy-accuracy denominator is the base-world count for its stratum. Primary A/B joint success requires both answers. Removing **every full-B prediction** yields zero primary successes while retaining all 96 registered worlds and all 480 registered records. It records 96 missing outputs. The supplied suite also checks a single missing output and an all-missing run.

The within-world conditions are dependent, and the instrument correctly declines to treat them as independent efficacy replications. No success threshold, p-value or language-model superiority conclusion is registered.

### 3.2 Visible semantics and counterfactual invariants are correct on all generated cases

An independently written regular-expression interpreter extracts value assignments and pointers directly from each public prompt, then resolves the requested destination. It matches all 480 private oracle answers.

All 96 worlds satisfy the inspected invariants:

- A and B preserve the query and fact order, change exactly one relevant value assignment, and require different answers.
- Permuted A changes order while preserving the complete fact multiset and answer.
- No-context cases contain no facts and require UNKNOWN under the policy.
- Broken-chain cases remove the queried assignment for direct retrieval or the relevant pointer for two-hop, producing UNKNOWN. In two-hop, the destination's value remains visible, so a value-extraction shortcut has something tempting to select.
- The split validator checks disjoint world IDs, prompt hashes, complete fact-set hashes and entity/value vocabularies. The supplied planted-overlap test demonstrates that the overlap checker can fail.

A fixture answering the original A target on every B record fails every counterfactual pair. First/last-value controls are appropriately imperfect shortcuts rather than claims that every shortcut must obtain zero. A first-value rule can answer a subset correctly when the relevant fact happens to occur first; that does not invalidate a joint metric, but it reinforces why multiple controls and task strata are retained.

The permutation is a deterministic cyclic rotation. It is a valid changed-order control, not evidence of robustness to all possible reorderings. The grammar template is shared across development/evaluation, so fresh symbols do not establish template transfer or natural-language generalization.

### 3.3 Original-target gain and abstention are separated correctly

The instrument compares the no-context output against **A's original target** for contextual gain, while separately comparing it against UNKNOWN for the abstention policy. This avoids subtracting accuracies against different targets.

The independent fixture gives all answerable cases the correct value and forces every no-context case to emit A's original target. It produces contextual gain zero and no-context policy accuracy zero. With the correct UNKNOWN no-context output, the perfect symbolic interpreter has contextual gain one and no-context policy accuracy one. The two quantities are therefore operationally separated, not merely differently named columns.

Permutation agreement is also separated from joint correctness. Two equal wrong answers may count as agreement, but they cannot pass the joint-correctness endpoint.

### 3.4 Basic tampering and parsing failures are rejected

Independent checks confirm rejection of duplicate prediction IDs, unknown IDs, wrong prediction-prompt hashes, gold-only tampering, and on-disk public-file changes without a corresponding hash change. The supplied tests also cover duplicate public IDs, incomplete worlds, inconsistent counterfactuals, conflicting fact keys, extra keys, duplicate JSON object keys and nonfinite values.

Strict answer parsing rejects Unicode nonbreaking spaces, vertical tabs, punctuation, extra words, wrong case, interior whitespace and multiple answers. Only ASCII space/tab/CR/LF at the outer boundary are stripped. This is a clear parsing policy. The initial EOS/newline defect is a **stopping-policy** inconsistency before parsing, not a reason to weaken the parser.

### 3.5 Context eligibility and error denominators have the right structural design

With a synthetic limit of 100 and an allowance of 32 new tokens, 68 input IDs are eligible and 69 are ineligible. The independent boundary test confirms the ineligible case never invokes generation. The runner retains the full encoded input list rather than shortening it to fit.

Generated receipts with ineligible counts are rejected. Unknown actual input counts cannot certify a generated record. Model/tokenizer identity mismatch and a declared truncation flag are rejected. Budget exhaustion is counted and scored incorrect even if its raw answer matches the target. Reset/encoding and generation failures remain explicit outcomes, with generation-attempt exposure distinguished from preflight ineligibility. Failed generation output-token counts are explicitly unknown instead of silently treated as known zero.

These are structural checks using synthetic integer lists. They do **not** establish how many tokens a native NanoLM tokenizer assigns to any prompt, whether all cases fit its 512-token context, or whether a real adapter obeys the interface. Byte lengths in the independent result are explicitly labelled as byte lengths, never token estimates.

## 4. Receipt and control boundaries that remain intentionally unverified

### Counterfeit declarations can satisfy a consistency schema

Unmodified `instrument_control` receipts cannot simply be relabelled as a checkpoint evaluation; their schemas and identity/count requirements differ, and that misuse is rejected.

However, an external caller can manufacture model-shaped receipts with arbitrary syntactically valid checkpoint/tokenizer/config/adapter hashes, input IDs `[1,2]`, output ID `[3]`, and symbolic-oracle answer text. The initial instrument accepts those consistent declarations and reports `EXTERNAL_PREDICTIONS_SCORED`, including perfect metrics. This is the expected limit of a scorer without access to an authenticated execution backend or actual tokenizer: it checks declarations, not that they happened.

The instrument already states this limitation, and the independent fixture is recorded under an explicit non-model classification. The following must remain unclaimed after the consistency fixes:

| Claim | Evidence still required |
|---|---|
| The named checkpoint ran | Trusted execution log tied to verified checkpoint bytes and adapter identity |
| Prompt IDs came from the named tokenizer | Actual tokenizer loading/hash verification and prompt-to-ID reproduction, including BOS/special tokens |
| Output IDs decode to `raw_text` | Decode verification under that tokenizer, with an explicit EOS/newline representation policy |
| Every token ID is in vocabulary | Actual tokenizer/model configuration validation |
| Model context was not cropped internally | Backend adapter verification and generation-input tracing |
| State was really reset | Backend-specific validation; calling `reset()` and receiving `true` is not independent attestation |
| Evaluation cases were absent from pretraining | Separate corpus/provenance audit; current status must remain NOT_CHECKED |

The generic interface is a practical integration point, but its `no_silent_truncation=true` and tokenizer-eligibility fields should always be read within the surrounding **declared-receipt** scope. They are not independent execution certification. Production isolation would be a separate requirement if the adapter itself were adversarial; the current protocol correctly makes no operating-system sandbox claim.

### Information boundary and publication

Candidate calls receive only `CandidateView(prompt)`, which has no ID, world metadata, split, seed or private gold field. That prevents accidental argument-level leakage. It does not stop code running in the same Python process from reading files or importing the deterministic generator. Source code and published evaluation cases can expose the answers; fresh symbol generation is not a contamination guarantee. Preserve the current `NOT_CHECKED` label and avoid presenting these deterministic cases as secret held-out evidence after publication.

## 5. Retest requirements and disposition

The corrected source should be frozen to a new SHA-256, preserve the initial snapshots and failed evidence, and pass all of the following before the final control receipt is published:

1. The supplied test suite, including regressions for the three findings and both mutable-buffer manifestations.
2. The independent 21-check script without changing the meaning of the original reproductions.
3. Identical public/oracle JSONL bytes under the unchanged frozen recipe; generator/scorer identity in the new manifest/report must reflect the corrected source.
4. Explicit retention of `INSTRUMENT_CONTROLS_ONLY` / model `NOT_RUN` status for all delivered symbolic checks.

Successful retesting will support **acceptance of this bounded synthetic evaluation instrument**. It will not support learned-model accuracy, native tokenizer eligibility, architecture improvement, contamination clearance, or secure execution custody.

## 6. Corrective retest and final disposition

The implementation owner repaired the three findings, retained the original generated dataset, and froze the corrected files. This reviewer copied the repaired files into `generation_independent/corrected_snapshot/`, inspected the source diff, ran the supplied suite, reran the unchanged independent reproductions, and checked the owner's existing dataset and final control receipt.

| Item | Independently checked final identity/result |
|---|---|
| Corrected instrument SHA-256 | `fb189cfdac616c3036f8bfe3e443bc39ff8bcaf746d0f09f145185d24eae2ff8` |
| Corrected test SHA-256 | `54f4505c20cc25d1a1a829df79a4a5e6a06072a5adf5e3a773e179c296012dd7` |
| Frozen protocol SHA-256 | Unchanged: `1d86c0b00d4c6db8974fd2bb3f739ba54809daedaa8a839109521c2184fd49b6` |
| Original generator provenance | Retained as `6f79b8f606bc7fa780c9724ad199a44628dfa886e762783292f4fba19ef14c73`, distinct from corrected scorer |
| Existing dataset manifest SHA-256 | `8cffe7ed4e5cd308cf675974cc30382462c82085720645502a0107c270b0706c` |
| Public JSONL | 480 records; 365,328 bytes; SHA-256 `bcacfd6c2efb7aa9e5eea4d28386dd059cd13d8e6edf37d59ccb41b166d71d71` |
| Private oracle JSONL | 480 records; 486,976 bytes; SHA-256 `7bf731d29ff6f8f19e7aea3e782b6ebc1347aee209b763fe5e25405444f5a3ab` |
| Supplied regression suite | **31/31 pass**, zero errors/failures/skips |
| Independent focused checks | **21/21 pass**, including all four originally failing manifestations |
| Delivered final control receipt SHA-256 | `44af92e3cf4938731559091368e1d205b838b25d450a2c840cdbed2ccdd917eb` |
| Learned-model execution | **NOT_RUN** |

### Closure evidence

- **D-IR-01 closed:** scoring now binds canonical current public/oracle rows and the manifest to their stored identities. The coherent in-memory world mutation is rejected before a score is produced.
- **D-IR-02 closed:** any LF in raw generated text requires a newline stop; newline output must contain exactly one terminal LF. All three EOS-labelled newline bypasses are rejected.
- **D-IR-03 closed:** preflight input IDs are frozen as an immutable tuple; receipt input/output lists are copied. An in-place encoder mutation becomes a retained error, and subsequent output-buffer reuse no longer changes the preceding receipt.

The existing public/oracle/manifest files are byte-identical to the files in the initial control receipt, and the public/oracle bytes match fresh deterministic regeneration under the frozen protocol. Metrics for all five symbolic controls are identical before and after repair. The corrections therefore repair custody and stopping enforcement without changing the evaluation instances or retroactively changing the measured symbolic-control outcomes.

The final evidence is preserved in `generation_independent/corrected_adversarial_results.json`, `generation_independent/corrective_retest_receipt.json`, and `generation_independent/corrected_sources.json`. Initial source snapshots and failed reproductions remain available. No instrument or protocol edits were made by this reviewer.

**Disposition:** suitable to include in the V6 architecture as a reviewed, locally frozen evaluation instrument. Native checkpoint binding, actual tokenizer eligibility, output decoding/backend checks, execution attestation, and pretraining-contamination review remain pending. Report its results as instrument controls and validation, not learned language-model performance.
