# 06 — V3 validation, readiness and unresolved work

**Review date:** 2026-11-21, America/New_York. **Scope:** corrected architecture specification, conditional mathematical arguments, source reanalysis, a tiny untrained forward reference, and maintainable research records. This release adds no neural training run, architecture victory, physical energy measurement or paid compute.

**Readiness:** ready for scientific and implementation review. The forward reference establishes a small executable contract; a trainable native baseline, controlled learning experiments and model-scale claims remain separate gates. APS remains an unselected hypothesis within Nano LM.

## 1. Final validation summary

The behavioral-suite receipt is `results/validation_v3_final.json`; the later document/ledger/source integration check is `results/integration_validation_v3.json`. Executable check and prototype hashes are unchanged between them. It records actual runtime versions, commands, outputs, source hashes, evidence-copy verification and before/after package hashes. Independent notes under `reviews/` describe earlier audit points; this report and the final receipt supersede their snapshot counts.

| Check | Final result | Supported conclusion | Limit |
|---|---|---|---|
| Mathematical examples, `checks/repo_lightweight_checks.py` | **12/12 passed** | Deterministic examples for coherent transport, actual optimizer displacement, nonlinear remainder, versioning and compute-policy counterexamples | Examples do not constitute formal verification or a neural result |
| Original NumPy suite, `checks/formal_sanity.py` | **5/5 passed** | Small algebraic numerical checks execute on the retrieved live source | No claim that every theorem assumption holds in a trained model |
| Package regressions, `checks/test_research_package.py` | **35/35 passed** | Valid extensions are accepted; corrupt histories, malformed inputs, forbidden overwrites and a deliberately broken curvature check are rejected | Does not authenticate remote evidence, authorship or scientific truth |
| Forward reference, `prototype/test_native_reference.py` | **13/13 passed** | Dense/shared-block token flow, logical caches, causal/reset behavior, exact inventories and input contracts work in tested tiny configurations | Forward-only NumPy float64; no gradients, optimizer, training or serving performance |
| Transport-anchor diagnostic | **Five seeds reproduced the stored summary** | A constructed 16-dimensional linear identification example is reproducible | Known rotations and full-span anchors do not establish learned LM migration |
| Event and document validation | **PASS: 27 events, 23 machine-covered decisions, 14 Markdown files, ten evidence-hash checks** | Schema, revision pointers, local links/headings and the pinned historical prefix are consistent | Ten artifact-reference checks do not mean ten independent scientific studies |
| Full previous-ledger comparison | **PASS against the exact PR #195 ledger at `3f9ef1a…`** | The complete prior 7,421-byte, ten-event log is an unchanged prefix | Future releases must supply the entire then-current prior log again |
| Frozen source copies | **Twelve files match SHA-256 and Git blob identities** | Ten completed-evidence/source copies match `2ef9ce5…`; two late protocol copies match `8d3b03e…` | A matching source file does not independently reproduce original training |
| Reanalysis and control-flow diagnostics | **Stored result reproduced** | Six-arm summaries recomputed; random-stream coupling and solver-status conflation demonstrated | No new model optimization; historical solver failures cannot be classified retrospectively |
| Default-script write behavior | **Package paths and bytes unchanged before/after** | Default checks and reanalysis do not overwrite packaged evidence | Direct filesystem comparison, not an inference from permission bits |

Tested environment: Python 3.12.14, NumPy 2.3.5 and jsonschema 4.26.0. Exact versions are in the receipt. Local diagnostic runtimes are not benchmarks of an owner's device, GPU/TPU, compiler or inference stack. Plot generation additionally uses Matplotlib, separately from validation dependencies.

## 2. Corrections to the validation record

### Earlier package and live branch were different artifacts

The downloaded V2 package and live PR #195 were not byte-identical. The live branch had already fixed the Markdown-link regex before this continuation. That repair is retained and is not claimed as a new discovery here. The old downloaded package's prose reporting “9/9” checks and ten ledger events is not a receipt for the revised source. The final report pins the actual files checked.

The original ten live ledger events are preserved byte-for-byte, with SHA-256 `aa55fc010e06563188c6d9b8899e58b20223807d8ecae1ab6d1259c85b7eb3a0`. Historical numerical JSON files remain unchanged. `results/v2_artifact_manifest.json` is retained as a **historical V2 inventory**, not a V3 integrity certificate. New source hashes appear in the final receipt and delivery manifest.

### A passing initial forward suite did not cover every contract

The initial 12-case receipt is preserved at `results/native_reference_v3.json`. Independent review then reproduced three real defects, repaired them, and added a regression group. The current receipt is [the final 13-case result](results/native_reference_v3_final.json).

| Defect found after the initial pass | Reproduced behavior | Current repair |
|---|---|---|
| Parameter replacement did not invalidate checkpoint identity | An old cache was accepted after replacing a K matrix; logits differed from fresh forward evaluation by approximately 0.113 | Verify configuration and parameter names, shapes, dtypes and bytes at state boundaries; reject a changed definition |
| Unsigned logits overflowed during stable-softmax subtraction | `uint8` logits `[0,255]`, target 0, produced about 0.3135 nats instead of 255 | Require floating logits and check a known floating-point answer |
| Fractional prefix length produced fractional cache accounting | Prefix length 1.5 returned 1,536.0 bytes | Require integer, non-boolean dimensions and valid bounds |

Checksum verification deliberately scans parameter bytes on calls in this tiny reference. Its **O(parameter bytes)** cost makes this implementation unsuitable for a latency comparison. Production execution needs immutable, versioned model objects and controlled mutation/loading boundaries; that optimization has not been implemented here. See [implementation contract 07](07_IMPLEMENTATION_CONTRACT.md) and [independent code review](reviews/native_reference_review.md).

The final dense and shared-block tests agree between full-prefix and incremental evaluation to maximum absolute logit errors of approximately `6.14e-16` and `1.42e-15`, respectively. This is agreement for these configurations, not a universal error bound.

### Ledger validation now supports actual maintenance

The earlier validator conflated a decision with an event and assumed a fixed set of ten decisions. New records can introduce IDs or revise a decision through a unique event ID and a pointer to its latest previous event. Full JSON Schema validation includes UTC calendar checks, duplicate-key rejection, local evidence hashes and structured failures for missing or malformed inputs.

The register has **46 human decision IDs**, with **27 events covering 23 decisions** in the machine log. D-001–D-023 remain historical human rows without invented timestamps or provenance. D-026, D-032, D-044 and D-045 have follow-up events. D-044's final event corrects a broad `peer_reviewed` label to an engineering review of mixed primary evidence; preprints and author repositories retain their publication status. D-045 records the final prototype repairs; D-046 records the late protocol refresh without declaring a result.

Schema validity cannot establish that an author exists, a theorem is correct, an experiment was authorized or a source supports its stated claim. Literature audit, numerical review and repository inspection supply distinct evidence.

## 3. Reproduce the checks

From the architecture directory, install the declared dependencies in an appropriate environment and run:

```bash
python3 -B checks/test_research_package.py
python3 -B checks/repo_lightweight_checks.py
python3 -B checks/formal_sanity.py
python3 -B checks/transport_anchor_pilot.py
python3 -B prototype/test_native_reference.py
python3 -B analysis/reconcile_repository_evidence.py
python3 -B checks/validate_research_package.py --baseline-ledger /absolute/path/to/previous-ledger.jsonl
```

The combined receipt runner is `analysis/validate_release.py`. It is read-only by default and accepts `--baseline-ledger` for the entire prior committed event file. Optional `--baseline-package` verifies two preserved numerical artifacts against an independently recovered prior package. Use `--out /new/path.json` only for a **new** receipt; existing files are rejected.

```bash
python3 -B analysis/validate_release.py --baseline-ledger /absolute/path/to/previous-ledger.jsonl --out /new/path/validation.json
```

Without `--baseline-ledger`, a fresh run reports full-history comparison as `NOT_CHECKED`; the permanent legacy-prefix pin is still checked. Omitting that argument cannot provide the same full-history evidence as this release's recorded run. The suite uses temporary corrupt fixtures and independent mutation checks; it does not modify canonical evidence.

## 4. What has been closed, and what has not

| Area | Closed within this revision | Still open | Required next evidence |
|---|---|---|---|
| Architecture specification | Token shift, masks, reset/state ownership, logical depth, parameter counts and forward interfaces explicit | Training dynamics, precision, fused execution and useful scale | Reproducible native training baseline and held-out bytes |
| Mathematical scope | Functional conjugacy, finite-step remainder, fixed-bit storage, bitemporal cutoff and finite-budget conditions clarified | Whether learned transformations satisfy those assumptions efficiently | Non-oracle fitting, full dependent-state migration and cost measurements |
| Correction research | Completed pilot incorporated; locality/durability and cohort definitions separated | Causal advantage of proposed mechanisms | G03b common event manifests, paired edit survival, targeted replay and true residuals |
| Optimizer mechanism | Raw-gradient projection error identified; complete realized displacement specified | Address congestion versus optimizer-history mechanisms at useful scale | G02 address-policy × optimizer-history factorial with matched acquisition/retention |
| Store augmentation | Recorded BPB and 1,022,224,980-byte store accounted for | Generality, training exposure controls and system efficiency | Whole-document exclusion, fresh seeds and isolated runtime |
| Transfer and procedures | Transfer, retention, plasticity and sample efficiency distinguished | Reusable operations beyond supplied demonstrations | Same-learner prior-history controls and fresh-family generalization |
| Literature | 36 primary-source identities and narrow claims audited with explicit reading depth | Replications, all proofs/methods, full data and model-license audits | Version-pinned comparators and exact dependency inventory |
| Decision history | Extensible events, explicit revisions, legacy hashes and prior-log comparison | Authenticity and scientific soundness cannot be inferred from JSON | Human review of each event and source |
| Efficiency and scale | Unique parameters separated from logical compute and runtime state | Joules, throughput, cost, scaling fit and strict total-parameter goal | Instrumented equal-device trials and separately authorized resources |

## 5. Execution order and stopping gates

1. **Recover and reproduce the native baseline.** Pin code, tokenizer, data manifests, split lineage, optimizer state and evaluation. The untrained NumPy model is not a replacement for that work.
2. **Run the repaired small mechanism tests when their protocol is frozen.** G02 isolates address/optimizer interactions; G03b isolates editing, replay, drift and survival of the same correction versions. These protocols are specified, not already preregistered or executed.
3. **Bridge a supported mechanism into a trainable native model.** Compare against tuned dense and targeted replay using equal data, measured compute, retention and acquisition guardrails.
4. **Test transport, shared computation and adaptive depth separately.** Retain fixed logical depth until a causal cache design for variable depth is implemented. Count migrations, controllers, plugins, targets and memory.
5. **Scale after evidence and resource gates are met.** A smaller parameter inventory or one successful edit does not justify a larger run. This continuation grants no new paid-compute authorization.

Stop or redesign an arm when acquisition/retention guardrails fail, a control reveals leakage or unfair exposure, state migration breaks its prediction tolerance, or measured cost removes the claimed advantage. Archive the result and append a reasoned ledger event; do not erase the failed hypothesis.

## 6. Publication and remaining access limits

The revision belongs in draft [PR #195](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/195), targeted at `research/ontogenesis-phase1`. Completed-pilot reanalysis is pinned to `2ef9ce5827cb8ea1fb6c8946df32fbcf5e06dec0`; the final integration base is `8d3b03e08a5c14d2e9ab330ac8d5006e87e61613`; prior architecture history is `3f9ef1a1b5643acda9ee4cc2a780d4fdcbf4bce8`. Carry that research base into the review branch while preserving its experiment files and raw results. Publication is established only by a successful write and subsequent read of the resulting Git objects. The delivery receipt records the result separately, avoiding a future self-referential commit hash in this document.

The full repository test suite and remote CI were not run as part of these local checks. Owner-local worktrees, uncommitted changes, live processes and hardware state were not inspected. Dataset rights, private inputs, current quotas and prices were not independently audited here. The latest inspected nested-protocol note at 8d3b03e supersedes the earlier withdrawal: it again specifies cloud host CPUs after a stopped laptop-GPU attempt, with device-consistent control reruns. This is a source-reported plan, not a verified runtime outcome. The new margin protocol and nested protocol supply no completed result at the fixed integration cutoff.

**Current scientific conclusion:** the documents, implementation contract and diagnostic evidence are more precise and testable. There is still no proof or experiment establishing a winning Nano LM backbone, native lifelong learning, frontier capability below four billion total parameters, or a deployment-scale cost advantage.
