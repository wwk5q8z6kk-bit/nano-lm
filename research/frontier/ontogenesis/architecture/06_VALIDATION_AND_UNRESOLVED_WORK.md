# 06 — V4 validation, readiness and unresolved work

**V5 current validation notice.** The new combined gate is [validate_release_v5.py](analysis/validate_release_v5.py); its final receipt is written as results/validation_v5_final.json in the [results directory](results/). Historical V3/V4 receipts and all 17 previous result files remain unchanged. The24-group current gate checks the entire36-event V4 machine prefix and the45-event core V5 prefix and the later PR canonical-ledger prefix, including the external contribution. Current commands and remaining research appear at the end of this chapter; old receipts retain their original counts and scope.

**Review date:** November 21, 2026, America/New_York. **Scope:** architecture and conditional proofs, frozen-source reanalysis, tiny native training and correction integration, independent measurement review, and append-only research records. Local synthetic training is now executed. No corpus-scale training, new held-out language benchmark, device efficiency result or paid compute is part of this release.

**Readiness:** the tiny implementation and measurement interfaces are ready for scientific and engineering review. They do not establish an effective general-purpose continual learner or select an architecture. APS remains an unselected Nano LM hypothesis.

## 1. Current validation and receipts

The combined V4 runner is [analysis/validate_release_v4.py](analysis/validate_release_v4.py). Its final receipt is saved as validation_v4_final.json in the [results directory](results/), recording commands, actual runtime versions, semantic outcomes, source hashes, before/after immutability, frozen-source identities and full V3-history comparisons. No receipt is a remote-CI or scientific-authenticity certificate.

| Check | Executed outcome | What it establishes | Boundary |
|---|---|---|---|
| Existing package regressions |35/35|Extensible ledger/schema, corrupt fixtures, links, no-overwrite and numerical-check regressions|Structural validity does not authenticate claims|
| Existing mathematical examples |12/12|Transport, realized-step, remainder, temporal and stopping counterexamples|Finite examples, not proof automation|
| Original NumPy examples |5/5|Small numerical identities rerun|No neural result|
| Original forward reference |13/13|Tiny dense/shared causal and cache contracts|Unoptimized float64 reference|
| Native training |17/17 groups;62 finite-difference directions over31 named tensors|Analytical backward, shared/tied accumulation, actual AdamW displacement, snapshots and tested resume|Synthetic training data only|
| Correction instrument |26/26|Event identity, causal views, revision policy, missingness and deployed-value checks|Requires correct integration and actual experiment data|
| Independent cohort oracle |24,000 rows across320 histories|Separate forward scan agrees on both censor policies and historical cutoffs|Synthetic instrument verification|
| Bilinear durability suite |9/9|Joint-drift identities, finite telescoping, counterexamples and failure paths|Float64 diagnostics, not interval certificates|
| Native bridge |9/9|Real updates and predictions are wired into event/state/horizon receipts|No accuracy threshold or efficacy comparison|
| Independent native replay |16 updates,16 event-boundary states,176 displacement arrays,23 predictions|Direct loss and original-forward comparisons reproduce the bridge|Same tested runtime; no performance claim|
| Transport-anchor diagnostic |Five stored seeds reproduced|Preserved toy linear identification summary|Known rotations/full anchors do not establish learned LM migration|
| Source reanalyses |Existing and new stored summaries reproduced|Arithmetic, event reconstruction and constructed LP example|No original model training replay|
| Frozen evidence |21 source/protocol/result copies match source hashes and Git blob identities|12 V3 copies preserved plus9 new V4 copies|Matching files do not recover unrecorded runtime metadata|
| History |Full27-event V3 machine log and complete V3 canonical ledger preserved as byte prefixes|Nine new events revise D-046 and add D-047–D-054|Human review still decides evidence meaning|
| Default commands |Package files unchanged before/after execution|Checks do not overwrite evidence|Output creation is separately explicit|

The current event register has **54 human IDs and36 machine events covering31 decisions**. The original ten-event pin is still retained. The old D-001–D-023 human records were not backfilled with fictitious machine timestamps. All V3 result files are compared byte-for-byte with the frozen V3 baseline.

The recorded runtime is Python3.12.14, NumPy2.3.5, SciPy1.17.0 and jsonschema4.26.0; the final receipt records actual versions. Local diagnostic duration is not a hardware benchmark. Optional charts require Matplotlib and were visually inspected separately.

### New executed receipts

| Artifact | Scope |
|---|---|
| [trainable_native_v4.json](results/trainable_native_v4.json) |17 groups and the dense/shared60-step cyclic-token fixtures|
| [correction_protocol_v4.json](results/correction_protocol_v4.json) |26 measurement/numerical-contract cases|
| [native_correction_smoke_v4.json](results/native_correction_smoke_v4.json) |Eight events,16 actual updates and16 scored target exposures; real logits/state identities and all denominators|
| [margin_reanalysis_v4.json](results/margin_reanalysis_v4.json) |Five committed arms, source identities, conditional stream reconstruction and actual-source LP example|
| [margin_durability_v4.json](results/margin_durability_v4.json) |Constructed drift counterexamples and sufficient-bound diagnostics|
| [trainable_independent_verification_v4.json](results/trainable_independent_verification_v4.json) |Independent48 storage-overlap atomic rejections and the declared F-order precision boundary|
| [literature_audit_v4.json](results/literature_audit_v4.json) |40 primary identities; source-specific inspection depth, no empirical replication|

The native bridge's untuned fixture records one of five assignments correct immediately after its own write. That20% is retained in the receipt rather than filtered away. Its acceptance criterion is correct execution and accounting, not successful correction. The separate cyclic-token fixture reports training CE reduction; neither fixture is a held-out or durability result.

## 2. V4 defects actually reproduced and repaired

| Issue | Consequence before repair | Resolution and reviewed limit |
|---|---|---|
| Very large finite update with overflowing diagnostic norm | Potential state mutation before receipt failure | Validate finite candidate and receipt values before all commits |
| Named parameters share memory | Sequential writes invalidate actual-displacement receipts and resume semantics | Pairwise disjoint storage |
| One strided parameter overlaps itself | Pairwise tests alone miss invalid storage | C- or F-contiguous arrays required; noncontiguous layouts rejected |
| Solver and phase-end accuracy treated as the same endpoint | Historical errors could be attributed to the wrong stage | Separate per-write, remaining-phase and subsequent-learning observations |
| Same-value assignments reset revision age | Apparent cohort changes can be confused with semantic learning | Every assignment retained; explicit revision/value-change censor policy |
| Large initial margin treated as a guarantee | Future decoder/address drift remains unconstrained | Conditional cumulative-drift proof, counterexamples and inconclusive-bound status |

The independent reviewer tested48 atomic rejections; the published behavioral suite contains42 related rejection cases. Those are different scopes, not conflicting counts. The default C-order resume fixture matches bitwise in the recorded runtime. F-order reload can change several diagnostic norm values by a few ulps; cross-layout/cross-environment universal bitwise equality is not claimed. Checkpoint writes are exclusive but not crash-atomic.

Independent records: [training review](reviews/trainable_native_review.md), [measurement review](reviews/correction_protocol_review.md), [native bridge review](reviews/native_correction_smoke_review.md), [source review](reviews/margin_evidence_review.md), and [theorem review](reviews/margin_durability_review.md).

## 3. Reproduction

Run from the architecture directory with [declared dependencies](checks/requirements-checks.txt). Default checks print results without writing inside the package. The combined runner invokes all relevant old/new suites and independent oracles.

```bash
python3 -B analysis/validate_release_v4.py \
  --baseline-ledger /absolute/path/to/V3/ledger_events.jsonl \
  --baseline-package /absolute/path/to/V3/architecture \
  --baseline-canonical /absolute/path/to/V3/LEDGER.md \
  --out /new/path/validation.json
```

Full-history preservation needs the actual previous artifacts. Omitting a baseline records NOT_CHECKED for that comparison; the permanent original-ten-event pin alone does not establish preservation of the whole V3 log. Existing output files are refused.

For focused reproduction:

```bash
python3 -B prototype/test_trainable_native.py
python3 -B experiments/test_correction_protocol.py
python3 -B checks/review_correction_oracle.py
python3 -B checks/test_margin_durability.py
python3 -B experiments/test_native_correction_smoke.py
python3 -B checks/review_native_bridge_replay.py
python3 -B analysis/reconcile_margin_results.py \
  --source-dir evidence/upstream_v4/research/frontier/ontogenesis \
  --prior-results evidence/upstream/results/correction_interference/constrained.json
```

The V3 runner and V3 result receipts remain preserved as historical artifacts. They describe the earlier forward-only package and do not replace this V4 validation scope. Tests use temporary files for corrupt/checkpoint fixtures. Mathematical proofs are inspected arguments; numerical checks do not turn them into machine-verified proofs.

## 4. What remains open

| Area | Implemented or resolved | Evidence still required |
|---|---|---|
| Native core | Forward/backward, shared/tied gradients, AdamW, snapshots and checkpoints | Identified tokenizer/data/checkpoint baseline and clean held-out evaluation|
| Controlled corrections | Complete event instrument and tiny native bridge | Scientific G03b comparison with fresh seeds, fixed budgets and guardrails|
| Retention theory | Finite-horizon sufficient bound with assumptions and counterexamples | Useful native drift/curvature estimates and a beneficial control mechanism|
| Representation migration | Coherent-transform conditions and toy diagnostics | Non-oracle map fitting, all-state migration and net benefit over replay/freezing|
| Transfer | Distinct acquisition/retention/forward-transfer endpoints | Fresh task families and learning curves beyond context/exposure advantages|
| Efficiency | Logical compute, unique weights and state costs distinguished | Isolated measured runtime, memory, energy and full preprocessing/update cost|
| Literature |40 identities and inspection depth explicit | Selected implementation/proof replications, complete license/data audits|
| Scale and deployment | Proposed gates and resource accounting | Valid data/model lineage, measured kernels and a concrete scoped experiment|

The next substantive research step is to freeze a native-data protocol using the implemented instrument, with same manifests, explicit primary horizon, per-seed contrasts, cost accounting and unrelated-quality/new-learning guardrails. A mechanism that preserves old facts by blocking useful new learning has not met the goal. Scale decisions follow that evidence.

## 5. Publication and preserved source boundary

The update belongs in existing draft [PR195](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/195). Prior published architecture head:176ece70b17caffd71e2f0f522bb9462ce5f76a8. Current fixed research cutoff:9d88a2c20233f42b4a49c4430264b20f842895ad. Preserve all research-base files outside the architecture directory and the append-only canonical ledger. Publication requires a successful branch update and independent readback of Git objects; the export's publication receipt records the new commit to avoid self-reference here.

This continuation did not run full-repository CI, inspect owner-local worktrees or live processes, launch paid resources or merge the PR. The new margin results exist at the inspected cutoff; newly committed TPU controller changes do not supply completed nested/native results. Old raw source interpretations are retained and corrected in the review, not rewritten as if the original authors had recorded the later qualification.

## 6. Historical V2/V3 validation corrections retained

The following earlier correction record remains relevant. Its “initial” and “final” counts refer to the V3 forward-reference review. The complete earlier chapter is available at the [immutable V3 commit](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/176ece70b17caffd71e2f0f522bb9462ce5f76a8/research/frontier/ontogenesis/architecture/06_VALIDATION_AND_UNRESOLVED_WORK.md).


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



## V5 reproduction, exact history and limits

Run from the architecture directory with Python3 and the existing requirements file. The baseline paths must identify separately preserved preceding material, not the current files.

```sh
python3 -B analysis/validate_release_v5.py \
  --baseline-package /absolute/path/to/preserved-v4-architecture \
  --baseline-ledger /absolute/path/to/preserved-v4-architecture/ledger_events.jsonl \
  --baseline-canonical /absolute/path/to/canonical-ledger-at-0a1d11f.md \
  --out /absolute/path/to/new-validation-receipt.json
```

The explicit output must not already exist. Omitting it prints a new check result without changing package files. The command executes bounded synthetic CPU training/checks, not corpus or remote training.

| Current check area | Cases or invariant |
|---|---|
| Existing contracts | All16 V4 command groups retained, with current ledger-count expectations only |
| Late source addendum | Five new copies,15rounded rows,8unchanged model/evaluation ASTs and scale-specific qualified gates |
| Optimizer reanalysis |8 focused tests,53 exact inputs, raw phase-log means and literal finite predicates |
| Edit geometry |7 cases plus exact rational diagnostics; independent numeric LP cross-check |
| Causal cache |8 grouped checks,1,093 oracle prefixes,40 suffix variants, reviewed input and subnormal-mixing cases |
| Store/cache source audit |10 exact copies, decimal arithmetic and executed information-boundary counterexamples |
| Native comparison |14 behavioral tests; exact parsed stdout equality with losslessly compressed full receipt |
| Human/machine history |62 human IDs;46 events covering39 decisions;36-event prior prefix unchanged |
| Canonical history |Entire current PR prefix retained, including the external43-line append |
| Frozen sources |85core plus5late copies;90total, including the original21-copy manifest |
| Historical outputs |All17 prior V4 result files preserved byte-for-byte |
| Mutation contract |Package, canonical ledger and supplied baselines match before/after; new receipt created afterward |

The native suite itself emits the full comparison receipt. The gate compares it with the gzip artifact and every projected field in the readable summary, then stores a compact command summary and canonical parsed-JSON digest. It does not run a separate additional comparison merely to populate the final receipt. The full event, prediction, checkpoint and exposure records remain in the gzip file; compaction of the gate receipt does not discard experimental data.

The first cache fixture summary and the final repaired-cache summary have identical named-group results; both remain preserved. Final code/test hashes and the independent review distinguish the expanded numerical assertions. The old summary is not silently rewritten to imply it checked code that did not yet exist.

The gate verifies local reproducibility and package contracts, not an external lab replication, a formal proof-assistant certificate, full-repository CI or architecture superiority. Exact output equality is scoped to the observed Python/NumPy environment. The elementary mathematical proofs have separate assumptions and independent review notes.

### Work still unresolved after V5

The actual language evaluator still needs equal prefix/reset information, raw byte/crop accounting and checkpoint/data identities. The40M cache has met the numerical stopping condition in the late reported rows while full control clearance remains open; the registered101M/L store gate still lacks its required result. Native language-quality and long-term continual-learning claims require meaningful held-out tasks, sufficient eligible cohorts, independent tuning/evaluation and full cost. The tiny run shows why favorable acquisition alone cannot close a correction-durability gate.

The source audits do not recreate historical Torch execution order or recover missing model/corpus bytes. The parallel contributor's V3 subset rerun remains that contributor's attributed historical check, not an independent rerun of the complete V5 gate.
