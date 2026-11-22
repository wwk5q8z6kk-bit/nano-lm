# Independent source review: V5 release gate and appended ledger

**Reviewer:** /root/v5_native_comparison  
**Final source-review verdict:** no remaining source-level blocker found for the scoped local release gate, including the bounded late-evidence amendment. Coordinator execution is still required.  
**Reviewed validator:** analysis/validate_release_v5.py  
**Final validator SHA256:** d5824d28cdc16c09ba865106a2c99639e1c446228b3a45bfdd9976ecfd724108  
**Earlier core-snapshot validator SHA256:** c405e4af09c862450b652757bf65e07f11709e4b9b3d2df120cbce980f6736ab

This review inspected source and existing local artifacts. It did **not** run the complete release gate, execute the native suite, retrain a model, access a corpus, or validate remote CI. Byte/hash comparisons below read preserved files only. The validator's eventual receipt is separate execution evidence.

## 1. Core command coverage and failure semantics

AST inspection of the core snapshot identified **23 distinct command names: 16 inherited V4 groups and seven V5 groups**. The finalized bounded addendum supplies one further group, making **24 total: 16 V4 and eight V5**, as reviewed in section 7. The legacy command list preserves package, mathematical, native training, causal correction, margin, bridge and independent oracle/replay checks. Its package-validator expectation is adapted only in memory to the expanded ledger inventory and restored in a finally block. The inherited V4 helper itself is byte-identical to the supplied baseline.

| V5 group | Required behavior in the reviewed source |
|---|---|
| optimizer_contracts | Eight unittest cases, exact terminal OK and no unexpected stdout |
| optimizer_reanalysis | Pinned research/factorial cutoffs, 53 verified inputs, exact finite verdicts, three correction arms and three exogenous streams |
| geometry_contracts | Seven passing tests, zero errors/failures and matching unittest stderr |
| geometry_diagnostics | Exact target-only cost, full minimizer segment, tie distinction, protected-margin formula and unattained strict infimum |
| causal_cache_contracts | Final reviewed source hashes, required grouped checks, 1,093 oracle prefixes, 40 future-suffix variants and synthetic scope |
| store_reanalysis | Pinned cutoff, ten verified sources, five-domain arithmetic, preserved primary rows and reproduced crop/identity counterexamples |
| native_update_contracts | Explicit fourteen-test adapter, frozen protocol, all three seeds/four arms, exact update accounting and equality to the complete saved receipt |

Each command is launched with bytecode writing disabled and optimization disabled so assertions remain active. A nonzero exit code, semantic failure, parsing failure or timeout prevents that group from passing. The native and unittest adapters reject skipped, failed or unexpectedly successful cases rather than equating an exit code alone with complete coverage.

The original commands_expected value was 23; the final value is 24 and agrees with the amended lists. This review establishes those source catalogs; a later change requires another review. The gate is a collection of concrete assertions and saved-result comparisons, not exhaustive verification of every scientific claim in the package.

## 2. Explicit native stdout contract

An initial review inference incorrectly suggested that direct execution of the frozen native suite emitted no JSON. Source tracing and the coordinator's already-captured stdout disproved that inference: the I/O test calls comparison.main with --out, and the inherited write_report both writes the file and unconditionally prints the report. The preserved stdout is exactly **8,453,400 bytes**, SHA256 **75df20be30342b30664da2f656026b9bc8eb2b57c9ca816c3267605fea8e9911**, and parses identically to the saved gzip receipt. There was no reproduced empty-stdout defect.

The validator owner made the contract explicit without changing frozen experiment files. Its -c adapter:

1. Redirects incidental stdout during test-module import and suite execution.
2. Loads NativeUpdateComparisonTests explicitly and requires exactly fourteen cases before running them.
3. Sends the unittest report to stderr.
4. Rejects failed, skipped, expected-failure and unexpected-success outcomes.
5. Serializes the cached setUpClass report once, after success, with nonfinite values forbidden.

The adapter does not call run_comparison again. The full three-seed comparison executes inside setUpClass; counterfactual and repeat fixtures remain additional validation work within the suite. The receipt's main-experiment counts must not be mistaken for the entire gate's compute.

The adapter was inspected and parsed but not executed by this reviewer. Its owner reported four lightweight fake-suite probes; those are owner checks rather than independent native execution evidence.

## 3. Full receipt, summary projection and compaction

Before compaction, native_saved_comparison decompresses the saved gzip, parses both full JSON objects with duplicate-key and nonfinite-value rejection, and requires equality between actual suite output and the complete saved object. It also verifies gzip filename, compressed/uncompressed byte counts and both hashes against the summary.

The summary contract distinguishes two kinds of completeness:

- Every full top-level field except runs must appear unchanged in the summary. Only full_receipt and per_seed are added.
- Each seed/arm view must retain six mandatory fields: branch checkpoint, final checkpoint, counts, correction metrics, new-fact acquisition and memory scope. Every projected field must exist and match in the full arm. Ordered seed identity and arm membership must agree.

Independent read-only comparison of the saved artifacts confirmed every required top-level value and all **12 arm views × six required fields**. The per-seed view intentionally omits detailed transactions, logits, phase snapshots and the common-prefix training trace; those remain in the full gzip. The gate does not claim that the compact view contains all those details.

Only after complete equality is checked does run_v5_command remove runs from the combined release receipt. It retains the comparison's status/checks, full gzip identity, uncompressed identity, summary identity, canonical parsed-output hash and compact result summaries. A mismatch fails the gate even if the compact summary alone looks plausible. The comparison is **exact parsed JSON equality**, not a requirement that current serialization whitespace or key order equal the saved raw bytes.

This compaction claim is appropriately scoped: when the comparison passes, the full saved object represents the full parsed suite output. It does not establish independent scientific replication, remote provenance or a second experiment.

## 4. Historical evidence and mutation checks

The supplied historical baseline used for this review is:

    /workspace/scratch/2a0f25a4f985/v5_review/baseline/architecture

The current machine ledger begins with the **exact complete bytes** of that baseline's 36-event ledger. The baseline ledger SHA256 is **9d0db714ba817d22304179e4cea613c1374c28261254d05063f411f1d268204f**. All **17 files** discovered under the baseline results directory have byte-identical current counterparts. The V4 validator helper is also byte-identical; its SHA256 is **0b62d25c70a9f68a00488126115d5815978c5989e4abdc160ee2170a5e9b2f4b**.

The reviewed gate checks:

- The complete machine-ledger baseline as a byte prefix.
- The separately supplied prior canonical ledger as a byte prefix, including its external append.
- Every result file discovered in the supplied prior package.
- Before/after package file inventory and bytes.
- Before/after canonical-ledger bytes.
- Before/after supplied baseline inventory and separate baseline-file bytes.

Separate-baseline checks reject comparison against the same current results location or the same current ledger path. Missing/empty historical results and empty prefix baselines fail their comparison. The inventory snapshots measure persistent file bytes and paths at their endpoints; they are not a tracing proof that no transient write occurred.

A new --out receipt is exclusively created only after these snapshots. The source inventory therefore cannot include the receipt's own final hash, which the scope note acknowledges. Existing output files and dangling symlinks are refused before execution.

The consolidated source check verifies 85 distinct local source-copy paths, fixed manifest cutoffs, SHA256, Git blob digest and byte length; source paths must remain within the package. These are local integrity checks against the supplied manifests. They do not authenticate a remote revision or establish that a research conclusion is true.

## 5. Nine ledger additions in the inspected pre-addendum snapshot

The ledger SHA256 at this snapshot review is **1a4f2750d855412afec1c2b47335948a4e9b29b0b0bd632317700509a052cdf0**. It contains 45 events for 39 distinct machine-recorded decisions. This is the inspected pre-addendum inventory, not a claim about the final pending release. Relative to the exact historical prefix, the additions are exactly:

| Event decision | Role and evidence scope |
|---|---|
| D-054 revision | Links the original bridge decision to the exploratory native comparison |
| D-055 | Narrows optimizer attribution using the five-seed factorial |
| D-056 | Separates detached correction fit from deployed shared-entity state |
| D-057 | Records the coordinate-scoped exact L1 trade-off and tie qualification |
| D-058 | Preserves qualified store evidence and historical invalidations |
| D-059 | Records the repaired causal comparator and its final fixture scope |
| D-060 | Retains mixed native outcomes, exposure distinctions and open transfer questions |
| D-061 | Proposes a resource frontier without claiming it was executed |
| D-062 | Records source-specific provenance and preservation duties |

D-054 names **evt-D-054-v4-20261121** as its previous event. Every new decision D-055 through D-062 appears once and has no fabricated prior event. All nine source-reference paths exist. All **26 evidence-artifact hash entries**, including repeated references across events, match current bytes.

In particular, D-059 points to the final cache implementation/test hashes and the final fixture receipt; D-060 points to the unchanged native implementation, protocol, tests, compressed full receipt and summary. The cache's grouped receipt is byte-identical across the last numerical repair because group names/counts did not change; final source/test hashes correctly distinguish the expanded assertions.

The new records explicitly describe inspected prior source commits separately from local artifact hashes. Their status, rationale and open questions do not promote the tiny native comparison to language efficacy, revive invalidated cache results, or treat a proposed quality-resource frontier as measured.

## 6. Frozen identities and remaining execution boundary

| Artifact | SHA256 |
|---|---|
| Native comparison implementation | 1540a68e526dad7097b2b5ee3d9cc5bb0e740049782dcae05a8575a95927a189 |
| Native comparison tests | d26087c32444b291eeec1636a0c81b9aff71acdc3aa36f9a8a828fbc8d0c938c |
| Native comparison protocol | 4cd9bdabedd2b507663f5d5d368dab34c037c2b078b386d872d9238b23f18425 |
| Final cache implementation | 8836cf8ab645628499a6b438963d798c6c064cf832bb67df8a42a8ac32325732 |
| Final cache tests | b22f37dc4fc9a9355683eec6d30f7b5a9471bc4b2a89a0178e2e3174c0abd23f |

The coordinator must run the reviewed gate after package edits stop and inspect the resulting PASS/FAIL receipt. This source review does not supply that result. The final gate remains local engineering verification with explicit synthetic scope, not remote CI, corpus evaluation, architecture superiority, mathematical proof-assistant certification or ongoing background execution.

## 7. Bounded late-evidence amendment and final closure

After the core snapshot review, the coordinator identified later cache/store evidence at source revision **c597d5fefdd378c3e5c66a51e7690cdb90853287**. The earlier 85-file manifest remains unchanged at SHA256 **d99569a08eb00ae9264f5575746c99724d6fc1f5d8b84a40347ede040507a804**. The five separately packaged late copies are the S/M cache results, M store result, investigation document and training source.

The reviewer independently checked all five addendum paths against declared byte lengths, Git blob hashes, SHA256 values and the full late commit. The core and addendum manifest paths form **90 distinct local copies**. The addendum's recorded underlying source-manifest digest matches that file's bytes.

The final validator at the hash stated above preserves the native adapter and full-receipt comparison path. Its narrow extension:

- Adds late_store_cache_reanalysis as the 24th distinct command group.
- Requires the late output to equal its saved receipt as parsed JSON.
- Checks five late sources, ten prior sources, fifteen reconciled rows, eight unchanged model/evaluation ASTs, two unchanged exchange-script identities and the three source-derived parameter counts.
- Requires the cache stopping branch to retain its unresolved control qualifications, the registered L-sized store gate to remain pending, and the absence of a repaired-sampler training result within this bounded addendum.
- Verifies the separate five-copy manifest at the fixed late cutoff, while pinning the earlier 85-copy manifest's exact bytes.
- Changes the ledger inventory to 46 events/39 machine decisions/62 human rows, checks the complete earlier 45-event prefix by hash and byte length, and requires the final D-058 event to link its prior event.

The finalized ledger contains **46 events**, **39 distinct machine decisions** and **62 human decision rows**. Its SHA256 is **ddb31fd6313090fc1e61f7407a3e67bed890c30683eeb8e8e4b3d76e71e6d356**. The complete earlier 45-event prefix remains exactly 66,297 bytes with SHA256 **1a4f2750d855412afec1c2b47335948a4e9b29b0b0bd632317700509a052cdf0**; the original 36-event historical prefix also remains byte-exact.

The new event **evt-D-058-v5-late-20261122**, timestamped 2026-11-22T02:44:16Z, links **evt-D-058-v5-20261122**. All four new artifact entries match the final local files:

| Late event artifact | SHA256 |
|---|---|
| analysis/reconcile_store_cache_late.py | 78c516e2eac1715d5a34a61ee0708b5597a33dc2e4dc1d6c2e4db4d1a823e158 |
| results/store_cache_late_reanalysis_v5.json | fa75a633bab73d3b1e701f79d6a053867711157b0f9290b4549c4a77928d0805 |
| evidence/source_manifest_v5_addendum.json | 1c5de7e54d026441c6e514b0dea0405cd830b8d5d8aff712576e3eb80e6cb18a |
| evidence/upstream_v5_store_late/source_manifest.json | 2f6a0b5c97eacdd1e44e877b4559853212aee7aa43d540dbd9fb81b112f9ddea |

The new rationale correctly narrows the result to a registered numerical stopping branch at the reported M scale while retaining baseline, control and provenance qualifications. It does not reinterpret the still-pending L store gate, revive the invalidated earlier 0.62 result, or treat a source change as proof of completed repaired-sampler training.

**The late source/hash amendment is closed.** These are source and existing-artifact findings, not execution of the final 24-group gate. The coordinator must still obtain and inspect that execution receipt after all package edits stop.
