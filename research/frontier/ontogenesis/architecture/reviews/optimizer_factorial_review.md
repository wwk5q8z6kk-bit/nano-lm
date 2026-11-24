# V5 optimizer controls and address × optimizer factorial review

**Review date:** November 21, 2026, America/New_York. **Status:** source-and-result reconciliation complete at the fixed cutoffs below. No model training, paid compute, remote write, historical result modification, or independent empirical replication was performed.

## 1. Decision summary

The correction controls show poor neighboring-fact accuracy under three particular unconstrained correction recipes, including fresh-state Adam and plain gradient descent. O1 is satisfied in all nine arm–seed records. This supports the narrower conclusion that stale state in the local correction optimizer is not required for the observed failure. It does not establish that optimization is causally irrelevant, that every optimizer fails, or that missing constraints are the unique cause.

The separate address × optimizer experiment shows a large conditional effect of resetting Adam on acquisition with the original addresses, alongside an acquisition ceiling under both hashed-address Adam variants. Resetting increases original-address late acquisition from 2.416667% to 63.500000%. The conditional recovered fraction of the original-to-hash gap is 62.596072%, rather than exactly two thirds or a unique decomposition of causes. F1, F2 and F4 hold; F3 and F6 fail; F5 is partial under its registered rules. The study fixes nominal memory size and changes addressing and optimizer recipes. It is not a capacity sweep or a language-model result. [S1–S6]

The architecture should retain two distinct mechanisms in its ledger: within-entity correction interference, and phase-to-phase plasticity of a fixed-address value table. Neither study settles the general native language-model design.

## 2. Immutable evidence and registration order

| Item | Immutable identity | Verified interpretation |
|---|---|---|
| Correction research snapshot | 0afd5629990b4e442fb3aa8f7c700b030d8faede | Fixed V5 cutoff |
| Correction source registration | 27cc27c80d969f765785a5c5da5ca8d3870278b6 | At 01:32:08 UTC, adds only correction_optimizer.py; no optimizer result file |
| First correction results | f1a66b00f4d6e584518b0fd5825c3b249d8742c9 | At 01:45:05 UTC, directly follows registration; adds adam_200 and gd_stop |
| Remaining correction result | 7351fb137985d3cfefc71b0eb96e850cab53ec81 | At 01:54:49 UTC, adds adam_stop |
| Factorial protocol | b82d0f018ebcbe1e6604afc129731d96774d0f69 | At 00:36:45 UTC, adds protocol and experiment source |
| Factorial result snapshot | 767857e72bfa51e3fa6f32d047507aabef077fde | At 02:05:38 UTC, direct child of protocol commit; adds results, scorer and driver |
| Additive architecture-ledger reference | 0a1d11f19b91e3d3efb2ab5cad4cbad617a56971 | Separately supplied integration reference; not used as raw empirical data |

All commit dates in the table are 2026-11-22 UTC. The factorial prose says written/run 2026-11-21; this can reflect local dating, but the audit uses the recorded UTC commit metadata. Source and protocol blobs are unchanged between their registration and the fixed result snapshots. Git order establishes source-before-result publication, not an independent execution-start timestamp, complete run provenance, or an absence of unrecorded exploratory work. Commit titles were not used as evidence of registration. [S1–S6]

The frozen evidence directory contains 53 exact Git files: correction source, dependencies, investigation narrative and three raw result arrays; factorial protocol, source, dependencies, scorer, narrative, all 30 confirmatory JSON runs, all six development JSON runs, development selection, driver and two driver logs. Every file's Git blob SHA-1, SHA-256, byte count, immutable commit and source URL are in the manifest.

Key blobs:

| Source | Git blob |
|---|---|
| correction_optimizer.py | bcf329aec01b1a659efe6127f87258c7c9a38ad0 |
| optimizer-adam_200.json | a1c38dd37a1ebb00a95a06119428178f4f593caa |
| optimizer-adam_stop.json | 293113310ccbe8e9a306b0acfd29bb52317c1267 |
| optimizer-gd_stop.json | 6f324702a00ddd195f235312e026fb720b5c50c3 |
| PROTOCOL-MEMSAT-FACTORIAL.md | 1a6fca6467ac212d22d7262c2fa2c9bfd7ba4cb7 |
| memory_saturation_factorial.py | c6a22b62097e7f55dd8355f54f2845822d0c592b |
| memory_saturation_factorial_score.py | 2e863d6c4636c96e89d12acd5f23cd0d60e16446 |
| RESULTS-MEMSAT-FACTORIAL.md | 7a4731aa23deaad7379cd7fde91e26436ddb7ee8 |

## 3. Correction-optimizer controls

The local correction decoder is frozen while detached address candidates are optimized. The source uses 40 phases, 64 new entities per phase, 16 selected correction facts per noninitial phase, address dimension 24, and seeds 8001–8003. Each phase first trains the base model on its new facts. Then the correction routine operates on the selected old facts. All correction arms initialize a fresh local optimizer. Gradient descent uses learning rate 0.2; Adam uses 0.003. The fixed Adam arm takes 200 updates; the stopped arms have a 5,000-iteration cap and attempt to stop each candidate when its new value leads by at least one logit. [S1, S3]

### Recomputed summary

Values below are equal-seed means of the committed four-decimal accuracy summaries. Extra displayed digits describe arithmetic on those saved summaries, not additional measurement precision.

| Recipe | Corrected now, mean [range] | Neighbor accuracy now, mean [range] | Old regular facts on corrected entities | Old corrections |
|---|---|---|---:|---:|
| Fresh Adam, 200 updates | 47.863333% [46.47, 49.84] | 62.530000% [58.23, 68.60] | 60.893333% | 38.946667% |
| Fresh Adam, candidate stop | 99.040000% [98.72, 99.20] | 39.836667% [37.95, 41.75] | 39.060000% | 47.380000% |
| Plain GD, candidate stop | 99.360000% [99.04, 99.68] | 14.333333% [13.06, 15.54] | 11.963333% | 65.320000% |

O1 requires collateral_now ≤ 0.85 for all three recipes; all three seeds in every recipe satisfy it. No record reaches the rival-trigger threshold of 0.95. There is no registered significance test here, and the nine records are three recipes on three shared seeds, not nine independent trials. The source does not define what every possible intermediate outcome between its two thresholds would establish; that ambiguity does not affect these observed values. [S1, S2]

### Denominators and update counts

| Quantity | Definition / reconstructed counts |
|---|---|
| corrected_now | Mean of 39 phase-end accuracies, each on 16 selected facts; equivalent to 624 event outcomes because each phase has the same denominator |
| collateral_now | Mean of 39 accuracies on touched entities' shown, never-corrected relations; variable denominators, not one pooled event accuracy |
| Collateral cohort size, seeds 8001 / 8002 / 8003 | Ranges 25–32 / 26–32 / 23–32; summed phase denominators 1,153 / 1,155 / 1,133 |
| Final old-correction denominator | 272 / 277 / 270 distinct keys whose latest assignment is at least 20 phases old |
| Final old-regular denominator on corrected entities | 749 / 782 / 743 shown, never-corrected keys on touched entities aged at least 20 phases |
| Same-value replacement assignments | 16 / 18 / 19 among 624 events; assignment is not synonymous with changed truth |
| Repeated-key assignments | 37 / 35 / 41; final distinct corrected-key counts 587 / 589 / 583 |

The stored write_steps field is not the mean number of updates per correction. It averages the returned zero-based loop index across phases. Fixed Adam stores 199 despite executing 200 updates. The stopped-arm means are 4,182.233333 for Adam and 748.733333 for GD. Each is a phase-wide final loop index governed by the slowest unfinished candidate or the cap; individual candidates can freeze earlier. A final index of 4,999 can mean an early stop before that iteration's update or completion of 5,000 updates without all candidates having passed. Saved aggregate files contain neither individual stopping times nor a cap-failure count, so those quantities cannot be reconstructed. [S1, S2]

### Batched candidate/deployment mismatch

The code clones one detached address for every selected fact, so two relations of the same entity produce two independently optimized candidate rows. It later assigns all rows through advanced indexing into one address table. The entity indices can repeat even though selected fact keys do not repeat within the phase. PyTorch documents this assignment as equivalent to index_put_ with accumulation disabled; behavior is undefined when indices contain duplicates. It is therefore incorrect to assume a guaranteed last-write-wins order. [S1, S7]

The exogenous event generator produces duplicate-entity candidate batches in 2, 4 and 5 phases respectively, containing 2, 4 and 6 candidates beyond the number of distinct entities. A candidate-level margin pass cannot certify simultaneous deployment of all those independent rows. The review calculator includes an elementary constructed example: candidates (1,0) and (0,1) separately meet margins on two coordinates, yet neither meets both margins of one. The jointly feasible state (1,1) exists. This demonstrates a contract gap; it does not identify which historical prediction errors came from the scatter, from stopping failures, from preexisting errors, or from future training.

### Causal limitations

1. Correction success is markedly unmatched: the 200-step arm achieves only 47.9% new-fact accuracy versus approximately 99% in the stopped arms. Its better neighbor accuracy is not a matched-efficacy superiority result.
2. Learning rate, adaptive geometry, candidate trajectory, step count and stopping rule vary together. No fresh-versus-carried local-state pair holds all those elements fixed.
3. collateral_now measures phase-end accuracy, rather than a saved pre-write-to-post-write change. The aggregate records cannot identify every newly induced error.
4. The decoder is frozen during a local write, but subsequent base training updates it. Final old-correction accuracy mixes write success, later decoder motion and other updates.
5. The base Model optimizer remains stateful during new-fact training; fresh local correction state does not reset the whole training system. Local stale moments are unnecessary for low phase-end neighbor accuracy in these recipes, but broad claims about every role of optimizer state are unsupported.
6. No per-candidate state, committed margin, constraint residual, convergence receipt or paired pre-write prediction is retained. The raw evidence is three rounded summaries per recipe.

Recommended ledger wording: **Poor neighboring-fact accuracy persists in the tested fresh-Adam and plain-GD correction recipes; matched-efficacy causal isolation and deployment-level margin validation remain open.**

## 4. Address × optimizer factorial

The distinct experiment fits a 16,384 × 1,024 value table, reading eight slots per query. It fixes 200 updates per phase for 60 phases, with 128 new facts and 16 selected corrections per noninitial phase. The four principal cells cross original versus hashed addressing with carried versus phase-reset AdamW state, with weight decay zero. Two mechanism arms use original addresses with β2 = 0.9 or plain SGD. Confirmatory seeds are 3101–3105. [S4]

The six development runs use seed 3001 and hashed addresses. All rates 1, 3, 10, 30, 100 and 300 produce late acquisition 1.0. The registered tie rule therefore chooses rate 1, making F6 informative. The driver implements the tie rule; the calculator independently verifies it. The published scorer prints the development selection but does not itself apply the informative gate when declaring F6. This causes no error on these files, because all development scores pass, but the published scorer is not a complete generic implementation of the conditional F6 rule. [S4–S6]

### Phase-log reconstruction

The primary endpoint uses phases numbered 45–59 in the zero-based logs, corresponding to phases 46–60 in the protocol. Values below are equal-seed means calculated from the saved phase-level values, not averages of the rounded per-seed header fields. Retention records contain floating-point accuracies; displayed precision refers to those stored numbers.

| Cell | Mean late acquisition | Seed range | Mean late retention | Retention range |
|---|---:|---:|---:|---:|
| Original / carried Adam | 2.416667% | 1.927083–2.916667% | 58.776850% | 57.206529–60.255903% |
| Original / reset Adam | 63.500000% | 54.739583–69.375000% | 59.992129% | 50.170484–66.026251% |
| Hash / carried Adam | 100% | 100–100% | 89.222279% | 89.087844–89.337844% |
| Hash / reset Adam | 100% | 100–100% | 99.686843% | 99.627664–99.752322% |
| Original / β2 = 0.9 | 41.864583% | 33.437500–48.177083% | 58.493065% | 49.214674–64.118996% |
| Original / plain SGD | 58.479167% | 49.531250–64.843750% | 55.576570% | 46.042492–61.728558% |

| Registered prediction | Literal count | Outcome |
|---|---:|---|
| F1: original/carried acquisition < 0.20 | 5/5 | Holds |
| F2: hash/carried acquisition > 0.90 | 5/5 | Holds |
| F3: original/reset acquisition ≥ 0.70 | 0/5 | Fails |
| F4: absolute hash reset/carried difference ≤ 0.05 on every seed | 5/5 | Holds |
| F5: β2 arm ≥ 0.70; refuted as stated if < 0.30 on at least four seeds | 0/5 pass; 0/5 refutation | Partial |
| F6: original/SGD ≥ 0.70, conditional on informative development fit | 0/5; development gate passes | Fails |

These verdicts are unchanged whether using the saved four-decimal summaries or recomputing the full phase logs. F4 has an explicit every-seed requirement; the generic four-of-five rule does not replace it. No population-level confidence guarantee is inferred from the five runs.

### Paired effect sizes and the “two thirds” statement

Resetting Adam changes original-address acquisition by 52.812500, 60.885417, 61.145833, 64.114583 and 66.458333 percentage points across the five seeds. The mean difference is 61.083333 points. The corresponding reset effect under hashed addresses is zero at the observed acquisition ceiling. Thus the descriptive difference-in-differences, defined as reset effect under original addressing minus reset effect under hashed addressing, is also 61.083333 points.

The ratio of mean improvements to the mean original-to-hash gap is

$$
\frac{0.635-0.0241666667}{1-0.0241666667}
=0.6259607173.
$$

This is a **62.596072% conditional descriptive recovery fraction**. Per-seed fractions range from 53.850239% to 68.454936%; their mean is 62.610937%, a different aggregation. Neither fraction identifies a unique share of causation attributable to optimizer state. With a strong interaction and an acquisition ceiling, such an allocation depends on the chosen reference and intervention.

The residual acquisition deficit after original-address reset is 36.5 percentage points against the hashed result. For original-address SGD it is 41.520833 points. These are residual differences under particular optimizer budgets and address recipes; they are not proven irreducible collision or representational-capacity limits.

Resetting changes mean original-address retention by +1.215279 points, with paired differences ranging from −7.036045 to +5.770348 points. “No improvement” is too categorical; the mean is positive, but the effect is mixed and small relative to the acquisition effect. Hash reset improves retention by 10.464564 points on average, with all five paired effects between 10.313280 and 10.584725 points. Retention was a registered secondary endpoint, but its directional reset effect had no registered prediction; it is a descriptive observation awaiting its own confirmatory test. [S4–S6]

### Remaining confounding and limits

The hash intervention changes query representation, slot-index distribution and weights: hashed reads use uniform weights while original reads use softmax weights from random fixed projections. This is not a collision-only intervention. Final distinct slots touched are recorded—original seeds use 2,539 / 3,361 / 3,280 / 3,080 / 3,840; hashed seeds use 15,995 / 16,011 / 15,990 / 15,979 / 15,999—but effective slot count, pairwise weighted overlap and per-query conditioning were not saved. Distinct touched slots are not an effective-capacity estimate.

Resetting Adam simultaneously changes first moments, second moments and bias-correction step history. The β2 variant also changes the time course of adaptive scaling. Its intermediate result neither proves nor universally refutes a second-moment mechanism. SGD's learning rate was deliberately selected on hashed addresses, so failure on original addresses does not establish failure of every rate or longer convergence budget there. Seed changes also change the task stream and initialization, so similar rank ordering across recipes cannot isolate the address draw as its cause.

The result narrative discloses a killed driver and restart of three incomplete hash/reset runs with empty files. Completed results are exclusively created by the experiment script; the driver removes only empty outputs before retrying. The selection JSON uses ordinary write mode, so “every artifact is never overwritten” would be too broad. All 30 confirmatory outputs are present and finite in the pinned result snapshot. The driver history is a disclosed interruption, not a fresh independent replication. Saved versions report PyTorch 2.11.0 and two threads; concurrent laptop load makes seconds unsuitable for efficiency comparisons. An exact keyed lookup solves this arbitrary-fact task; no language capability or native-model result follows. [S4–S6]

## 5. Architecture and execution consequences

| Issue | Required next design contract | Evidence still needed |
|---|---|---|
| Multiple candidates for one entity | Group corrections by entity and commit one validated row atomically | Joint margin and residual receipts on the deployed row |
| Candidate stop versus deployed outcome | Record initial, accepted and final margins; distinguish converged, initially satisfied and capped candidates | Per-candidate update counts and post-commit checks |
| Optimizer causal overclaim | Compare paired recipes with matched write success, protected constraints, budget and state policy | Registered controls that separate geometry, state and stopping |
| Address/collision attribution | Preserve weights while changing sharing, and vary sharing while measuring it | Weighted overlap, effective occupancy, conditioning and paired streams |
| Long-term preservation | Revalidate protected claims after later decoder updates | A bounded drift argument or direct longitudinal endpoint measurements |
| Transfer to native language modeling | Retain these as mechanism probes and acceptance-contract tests | Evidence from the actual native trainable language architecture |

These are specification and evidence requirements. They do not retroactively repair historical measurements or justify silently changing protected source files.

## 6. Reproduction and checks

The new read-only calculator is analysis/reconcile_optimizer_factorial.py. It verifies every manifest blob and SHA-256, rejects missing or inconsistent identities, checks all 60 phase positions, recomputes summaries and registered gates, applies the SGD selection gate, calculates paired effects, and reconstructs only exogenous correction events. Its output uses exclusive creation and rejects nonfinite JSON values.

Eight focused checks pass in checks/test_optimizer_reanalysis.py: late-window indexing; source/summary identity and finite values; threshold inclusivity; F4's every-seed rule; development tie and F6 gate; paired arithmetic; candidate-versus-joint margin contract; and exclusive finite output. The full reconciliation verified all 53 inputs. No historical neural predictions, weights, margins or trajectories were recreated.

~~~sh
python3 analysis/reconcile_optimizer_factorial.py \
  --evidence-dir /path/to/optimizer_evidence \
  --out /new/path/optimizer_factorial_reanalysis.json
python3 -m unittest discover -s checks -p test_optimizer_reanalysis.py -v
~~~

## Primary references

- **S1:** [Correction optimizer source at fixed cutoff](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/0afd5629990b4e442fb3aa8f7c700b030d8faede/research/frontier/ontogenesis/correction_optimizer.py).
- **S2:** [Committed correction raw arrays](https://github.com/wwk5q8z6kk-bit/nano-lm/tree/0afd5629990b4e442fb3aa8f7c700b030d8faede/research/frontier/ontogenesis/results/correction_interference), specifically optimizer-adam_200.json, optimizer-adam_stop.json and optimizer-gd_stop.json.
- **S3:** [Correction model and stream source](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/0afd5629990b4e442fb3aa8f7c700b030d8faede/research/frontier/ontogenesis/correction_interference.py) and cab_s.py at the same commit.
- **S4:** [Factorial protocol](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/b82d0f018ebcbe1e6604afc129731d96774d0f69/research/frontier/ontogenesis/PROTOCOL-MEMSAT-FACTORIAL.md) and [factorial source](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/767857e72bfa51e3fa6f32d047507aabef077fde/research/frontier/ontogenesis/memory_saturation_factorial.py).
- **S5:** [All factorial raw results and driver records](https://github.com/wwk5q8z6kk-bit/nano-lm/tree/767857e72bfa51e3fa6f32d047507aabef077fde/research/frontier/ontogenesis/results/memory_saturation_factorial).
- **S6:** [Factorial result narrative](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/767857e72bfa51e3fa6f32d047507aabef077fde/research/frontier/ontogenesis/RESULTS-MEMSAT-FACTORIAL.md) and [scoring source](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/767857e72bfa51e3fa6f32d047507aabef077fde/research/frontier/ontogenesis/memory_saturation_factorial_score.py).
- **S7:** [PyTorch 2.11 Tensor.index_put_ documentation](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.index_put_.html), checked November 21, 2026, America/New_York. Used only for the documented duplicate-index contract; the correction summaries themselves do not record a Torch version.

## Follow-up: chapter 12 and figure audit

The first chapter 12 draft inspected in this review transcribed several registered predicates incorrectly. The review reported these to the chapter owner without editing the chapter:

| Draft defect | Required literal protocol |
|---|---|
| F1 used below 10% | Strictly below 20%, at least four of five seeds |
| F2 used above 95% | Strictly above 90%, at least four of five seeds |
| F3 used exceeds 70% | At least 70%, at least four of five seeds |
| F4 used hash/reset above 95% | Absolute hash/reset minus hash/carried acquisition difference at most 5 percentage points on every seed |
| F5 used above 70% without a full count rule | At least 70% on at least four seeds; below 30% on at least four seeds is the stated refutation band |
| F6 used exceeds 70% | At least 70% on at least four seeds, conditional on the informative development selection |

The observed outcomes do not change under those mistaken thresholds in these particular data, but that does not make the wording interchangeable. The registration table must faithfully state what was registered. The review also flagged that only factorial JSON records supply Torch 2.11.0; correction-summary JSON has no environment version. The event reconstruction performed here concerns the correction generator, whereas factorial work recomputes saved phase logs. Chapter 12's numerical means, paired effects, conditional recovery calculation and exact geometric derivation were checked and agree with the audit. Its interpretation appropriately avoids a unique optimizer causal allocation; describing low phase-end collateral accuracy is more precise than equating every recorded error with a newly caused write error.

The new figure generator, analysis/plot_optimizer_factorial.py, reads results/optimizer_factorial_reanalysis_v5.json and plots all six recipe cells. It separates acquisition from retention, shares the percent scale, shows five individual observations, marks equal-seed means and draws only observed seed minimum–maximum ranges. The caption explicitly states that these are descriptive ranges, not confidence intervals. The complete plotted observations, fixed source commit and input SHA-256 are embedded identically in SVG and PNG metadata.

Generated and visually reviewed figures:

| Artifact | SHA-256 |
|---|---|
| figures/optimizer_factorial.svg | aed08eff538f94d924190869cb809a28d8479b80d8e20308a0162c55c81526ca |
| figures/optimizer_factorial.png | 881c733ad25fab1fd0117ddfc0037b306a232558507139ade17272aa93801410 |
| Plotted reanalysis input | 7b0d97a10ef59032b17589137228d4c10dc2585e2920d92d35033337593a6d2f |

The PNG is 2,376 × 1,296 pixels. SVG text remains searchable. A missing output directory argument and existing output files were both rejected, and artifact hashes remained unchanged. The script requires an explicit destination and opens both outputs exclusively. No default invocation writes files.

**Closure check:** after the chapter owner applied the requested edits, the reviewer reread the F1–F6 rows, reconstruction-scope row and Torch-version paragraph. All material transcription and provenance findings listed above are corrected in the inspected revision. The separate geometry review remains passed. This closure concerns those reviewed claims, not a fresh empirical confirmation.

## Follow-up: V5 gate adapter

The new analysis/validate_release_v5.py retains the sixteen V4 command groups and reuses the unchanged V4 runner, saved-result comparisons, 21-copy manifest check, complete-result preservation check and ledger-prefix helpers. Its only runtime adjustment to V4 expectations is the V5 package-validator inventory: 45 events, 39 machine decisions, and a successful complete supplied-prefix comparison. The original expectation object is restored after that one call; the historical source file is not edited.

Seven additional groups check the optimizer audit, exact geometry, causal cache, store reanalysis and native comparison. The native comparison runs within its fourteen-test suite only. The suite's complete JSON stdout is compared to the losslessly compressed full receipt and every corresponding field of the readable summary before large per-event arrays are omitted from the combined gate receipt. The compact receipt records the canonical parsed digest and selected summaries. No separate full comparison command is scheduled.

The adapter verifies 85 consolidated source copies, the complete supplied 36-event machine prefix, the latest supplied canonical-ledger prefix, all dynamically discovered historical result files, and package/baseline bytes before and after execution. The supplied historical result directory contains 17 files; no assumed count of 18 is used. Human inventory requires 62 distinct IDs; additions require eight new decisions D-055–D-062 and one D-054 revision. The repaired cache implementation and test hashes are explicitly pinned, and new cache output is compared against causal_cache_checks_v5_final.json while the earlier receipt remains preserved.

**Validation status at handoff:** syntax and read-only helper checks passed, including all four saved V5 semantic fixtures, the 85-copy consolidated manifest, strict unittest success/skip/count parsing, and native gzip/summary agreement. The native full receipt's canonical parsed SHA-256 is 790723a0ac1c2f3423f21532efb099c45a103ab262a8b47ee825a1f37ee7c091. No command group or final gate was executed during this adapter work; the release owner must run the full gate after documentation and ledger edits stop.

**Review risk:** exact parsed receipt comparisons intentionally bind this release to its recorded deterministic runtime and code. A changed runtime, changed reviewed cache source, a concurrent package edit, or a changed historical baseline can fail the gate even when a different scientific claim might remain plausible. Such a failure must be investigated; the adapter does not weaken old predicates or rewrite saved results to obtain a pass. Supplied local baselines and hashes do not authenticate remote repository history.

### Native suite stdout contract hardening

The original native test suite already emits one full JSON receipt through its CLI I/O check; the release owner's captured 8,453,400-byte stdout matches the preserved gzip. There was no reproduced empty-stdout failure. The gate now makes this behavior explicit through a validator-owned command adapter: it suppresses incidental stdout while importing and running exactly fourteen tests, keeps the unittest result on stderr, rejects failures/skips/expected failures, then serializes the cached class report exactly once. It does not call the comparison again.

Four small synthetic adapter fixtures check successful emission, failure suppression, skip rejection and wrong-count rejection. The successful fixture invokes setUpClass once. No native model suite or final gate was executed for this adapter change, and native test/model/protocol source hashes remain unchanged. The change is an output-contract improvement, not a claim of a newly discovered empirical defect.

### Final bounded addendum configuration

The release owner subsequently authorized a bounded late audit at c597d5fefdd378c3e5c66a51e7690cdb90853287. This adds one deterministic source/rounded-result reanalysis group, bringing the final gate to **24 groups: 16 preserved V4 groups and eight V5 groups**. The original review cutoff remains 0afd5629990b4e442fb3aa8f7c700b030d8faede; the late cutoff and integration base are separately recorded as c597d5fefdd378c3e5c66a51e7690cdb90853287.

The core 85-entry manifest remains byte-identical at SHA-256 d99569a08eb00ae9264f5575746c99724d6fc1f5d8b84a40347ede040507a804. A separate five-entry addendum manifest gives 90 distinct local source copies in total; the old 21-copy set is already included in the 85. The final machine ledger is expected to contain 46 events for 39 decisions, with 62 human IDs. Its ten events after the V4 baseline comprise eight new IDs plus D-054 and D-058 revisions. The gate preserves both the supplied 36-event baseline and the complete 45-event core V5 prefix: 66,297 bytes, SHA-256 1a4f2750d855412afec1c2b47335948a4e9b29b0b0bd632317700509a052cdf0.

The additional group compares the deterministic late CLI output exactly to results/store_cache_late_reanalysis_v5.json. Its semantic checks require five late and ten prior verified inputs, fifteen reconciled rows, eight unchanged model/evaluation ASTs, two unchanged exchange-script blobs, and the source-derived S/M/L parameter counts. It preserves the distinction between a qualified numerical 40M cache stopping branch and the still-pending registered 101M store result. Neither the gate nor this review executes a language-model evaluation.

Syntax and saved-fixture checks of this final adaptation passed. Final package execution remains the release owner's pending step after the addendum manifest, copied evidence and appended D-058 event are in place. Earlier 45-event/23-group descriptions above document the preceding stage and are superseded by this final configuration.
