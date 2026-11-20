# Native whole-source selection and abstention

Sealed exploratory follow-up on November 20, 2026. This document fixes the experiment before fitting or inspecting model results. The previous experiment obtained unseen present exact recovery of 54.1% under source alignment and 10.4% under answer-only training. Alignment also increased erroneous assertions on absent fields. In particular, only 41, 30 and 13 of 52 retracted fields were correctly withheld across the three seeds.

## Primary question

Does jointly learning to choose a complete source word or abstain improve faithful final-state extraction on fresh, differently phrased episodes compared with the best prior aligned greedy procedure? A positive result would support bounded synthetic transfer. It would not establish real encounter performance.

Use four original toy fields in 240 paired training episodes (480 records), 48 development pairs (96), 64 familiar-style test pairs (128), and 80 primary transfer pairs (160). Each pair differs by retention versus explicit retraction of one focal slot. Interleaved other slots, overwritten values and explicit historical mentions provide controls. The primary transfer set uses distinct, newly authored surface patterns. Seeds and whole field values differ across all splits and earlier archives. No PHI, protected run inputs or external models. Seal exact data hashes, generator, code and native source before the full training run.

## Controlled comparison

For each of three new seeds 4919, 5849 and 7307 train the same 65,524-parameter native HDCD/attention/copy model from matched initialization. Compare (A) answer-only likelihood plus native source-position and gate alignment, as in Study 1, with (B) the same loss plus unit-weight cross-entropy for the active whole source word or a beginning-of-source abstain sentinel. Use the existing query/key projections for whole-word scoring. Both arms get identical tokenized inputs, sampled example order, AdamW schedule of 360 steps and batch size 24, decoder budget, model parameters and evaluations. Additional supervision and computation are the intended change. Use the exact active event when training an occurrence-level selector. Preserve the original latest matching word convention in both arms' common token-copy objective. Evaluate each trained arm with both its ordinary 16-token greedy decoder and whole-source decoder, so the effects of changing decoding alone and adding selector training are visible. A deterministic grammar parser using training templates and ordinary text input is an additional reference. Gold events and slot values are unavailable to all test decoders. An explicit source-word selector can only emit a source word or abstention. This design is limited to single whitespace-delimited values.

For each arm and seed, fix a calibration threshold once on development: among thresholds -4 to +4 inclusive in steps of 0.5 and an always-abstain endpoint, maximize exact recovery of present values subject to at most 5% erroneous fills among absent fields. Break ties by lower false-fill rate and then larger threshold. Freeze it and apply unchanged on both test splits. Report raw zero-margin decoding as a separate diagnostic. Any test distribution shift is part of the measurement.

## Prospective decision

Advance for further synthetic testing only if the following all hold on the primary transfer set: whole-source (B) averages at least 5 percentage points greater present exact recovery than greedy (A), with a positive difference in all three seeds and a positive lower bound on a paired 95% encounter-pair bootstrap interval; mean absent false-fill and final-retraction false-fill rates are each no greater than 5%. Report seed-level physical denominators, exact source occurrence accuracy, focal counterfactual pair both-correct rate, and errors on never-mentioned versus retracted fields. If transfer fails the rule, continue to report familiar-style results and the decoding/training ablations without promotion. Bootstrap resampling is over encounter pairs shared across seeds, not nine independent datasets. One fixed generator and lexical cues limit inference.

## Execution

One ephemeral Vertex AI n2d-standard-4 worker in us-central1, 4 vCPU, 16 GiB memory, 100 GiB temporary standard disk, pinned Google PyTorch CPU container, three fit workers. The existing seven-N2D-vCPU regional limit fits one such machine. Cap this job at 90 minutes with retry disabled, about $0.31 at the verified CPU/RAM and disk list rates before credits and tax. Use no endpoint, bucket, persistent runtime or GPU. A local one-step smoke may check code and shape invariants using training rows only. No analysis of model test predictions before the precommitted comparison.

The original sealed design included a billing-account suffix. This public copy omits that account detail. The original document and full input manifest fingerprints are in [design-commitment.json](design-commitment.json).
