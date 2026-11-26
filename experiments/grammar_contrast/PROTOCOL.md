# Grammatical preferences of recovered NanoLM checkpoints

Compare the frozen S2, M0 and M2 native models on English grammatical contrasts. Use Pythia-410M at training step 256 and its final checkpoint as public references. The earlier checkpoint supplies a similar training-token budget. Training corpora, tokenizers, optimizers and model families differ.

Before scoring, lock the BLiMP data revision and deterministically select 32 of the 1,000 pairs from each of its 67 paradigms, for 2,144 pairs total. Rank pair IDs by SHA-256 of `nano-grammar-20261120/{UID}/{pairID}` and take the first 32. Record a digest of the exact pair IDs and both sentences. BLiMP predates these checkpoints. Exposure to evaluation text in the training corpora is unknown.

The primary metric is the fraction of pairs for which the model assigns higher **total** log probability to the grammatical sentence. Prepend the checkpoint's end-of-document token as context and score each sentence token without a final end-of-document token. Use greedy-free, teacher-forced evaluation without prompts or sampling, at temperature one. Record ties separately and count them as zero wins. Report the macro average across paradigms, each phenomenon family, and the result on equal-subword-length pairs. Keep the floating-point per-pair margins for replay. A native training-frequency unigram is the word-order-insensitive baseline. Better results here establish a preference on these English contrasts. Their scope is narrower than general reasoning or instruction following.

The study is exploratory and uses public evaluation material. Native weights stay local. Allow up to $2 in Google CPU credits for the two public reference checkpoints if useful, with a worker timeout of two hours.

Warstadt et al. introduced BLiMP and the full-sentence comparison in their paper. [BLiMP, TACL 2020](https://aclanthology.org/2020.tacl-1.25/) · [Pinned benchmark data](https://github.com/alexwarstadt/blimp/tree/3e56b06fcabca9b30822fc66435fca6b1aa40bb1/data)
