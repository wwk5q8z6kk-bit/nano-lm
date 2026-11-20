# Grammar contrast results

Scored 2026-11-20 on the panel fixed in [PROTOCOL.md](PROTOCOL.md): 2,144 BLiMP pairs, 32 from each of 67 paradigms (panel digest `24918cf5…9ccb6`). A win means the model gave the grammatical sentence the higher total log probability. Each paradigm has 32 pairs, so pair accuracy and the paradigm macro average are equal.

| Arm | Wins of 2,144 | Accuracy | Equal-subword-length pairs |
|---|---:|---:|---:|
| Pythia-410M, final | 1,730 | 80.7% | 81.1% of 1,828 |
| S2 (native, 122M) | 1,693 | 79.0% | 79.2% of 1,781 |
| M2 (native hybrid, 350M) | 1,655 | 77.2% | 77.3% of 1,781 |
| M0 (native dense, 349M) | 1,636 | 76.3% | 76.3% of 1,781 |
| Pythia-410M, step 256 | 1,160 | 54.1% | 53.6% of 1,828 |
| Training-frequency unigram | 1,080 (159 ties) | 50.4% | 49.0% of 1,781 |

All three native checkpoints prefer grammatical English well above the unigram baseline and the early Pythia checkpoint. The fully trained public reference is ahead of all three. Among native models the smallest, S2, scores highest; it also saw about 50% more training tokens. M2 is 19 pairs ahead of M0, which is within what one seed per model can distinguish.

Family-level accuracies and per-pair margins are in [results/](results/); `python summarize.py` regenerates [results/summary.json](results/summary.json). The arms differ in training corpus, tokenizer, optimizer and size, so this compares complete trained systems and does not isolate architecture. Exposure of BLiMP text in any training corpus is unknown.
