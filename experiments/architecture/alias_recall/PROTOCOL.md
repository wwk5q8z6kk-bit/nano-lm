# Does seeing an alias answer earlier in context change a native model's recall?

Exploratory study specified on 2026-11-20. The earlier EXACT-RECALL-03 generator sampled queries with replacement. An earlier query could reveal the exact answer to a later one. This study isolates prior exposure while holding the record set and scored target fixed.

Train a native two-layer delta/attention backbone with its dual-offset copy head, the parameter-matched generator and a three-scalar identity-copy control. Use 800 updates with batches of 32 synthetic worlds per arm, seeds 31271, 71339 and 91823, the original 8-to-24 upper-bound curriculum, and disjoint RNG streams for training data and copy dropout. Only values 0–223 appear in training. Evaluate 1,024 independently generated worlds (seed 520241) containing keys from the seen primary-key range and two-token values drawn from 224–255. Each world has 24 records and 16 alias queries.

Score the same final alias twice. In one context all 16 aliases are distinct. In the other, replace one uniformly chosen earlier query and its answer with the scored target and its answer. Both arms have the same length. In each arm the model generates both answer tokens greedily. Record matched exact-pair counts and both discordant cells. Report each training seed separately. A prefix-only lookup should score 0/1,024 versus 1,024/1,024. An oracle supplied the alias mapping should score 1,024/1,024 on both.

The primary measure is unique-context pair accuracy for each arm. The secondary measure is the paired change after exposure. Gains that disappear on the unique arm point to a shortcut. A gain for the dual head over both controls across seeds would motivate a natural-text recall test. This synthetic experiment establishes neither general language quality nor a causal interaction between individual components.

Use one CPU worker on Google Cloud, a maximum two-hour runtime, no retry and a compute and disk allowance of $1. Emit per-arm results during training. A partial matrix is reported as partial. No model selection or hyperparameter search uses these worlds.

## Replay

Run `python check_worlds.py` and `python study.py --output results.jsonl` from this directory with NumPy and PyTorch installed. The default arguments run the full design. Each output row includes its arm, training seed, parameter counts, shared world digest and paired counts. Smaller smoke runs must be labeled separately.

The backbone is a byte-identical snapshot of the project's MIT-licensed `research/exact_recall/v3_model_snapshot.py` at source commit `679e9c675ffe2eaaf77a89853f56a6f03b766510`. The baseline classes, optimizer and training generator come from `research/exact_recall/dual_idcopy.py` at that revision. The historical experiment and its artifacts stay at their original revision.
