# ACE-X9.2 causal checkpoint scorer

Two recovered NanoLM Omega 47M checkpoints changed predictions on a fixed prefix when only its suffix or an unrelated batch row changed. The original evaluation refiner ranked scores over the entire batch and sequence. At training time it updated the uncertainty threshold from the current batch before using it on that same batch.

The scorer processes each example using a prefix budget. At position *t* an eligible token is admitted whenever the number already routed is below max(1, floor(3 × t / 10)). Training reads the ACE threshold before updating its moving average. The checkpoint state layout, X8 comparator, reference backbone, candidate attention and final margin check are preserved.

## Fixed-weight results

Across 16 blocks and two trained ACE seeds, 1,967 of 16,384 perturbed-position comparisons changed logits by over 0.0001 and 157 greedy tokens flipped. Both counts fell to zero with the prefix budget. The identical-input and X8 control counts were zero both ways. Comparisons reuse the same prefixes. Under training mode with independently restored initial EMA buffers, 20 and 22 positions changed route on the original two seeds, versus zero on both after the threshold fix.

For language quality the historical reference recomputes the original final-token decision separately at every single-sequence prefix. The audit-only vectorization matched route and argmax on 48 literal truncated forwards. Its highest absolute logit difference was below 0.00002. A batched vectorized ranking over a complete sequence would be an invalid reference for intermediate targets.

| Seed | Historical single-row serving, weighted BPB | Prefix-budget weighted BPB |
|---|---:|---:|
| 0 | 1.136141 | 1.135087 |
| 1 | 1.140272 | 1.139773 |

The paired mean change was −0.000777 bits per byte across 96 windows and 12,288 targets per model. A paired domain-stratified window bootstrap gave [−0.001168, −0.000366]. This interval conditions on the two available checkpoints and shares windows between seeds. Textbook loss increased slightly on seed 1. The seed-0 copy probe was unchanged at 1,914/2,016. Seed 1 gained one correct target and the X8 control remained strongest.

A separate 64-repository code panel showed small mean improvements for each ACE checkpoint. Three 129-token windows occurred verbatim in the historical training stream. Across the 61 without an exact full-window match, the uncertainty interval included zero. Older documents or shorter passages can still overlap. The row-screened code panel, real-language diagnostic and small synthetic copy task support different conclusions. None measures general model leadership.

## Replay

Python, NumPy and PyTorch are required. The original evaluation used CPU float32, PyTorch 2.11. Run from the repository root with the recovered local checkpoint and LAB-DATA-01 validation files. Each command verifies the checkpoint and all four token files by SHA-256. Neither source text nor weights are distributed. The JSON scorer output includes the 96 domain/offset observations.

```bash
python -m pytest -q research/ace_x9
python -m research.ace_x9.score "$CHECKPOINT" "$LAB_DATA" --seed s0
python -m research.ace_x9.score "$CHECKPOINT" "$LAB_DATA" --seed s0 --policy stock-serving
python -m research.ace_x9.audit "$CHECKPOINT" "$LAB_DATA" --seed s0
python -m research.ace_x9.audit "$CHECKPOINT" "$LAB_DATA" --seed s0 --archived-batch-gate
```

Repeat with the second checkpoint and `--seed s1`. The archived Track A source was recovered locally at commit `5c830ecf72648dc68ba31de89c5ce10e9b7ccbbf`, path `nanolm_omega/training/train_x9_tracka_cuda.py`, blob SHA-256 `4cb1a62be729e083bfc78d15cec52d006151ffb2690b4b3e4f339436fec906ee`. The portable scorer contains the required model definitions from that file. Exact checkpoint digests are in [score.py](score.py) and the domain data identities and bytes-per-token constants are in [audit_contract.py](audit_contract.py). [Per-seed results and controls](results.json)
