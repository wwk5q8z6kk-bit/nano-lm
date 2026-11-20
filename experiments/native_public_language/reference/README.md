# Dense public reference on identical text chunks

Pythia-410M's final checkpoint gave the lowest loss on these 24 documents. Its 256-step checkpoint gave the highest. S2, M0 and M2 fall between them. This comparison re-scores all five checkpoints on exactly the same 100 byte ranges. Each range starts from the model's native boundary token. Shorter contexts change the native results from the original 256-native-token experiment. [Protocol](PROTOCOL.md) · [per-chunk hashes and token counts](chunks.json) · [per-document results and intervals](results/summary.json)

| Source | Bytes | S2 | M0 | M2 | Pythia step 256 | Pythia final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monetary-policy minutes | 29,188 | 1.06442 | 1.07305 | 1.08873 | 1.87091 | 0.82832 |
| Federal laws | 9,868 | 1.28446 | 1.28815 | 1.31236 | 2.44941 | 0.83746 |
| Hurricane discussions | 16,717 | 1.31190 | 1.31795 | 1.35222 | 1.99732 | 1.12882 |
| **Pooled** | **55,773** | **1.17753** | **1.18451** | **1.20727** | **2.01115** | **0.92001** |

Scores are bits per UTF-8 byte. Final Pythia used 0.28726 fewer bits per byte than M2, with a paired within-publisher document bootstrap interval of −0.32371 to −0.25277 for Pythia minus M2. Final Pythia beat M2 on 24 of 24 documents. Step 256 used 0.80388 more bits per byte than M2, with an interval of +0.73620 to +0.89504. M2 beat step 256 on 24 of 24 documents. The interval reflects document variation in these three source groups.

Both Pythia checkpoints have 405,334,016 parameters. The early checkpoint saw 536,870,912 tokens. It lies within the run's 1,430-step learning-rate warmup. M0 and M2 were trained to completion at about 400 million tokens and S2 at about 600 million. Final Pythia saw 299,892,736,000 tokens, approximately 750 times the M0/M2 budgets. The 2025 documents postdate the original Pythia training run. Earlier texts in the non-deduplicated Pile can share phrasing. The Pile was not scanned for overlap. These contrasts compare whole checkpoints and cannot isolate architecture, pretraining efficiency or general knowledge. [Pythia training configuration](https://github.com/EleutherAI/pythia/blob/362f0dd808ae323a4fa5ff8ee475bfac69ca314c/models/410M/pythia-410m.yml) · [Pythia model and data card](https://huggingface.co/EleutherAI/pythia-410m/blob/9879c9b5f8bea9051dcb0e68dff21493d67e9d4f/README.md)

## Reproduce

The byte manifest was frozen before evaluating these five arms. An initial 768-byte limit exceeded 256 tokens on a law passage during preparation. The final 640-byte limit gave a maximum of 246 native or 217 Pythia tokens per chunk. All chunks decode exactly to their original bytes. The native tokenizers encode 12,076 targets and Pythia's tokenizer encodes 11,293. The scorer checks model and tokenizer hashes and compares every loaded public learned tensor with its value in safetensors. The original Pythia release uses float16 storage and the intermediate safetensors file uses float32. All inference here uses float32, with cross entropy accumulated in float64. [Immutable revisions and file hashes](inputs.json)

Starting in `experiments/native_public_language`, set `DATA`, `RUNS` and `OUT` as described in the [native replay instructions](../README.md). Generate `$OUT/passages.json` with the native corpus script. Install Transformers and safetensors for the Pythia runs. The recorded environment used Transformers 5.5.3, safetensors 0.8.0 and huggingface_hub 1.26.0. Download the two public checkpoints to an external directory.

```sh
hf download EleutherAI/pythia-410m model.safetensors config.json tokenizer.json tokenizer_config.json special_tokens_map.json --revision 9879c9b5f8bea9051dcb0e68dff21493d67e9d4f --local-dir "$OUT/pythia-final"
hf download EleutherAI/pythia-410m model.safetensors config.json tokenizer.json tokenizer_config.json special_tokens_map.json --revision d578bb6ee435535f06872425332b77b2bbc7a81e --local-dir "$OUT/pythia-step256"
mkdir -p "$OUT/comparison"
python reference/compare.py --mode prepare --data "$DATA" --reference "$OUT/pythia-final" --passages "$OUT/passages.json" --output "$OUT/comparison/chunks.json"
for arm in S2 M0 M2 Pythia-final Pythia-step256; do
  model="$OUT/pythia-final"
  if [ "$arm" = Pythia-step256 ]; then model="$OUT/pythia-step256"; fi
  python reference/compare.py --mode "$arm" --data "$DATA" --runs "$RUNS" --reference "$model" --passages "$OUT/passages.json" --output "$OUT/comparison/$arm.json"
done
python reference/summarize.py --results "$OUT/comparison"
```

To reproduce the numerical comparison from the committed loss totals with NumPy only, run `python reference/summarize.py`.
