# Frozen native checkpoint comparison on public US texts

This evaluates next-token prediction by three recovered native NanoLM checkpoints. The passages come directly from 24 public 2025 documents, eight per publisher. Inputs and the analysis plan were committed before scoring. [Protocol](PROTOCOL.md) · [evaluation source URLs and text hashes](sources.json) · [checkpoint and tokenizer hashes](inputs.json).

## Checkpoints and comparability

| Arm | Architecture | Parameters | Training tokens | Historical four-domain mean BPB |
| --- | --- | ---: | ---: | ---: |
| S2 | Gen3, twelve delta and four attention blocks | 121,668,990 | 599,916,544 | 0.91355 |
| M0 | Dense attention | 348,966,912 | 399,900,672 | 0.90660 |
| M2 | Gen5, eighteen delta and six attention blocks | 350,397,134 | 399,900,672 | 0.89471 |

M0 and M2 have the same training-token budget and one seed each. M0 used the fixed domain mixture and Muon. M2 used adaptive domain sampling and the NorCautiousMuon optimizer. S2 is smaller and consumed about 50% more training tokens. These checkpoints support a comparison of the complete trained systems. Isolating architecture would require matching the data stream and optimizer and varying the training seed. The prior loss column comes from the original training records, with their own domain weighting and byte estimates.

## Reproduce

Tested on CPU with Python 3.14.7, PyTorch 2.11.0, tokenizers 0.22.2 and NumPy 2.5.1. Python 3.11 or later provides the file-digest API. The historical input files are required for scoring. Set `DATA` to the v1 directory containing `tokenizer_32k.json` and `train.bin`, and `RUNS` to the parent of the three run directories specified in `inputs.json`. Set `OUT` to an output directory. The scorer validates the tokenizer, all checkpoint digests and the original training stream. It loads tensor weights with strict state keys on CPU.

```sh
cd experiments/native_public_language
mkdir -p "$OUT/scores"
python corpus.py --data "$DATA" --cache "$OUT/raw" --output "$OUT/passages.json"
python evaluate.py --mode control --data "$DATA" --passages "$OUT/passages.json" --output "$OUT/scores/control.json"
for arm in S2 M0 M2; do
  python evaluate.py --mode "$arm" --data "$DATA" --runs "$RUNS" --passages "$OUT/passages.json" --output "$OUT/scores/$arm.json"
done
python summarize.py --results "$OUT/scores"
```

A reader with the published per-document loss totals can reproduce the table and the interval without the model files or network access. This command requires NumPy only.

```sh
python summarize.py --results results
```

The archived module directory contains the inference definitions used by the recovered checkpoints. Source symbols, original file hashes and the historical Git revision are retained. Relative imports isolate the definitions. The evaluator uses the reference float32 CPU path, accumulates cross entropy in float64, and checks causal-prefix stability, native loss and a manually recomputed token loss. Each block restarts positional indices. The previous document token supplies one token of context at a block boundary. [Architecture provenance](archived/PROVENANCE.json)

## Source and training provenance

The Board of Governors supplies the economic-situation section of the FOMC minutes, the Government Publishing Office supplies the enacted laws, and the National Hurricane Center supplies forecast discussions. Normalization collapses whitespace, removes publication scaffolding, and caps each text at 1,024 tokenizer IDs. Six short laws use recurring resolution language. Adjacent meetings and storms can also reuse phrasing. These dependencies and the small number of documents limit generalization. [Federal Reserve terms](https://www.federalreserve.gov/disclaimer.htm) · [GovInfo policies](https://www.govinfo.gov/about/policies) · [National Weather Service terms](https://www.weather.gov/disclaimer)

The original training mixture included FineWeb-Edu (62%), Cosmopedia v2 (13%), FineMath (13%), and GitHub Code Clean (12%). The first three dataset cards declare ODC-By. The code collection declares Apache-2.0 at the dataset level. Its upstream source records carry individual licenses. The historical builder reads the code column and filters lengths, without a source-license allowlist. In the first training shard, the first six eligible records for each of GPL-2.0, GPL-3.0 and AGPL-3.0 yielded respectively six, six and four full-source matches, including EOS, in the archived code stream. This comparison publishes measurement code, provenance and result totals. Obtaining the historical weights and tokenizer separately remains a requirement for model replay. [Historical dataset revisions](training-sources.json) · [Code-source audit](training-license-check.json) · [CodeParrot source card](https://huggingface.co/datasets/codeparrot/github-code-clean)

To reproduce the optional source-license check, install PyArrow, download the first CodeParrot parquet named in the pinned data revision, and supply the archived `train_code.bin` in `DATA`.

```sh
python audit_training.py --data "$DATA" --shard "$FIRST_CODE_PARQUET"
```
