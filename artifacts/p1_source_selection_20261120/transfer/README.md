# Evidence selection in six simulated visits

Six ACI-Bench visits were sampled from the public ASR-corrected ACI training transcripts using a fixed salted ordering of encounter IDs. Each visit contains 26–95 speaker turns and 985–1,636 words. An agent read the normalized transcripts without the paired clinical notes or predictions. The agent recorded eligible patient-turn intervals, ten fields that should be withheld and six ambiguous exclusions. Multiple intervals can be admissible. These are development judgments from a single reader. [Source and attribution](source.json) · [Reference intervals](reference.json)

Both Vertex teachers received the same transcript and five-field instructions used in the synthetic study. An exact positive required one approved source interval. An unusable note or malformed field earned zero credit. The numbers below use the verifier at master commit `d109d068a8388a564a769a6d93d8d88c790ecce6`. [Proposals](proposals.json) · [Results](results.json)

| Method | Exact fields / 24 | Exact assertions / 14 | Malformed or unusable fields | Assertions on ten withheld fields |
|---|---:|---:|---:|---:|
| First eligible menu assertion, checked | 12 | 2 | 0 | 0 |
| Direct reader, checked | 10 | 0 | 0 | 0 |
| First prelisted reference interval, checked | 12 | 2 | 0 | 0 |
| Gemini Flash, raw | 17 | 7 | 3 | 0 |
| Gemini Flash, checked | 11 | 1 | 0 | 0 |
| Gemini Pro, raw | 14 | 5 | 4 | 0 |
| Gemini Pro, checked | 12 | 3 | 3 | 0 |

Pro returned one empty completion. Three scorable fields from that visit failed the checked route. The reference control recovered two of fourteen positive intervals when given the first admissible interval directly. Span choice and the verifier both limit recovery on this slice. A shorter symptom quotation or a different passage confirming a medication can miss the strict boundary criterion while preserving the underlying fact. Six visits with agent-authored labels support a transfer diagnostic only.

From this directory, run the following with Python 3.11 or later. The first command downloads the pinned public CSV and verifies its digest. For offline use, pass `--csv /path/to/train_aci_asrcorr.csv`.

```sh
python3 -B replay.py /path/to/nano-lm-at-d109d068 > replayed.json
diff -u results.json replayed.json
```

Yim and colleagues released ACI-Bench under CC BY 4.0. We standardized speaker tags and inside-turn whitespace. The paired notes were excluded. [Original publication](https://doi.org/10.1038/s41597-023-02487-3) · [Source repository](https://github.com/microsoft/clinical_visit_note_summarization_corpus)
