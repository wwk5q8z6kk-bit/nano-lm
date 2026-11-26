# Selecting patient evidence on 44 synthetic dialogues

Each method proposes one of five field states and may cite a contiguous transcript span. An assertion only earns an exact match when its polarity, offsets and quoted text match the reference. For a field without a reference assertion, any nonasserting state receives credit. The 44 development encounters contain 89 assertions or explicit denials and 131 fields that call for withholding. They were authored separately from the training rows. The developers had used the evaluation families during prior product development.

The Qwen readouts used hidden representations of the five-field candidate menus and 318 informative training groups from 132 synthetic encounters. The baseline uses the frozen model's judgments. Each fitted readout trains a linear scoring head. A separate 7B experiment fine-tuned 270,336 query and value adapter parameters in the upper six layers for 160 updates on two T4s. Its frozen control used the same shuffled slate prompt. The direct reader is a span-extraction model trained on 924 questions over the same training encounters, with an explicit null answer and patient-turn constraints. For the menu methods, fixed scores were checked against the available choices in field order, honoring offsets reserved by earlier fields. The exact training and evaluation runtimes, model IDs and revisions are recorded in [training receipts](training-receipts.json).

The reference verifier is master commit `d109d068a8388a564a769a6d93d8d88c790ecce6`. Candidate menus for training and initial selection were generated at `e1bf801c41ea2ecf35f9d34f87245eb000aec37a`. Each method's five-slot proposal below has already undergone its method-specific selection on the reference verifier. This archive replays those strings through the public grounded binder and note renderer. It checks actual retained evidence. It does not rerun a model or regenerate its menu if the verifier implementation changes.

| Method | Exact fields / 220 | Exact assertions / 89 | Total assertions | Assertions on 131 withheld fields |
|---|---:|---:|---:|---:|
| Always withhold | 131 | 0 | 0 | 0 |
| First eligible assertion | 193 | 65 | 81 | 3 |
| Reference-informed menu choice | 198 | 67 | 67 | 0 |
| Qwen 1.5B baseline | 189 | 61 | 81 | 3 |
| Qwen 1.5B readout | 191 | 63 | 81 | 3 |
| Qwen 7B baseline | 190 | 62 | 81 | 3 |
| Qwen 7B readout | 190 | 60 | 75 | 1 |
| Qwen 7B shuffled slate, frozen | 186 | 58 | 76 | 3 |
| Qwen 7B shuffled slate, adapter | 187 | 58 | 74 | 2 |
| Direct reader before training | 142 | 13 | 16 | 2 |
| Direct reader after training | 189 | 58 | 61 | 0 |

The 7B adapter took 22.4 minutes. It removed one assertion on a withheld field while preserving its exact positive count. The first-eligible selection recovered seven more exact positives. The 7B readout also fell below the 1.5B readout on this slice. The reference-informed menu choice could recover only 67 of 89 assertions. Even a correct selection among these candidates can omit evidence. The first-eligible rule sometimes treated a residence, appointment or prescription date as the duration of a symptom. That behavior and regression tests are recorded in [PR 182](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/182).

To reproduce the table, pass a separate local Nano LM checkout at the reference commit. The script requires the project's Python dependencies. It requires no model weights, GPU or provider credentials.

```sh
python3 replay.py /path/to/nano-lm-at-d109d068 > replayed.json
diff -u results.json replayed.json
```

[Evaluation dialogues and references](cases.jsonl) · [Per-method proposed source slots](proposals.jsonl) · [Full numeric results](results.json).
