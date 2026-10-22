# Compact model → checked scribing: development pilot

Twenty new synthetic dialogues (ten worlds, each with canonical/paraphrased
questions) were run locally with **$0 incremental paid compute**. The compact
model now reaches the existing transcript-only extraction → encounter → checked
review-draft path. It is **not ready for broad scribing**.

The model is the cached **base Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, not a recovered fine-tuned adapter
from the historical two-field compact study. Full checkpoint/tokenizer hashes,
runtime versions, prompt, raw generations, stage timings, contract outputs,
encounters and drafts are retained in each complete run directory. CPU inference
avoided cloud startup and paid resources; no RunPod/Kaggle/OpenResearch compute
was launched. The historical campaign balance was not treated as current.

## Instrument admission

[dialogues.json](dialogues.json) contains only invented, exposed development
examples. [spec.json](spec.json) independently specifies the 25-cell quota table:
66 supported, 14 absent, 12 missing, four uncertain and four conflicting fields
out of 100. Paired Patient replies/targets are identical; only Doctor questions
change. All ten worlds stay in the development split. Source/target span binding,
all quotas, unique sources/IDs, world pairing and split checks pass.

The test-only gold oracle scores 100% grounded exact fields; the all-missing
negative control scores 12%. Neither oracle is used for scribing. The reference
extractor scores 56% states and grounded exact fields over all 100 fields: 50/50
canonical fields and the six missing paraphrase fields. This exposes vocabulary
coupling before learned inference.

[annotation_review.json](annotation_review.json) records a separate logical
review by the **same coding agent**, not an independent annotator, clinician or
human IAA assessment. It corrected an ambiguous medication-conflict label:
concurrent medications were replaced by an explicit positive report versus
denial, without a correction resolving the contradiction. The interrupted first
attempt is retained in `run-01/`, including its original source and first failed
generation. Only `run-02/` and `run-03/` are complete 20-dialogue runs on the
revised source. These cases cannot later serve as confirmatory tests.

## Observed results

Failures count as wrong on all five fields; denominators never exclude them.
Raw means strict state/span proposals evaluated without the evidence binder.
Checked means the adapter's contract output, before the separate note check.

| Run / stage | Inference failures / 20 | State correct / 100 | Grounded exact fields / 100 |
|---|---:|---:|---:|
| Rules prompt, raw | 20 | 0 | 0 |
| Rules prompt, checked | 20 | 0 | 0 |
| Two worked examples, raw | 15 | 24 | 24 |
| Two worked examples, checked | 14 | 19 | 19 |

The rules-only run took 76.758 seconds after provenance hashing; the worked-example
run took 131.215 seconds, including model loading and pipeline work. Generation
and per-encounter total times are stored separately; their difference also
includes loading on the first case. Concurrent local tests affected throughput,
so these are observed runtime measurements, not a speed benchmark. No generation
hit the 256-token cap. CPU PyTorch 2.11.0 / Transformers 5.5.3 was used.

The prompt revision uses two additional invented demonstrations, not pilot
answers. This was a response to exposed development failures, not a preregistered
or independently confirmed treatment comparison. Prompts and source snapshots
are preserved, and failed outputs were never silently repaired.

| Worked-example final path | Canonical / 10 | Paraphrase / 10 |
|---|---:|---:|
| Checked draft with useful content, human review required | 2 | 0 |
| Checked draft abstaining from all assertions | 2 | 2 |
| Failed inference, no note | 6 | 8 |

Both useful drafts (`pilot-00-canonical`, `pilot-01-canonical`) have five exact,
correct supported claims. Their paired paraphrases have correct raw proposals
but are quarantined by the legacy question binder. The other successful draft
projections preserve uncertainty/review evidence and assert no content. Passing
the note checker is a bounded software property, not clinical validation.

Raw/checked failure counts differ because the adapter can quarantine a
syntactically parsed but contract-invalid raw state (for example an absence
proposal for severity) to uncertainty before constructing its final output.
It does not credit that raw proposal as correct. Full state-confusion tables,
per-field exact-evidence errors and failure categories remain separate in
`raw_evaluation.json` and `checked_evaluation.json` for each complete run.

## Decision and next work

Keep this integration as an explicit development compact option. Do not promote
this checkpoint or prompt to P1 admission. The immediate engineering work is
constrained, per-field state/span generation with exact source selection, then
a separately tested source-context verifier that handles question paraphrases.
More GPU time alone will not repair the observed format and verifier failures.
Any revised decoder/verifier must retain conflicts, uncertainty and abstentions
and be tested on additional development examples before a new scientific screen.

H6/H7 artifacts, terminal decisions, consumed evaluations and freeze tags are
unchanged. This pilot is a separate comparison group and is not pooled with
historical compact, OpenResearch, C2, H6 or invalid-gold H7 scores.

## Reproduce

```sh
python3 scripts/run_p1_pilot.py
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 python3 scripts/run_p1_pilot.py \
  --checkpoint /absolute/path/to/pinned/local/snapshot \
  --prompt-style fewshot --output /new/development/result/directory
python3 -m nanoscribe.scribe --transcript /absolute/path/to/transcript.txt \
  --compact-checkpoint /absolute/path/to/pinned/local/snapshot --format text
python3 -m pytest nanoscribe/test_compact.py scripts/test_run_p1_pilot.py -q
```

Model loading is lazy/offline, uses no remote model code, and requires an existing
local checkpoint. No targets, fixture specs or gold evidence enter inference.
