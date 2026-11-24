# TPU Research Cloud: application draft (Nano LM)

Status: draft for the owner to complete and submit. Applicant fields are blank because they need the owner's identity. Programme conditions were read on the TRC page on 2026-11-24; grant size, duration and approval criteria are not stated there and must be confirmed in the application form.

**Applicant:** [name] · **Affiliation:** [affiliation or "independent researcher"] · **Contact:** [email]

## Research question

Can a language model of about 3–4 billion parameters match much larger models on a defined set of capabilities by (a) inheriting knowledge from open pretrained models through staged architecture conversion and distillation, (b) replacing most attention with recurrent layers, and (c) adding two mechanisms aimed at the known weaknesses of that conversion: a small budgeted exact memory and adaptive recurrent depth with a convergence-based stopping rule? Quality is reported against training and inference cost.

## What exists and has been measured

- A JAX training stack with data parallelism, Muon optimiser, resumable checkpoints and per-run code hashes. It trained dense models from 14M to 386M parameters on a 10B-token corpus on Cloud TPU v5e (measured: 106k tokens per second at 386M on four v5e chips).
- A benchmark harness calibrated against published numbers, with documented variation from precision, hardware, prompt setup and seeds, and a methodology audit whose findings are published in the repository.
- Mechanism code and tests: Gated DeltaNet recurrence, budgeted memory with several write policies, a looped block with halting. Experiments report every seed, including failures.
- A verified GPU conversion path: recurrent kernels agree with a float32 reference (relative error 5e-7 in float32, 8e-4 in float16).

## What TPU access would be used for

| Stage | Work | Estimated TPU use (assumptions: 36% of 197 TFLOPS per v5e chip, as measured on our dense runs) |
|---|---|---|
| Kernel | A chunked JAX/Pallas implementation of the recurrent layer and its benchmark | tens of chip-hours |
| Scale test | 1.5B student from a 7B teacher, 20B tokens | about 1,900 chip-hours |
| Flagship | 3.5B student from an 8B teacher, 20B tokens | about 2,900 chip-hours |

The recurrent layer has no TPU kernel today. This is the main technical risk, and the first allocation would go to resolving it.

## What will be shared

Open code and raw results for every experiment (already public in the project repository), registered experiment protocols, negative results, and a technical report. The project follows the Google AI Principles; it trains only on public data and open models with permissive licences (Apache 2.0, MIT) and handles no personal or private data.

## Feedback to Google

We will report on JAX/TPU support for recurrent sequence layers: kernel gaps, compile behaviour, memory use and throughput measurements.
