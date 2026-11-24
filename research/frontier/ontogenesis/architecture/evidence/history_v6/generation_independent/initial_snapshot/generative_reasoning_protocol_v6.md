# V6 generated relational evaluation — frozen protocol

**Status:** local evaluation instrument. No checkpoint, corpus or learned model is executed by its control checks. The machine-readable recipe is `generative_reasoning_protocol_v6.json`; its SHA256 is recorded in every generated dataset and report. The recipe was written before the first generated check. This is a local freeze, not an independently witnessed preregistration.

## Question and scope

Can a candidate generate the answer entailed by visible relational facts, change its answer when the relevant fact changes, preserve it when fact order changes, and abstain when the inference path is absent? This instrument makes those requirements inspectable. It does not infer that a new architecture improves language modeling. The research cutoff is `3a38b5f1529a718838b5bc83a010574bfa9a4c6e`.

The two task families are direct lookup and two-hop composition. Entity and value identifiers are arbitrary, fresh and visible; a symbolic interpreter derives the answer from the facts. A two-hop case asks for the value of the destination of a named entity's pointer. The model is not asked to guess an independent hidden label.

| Record per base world | Visible change | Policy answer | Purpose |
|---|---|---|---|
| Full A | Original facts | A target | Answerable reference |
| Full B | Replace only the relevant value | Different B target | Same-query counterfactual |
| Permuted A | Reorder the same facts | A target | Order robustness |
| No context | Remove every fact | UNKNOWN | Answer-prior and abstention control |
| Broken chain | Remove queried assignment or pointer | UNKNOWN | Require the whole inference path |

Development has 16 worlds per family and evaluation has 32, for 96 worlds and 480 records with the default frozen recipe. Their world IDs, entity/value vocabularies and full-context fact sets are disjoint. Grammar templates are shared. Report the splits separately; the evaluation split is not for choosing prompts, decoders or checkpoints.

## Information boundary

The public JSONL stores an opaque coordinator ID and prompt. Candidate calls receive a frozen `CandidateView` with **one field: prompt**. They never receive world ID, split, condition, random seed, gold, original target, private oracle facts, or a choice list. The runner retains IDs for association. Candidate state resets per case and may not train, retrieve, use tools or retain earlier cases. A trusted coordinator validates the private oracle against a separate parser of the visible prompt.

This is an explicit data contract, not an operating-system sandbox. An adapter that deliberately reads oracle files violates it. A production run must isolate the candidate process and restrict its inputs if adversarial leakage is a concern. The current local checks do not claim such isolation.

## Generation, context and receipts

Use greedy decoding, at most 32 new tokens, and stop at the first newline or EOS. A budget-exhausted response remains in the denominator and is incorrect, even if its partial text happens to match. Strip only ASCII space, tab, CR and LF at the response boundary. The remaining response must be exactly the value identifier or `UNKNOWN`; explanations, punctuation, alternate casing, Unicode lookalikes and interior whitespace fail.

A real model run must count actual input token IDs using the model's tokenizer, including BOS and any other special tokens. Eligibility requires input tokens plus the full output allowance to fit the declared context limit. A token estimate or byte count cannot establish this. If it does not fit, record `context_ineligible`; never crop the prompt. The generic adapter runner verifies these counts and saves token IDs, hashes, stop reason and context policy. Binding a real NanoLM checkpoint and tokenizer remains a separate task.

For reproducibility, a model run identifies checkpoint, tokenizer, model config and adapter source by SHA256, as well as dataset and protocol hashes. The output records include the raw generated text and token IDs. Receipt validation establishes internal consistency, not independent proof that an external file was produced by the declared model.

## Metrics and complete denominators

The primary endpoint is **joint correctness on A and B within each base world**. Both required answers differ, so a fixed answer to the shared query cannot pass the pair. Aggregate separately by split and task family. Every registered world remains in the denominator, including missing outputs, invalid formatting, backend errors and ineligible contexts. Report exact counts and descriptive rates; no efficacy threshold or inferential claim is registered.

The original-target contextual gain compares A's correctness with the no-context output's correctness **against A's original answer**. UNKNOWN policy accuracy is a different quantity: no-context and broken-chain cases should abstain. Subtracting full-context answer accuracy from an UNKNOWN-control policy accuracy would compare different targets and is not contextual gain.

Also report condition-level policy accuracy, joint A/permutation correctness, raw same-answer consistency, and separate missing/error/censored/format-invalid counts. Agreement alone is not correctness; two identical wrong answers are consistent but fail the joint endpoint.

## Contamination and provenance

Dataset creation is deterministic from frozen seeds and refuses to overwrite an existing destination. The manifest records the protocol and generator source hashes, file hashes, exact world and record counts, planned prompt/context bytes, split overlap checks and the recipe timestamp. A loader revalidates the complete manifest and derives gold again from visible facts. Split overlap checks include IDs, full prompts, full fact sets and entity/value vocabularies, with a planted-positive-control test.

**Training-corpus contamination status remains NOT_CHECKED.** Fresh generated worlds and within-package duplicate checks do not establish what a model saw during training. Publishing evaluation cases can itself expose them. Corpus membership or paraphrase clearance requires actual corpus provenance and a separate audit.

## Local checks and execution

The local checks compare independent symbolic parsing with the private oracle, run constant/first-value/last-value/UNKNOWN controls, verify complete denominators and exercise malformed input and receipt handling. All are labelled `instrument_control`; none is a learned-model result.

From the architecture directory:

```bash
python experiments/generative_reasoning_v6.py generate --out /tmp/nanolm-v6-relations
python experiments/generative_reasoning_v6.py controls --dataset /tmp/nanolm-v6-relations --out /tmp/nanolm-v6-controls.json
python -m unittest discover -s experiments -p test_generative_reasoning_v6.py -v
python experiments/generative_reasoning_v6.py score --dataset /tmp/nanolm-v6-relations --predictions /path/to/predictions.jsonl --run-manifest /path/to/run_manifest.json --out /tmp/nanolm-v6-model-score.json
```

Each output path must be new. `controls` runs no language model and requires no corpus or paid compute. The Python `CandidateAdapter` interface supports later native-checkpoint integration; its prompt encoding and generation must agree exactly, including BOS, stopping and state reset. The checker cannot substitute for validating that real adapter on a real checkpoint.

