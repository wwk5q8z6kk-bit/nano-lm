# ONTOGENESIS mission assessment, 2026-11-20

Status: assessment and hypotheses. Conclusions and hypotheses are marked separately. Unreviewed; published results are cited from memory or search summaries unless a critic section says otherwise.

## 1. Capability gaps of current language models

| Capability | Do current models handle it? | Evidence (published) | Real gap? |
|---|---|---|---|
| Language representation and generation | Yes | — | No |
| Retrieval inside the context window | Yes, by scanning | induction-head studies | Cost only |
| Few-shot learning in context | Yes | in-context learning literature | No |
| State tracking and long algorithmic chains | Weak at fixed depth; fixed by more depth or written-out steps | circuit-complexity results; length-generalisation studies | Partly |
| Compositional generalisation | Weak by default; strong with meta-learning curricula in ordinary Transformers | meta-learning-for-compositionality results | Mostly a training question |
| Facts stored in weights | Stored directionally: learning "A is B" does not give "B is A" | the reversal curse | **Yes** |
| Combining two stored facts without writing the middle step | Unreliable | latent two-hop studies | **Yes** |
| Sample efficiency | Poor: about 10^11–10^13 tokens against roughly 10^8 words for a person | small-data benchmarks | **Yes** |
| Learning that persists after training (new facts, corrections) | No; handled outside the model by retrieval or fine-tuning | continual-learning and model-editing literature | **Yes, and unsolved** |
| Calibration and knowing what it does not know | Partial | — | Yes |

Conclusion: the largest real gaps are in how knowledge gets into the weights and stays usable (directional storage, weak latent composition, sample inefficiency, no persistence). They are not in reading the context window.

## 2. What the programme has been overlooking

1. **We have studied the wrong memory.** Every mechanism so far (KEYSTORE, RELATOR, FILTRA, DEMOS's store, PATHWAY) is memory inside one context. The directive's central aim, turning information into better internal knowledge, is about the weights. The published evidence for explicit memory is also stronger on that side: trainable key-value memory layers are reported to match dense models given substantially more compute. [search summary, unverified]
2. **Sample efficiency is the dimension a small lab can win and measure.** It needs little compute, small-scale results are accepted as evidence, and the one positive anomaly we have (DEMOS learning the update task while dense did not) is a learning-speed effect.
3. **Training signal may matter more than architecture for several target capabilities.** Systematic compositional generalisation has been obtained in a standard Transformer by changing the curriculum. Objectives and curricula transfer to any architecture and are cheap to test.

## 3. Where I disagree with the direction

- **A from-scratch 2B frontier model is not reachable on the current budget,** by a factor of several hundred at published training budgets. The discovery of a principle is affordable; the frontier model is a separate funding question. I would stop writing 2B into the plan until a 125M result exists.
- **More architecture-on-paper is the lowest-yield activity we have.** Five designs have gone to independent review; five were reduced to "components published, advantage unproven". One seven-minute experiment produced the only favourable signal.
- **Ten capabilities at once is not a testable target.** Two of them (persistent learning without retraining; a model improving its own computation) have no known training recipe anywhere.
- **Scope has changed faster than experiments finish.** Phase 1 is still at pilot stage after five directives.

## 4. Three leads (hypotheses, not conclusions)

### A. Emergent Computational Substrate, made concrete
The abstract form S' = T(S, x, O), O' = G(S, O), y = D(S, O) describes any recurrent meta-learning system, so as stated it cannot be falsified. A concrete instance: O is a set of low-rank weight updates ("operations") that the forward pass can create, select and add together; G writes a new operation when a task is solved; later tasks reuse it.
- Relatives: fast weights, hypernetworks, library learning in program synthesis, modular networks.
- Strongest simpler alternative: a standard Transformer trained with a meta-learning curriculum, which already shows reuse in context.
- Decisive experiment (REUSE): tasks are hidden compositions of unary functions on a small domain. Train on some compositions; test on unseen compositions and on new primitives shown by a few examples. Compare a model that can store and recall operations with a parameter-matched model that cannot, and with the meta-trained Transformer. Measure examples needed to reach 90%, accuracy on longer compositions, and compute. Pass only if the operation store beats both on at least two of the three.

### B. Why does DEMOS learn updates fast? (follow the anomaly)
Pilot, one seed: DEMOS 98.3%, dense 20.4% on one-step lookup with updates, in equal time. Candidate causes, cheapest first: the dense baseline is mis-tuned; DEMOS receives an explicit time coordinate; unnormalised interaction with mass gives "newer suppresses older" a direct path; the separation of store and probe is a helpful prior.
- Strongest simpler alternative: a tuned dense or looped Transformer, and the message-passing arm.
- Decisive experiment: the tuned baselines now queued, then ablations of mass dynamics, time coordinate and normalisation, three seeds. If a tuned dense model matches DEMOS, the lead is closed.

### C. Symmetric binding in the weights
Hypothesis: facts are stored directionally because feed-forward layers act as one-way key→value memories. A parametric memory whose bindings are symmetric (either side retrieves the other) would not show the reversal curse and should compose stored facts more reliably.
- Relatives: memory layers, hetero-associative memories, reverse-order data augmentation.
- Strongest simpler alternative: the same dense model trained with reversed-order augmentation.
- Decisive experiment (REVERSE): train small models on synthetic facts stated in one direction only, with other entities shown in both; test the reverse direction and two-fact composition. Arms: dense, dense with augmentation, dense with a symmetric memory layer. Pass only if the memory layer beats augmentation at equal training tokens.

These three address different gaps (reuse of operations, learning speed on changing facts, relational storage in weights). None is claimed to be new.

## 5. How the one model gets built

Evidence chooses the parts. A mechanism enters the unified model only after it beats its strongest simpler alternative at small scale across three seeds, and then again inside a small language model at equal training time.

| Stage | Scale | Gate |
|---|---|---|
| 0 | Laptop, synthetic | Phase 1, REVERSE, REUSE: three seeds each |
| 1 | 10–20M language model, laptop or free GPU | sample-efficiency curve against dense at equal tokens and equal time |
| 2 | 125M, TPU (needs approval) | repeatable gain, three seeds |
| 3 | 350M | one pilot is what the credits allow |
| 4 | 1B–2B | separate funding decision |

## 6. Open items

- Phase 1 confirmatory runs: protocol not yet committed; S2 and M21 not ported.
- L0: needs a corpus of long documents; none is available locally.
- Independent critic for this document: pending.
