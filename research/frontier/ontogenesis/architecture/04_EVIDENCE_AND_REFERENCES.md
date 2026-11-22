# 04 — Evidence register, primary references, boundaries and open verification work

**Cutoff:** 2026-11-21. This register distinguishes: **[N]** Nano LM repositories and uploaded work; **[P]** published/official literature; **[H]** newly stated hypotheses; **[M]** elementary mathematical propositions and numerical checks. Primary-source identities and the narrow claims summarized below were rechecked on 2026-11-21. Inspection depth varies by source: abstract, official documentation, model card, author repository, selected methods text, or primary search/PDF excerpt. The [independent source audit](08_LITERATURE_AUDIT.md) records the evidence depth, retrieval limits, reference aliases and corrections for 36 sources. No literature result was independently reproduced. Most full methods and appendices, implementations, and data/model licenses remain unaudited.

## 1. Repository and historical data provenance

**Additive current research evidence:**

| ID | Source | Scope | Next verification |
|---|---|---|---|
| N-11 | `checks/transport_anchor_pilot.py` + `results/transport_anchor_pilot_v0.json` (new local 2026-11-21) | Five-seed synthetic 16D linear basis migration, not a neural-language experiment | Replicate controlled drift in learned LM features with total-cost accounting |


| ID | Source | What is actually supported | Priority verification |
|---|---|---|---|
| N-01 | [Project charter](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/master/docs/PROJECT_CHARTER.md) | Native general-purpose LM is research goal; former NanoScribe is history | Read actual current branch when merging |
| N-02 | [Active state](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/master/docs/ACTIVE_NOW.md) | Current gate, private-data exclusion and experiment-scoped costly compute authorization | Verify environment and latest permissions before paid work |
| N-03 | [Execution plan](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/master/docs/EXECUTION_PLAN.md) | Baselines, evaluation gate, data and compute requirements | Exact run manifests |
| N-04 | [PR #194 merged](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194) | 5M code dense 4.65 vs M21 5.02 bits/token; copyable M21 edge | Recreate actual wall time per seed and evaluate text |
| N-05 | [Research ledger](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/LEDGER.md) | Historic proposed and disproven mechanisms and branch map | Several local branches and uncommitted V4 not checked |
| N-06 | [Investigation](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/INVESTIGATION.md) | Multi-phase acquisition, routing, retention probes; time-specific notes | Registered raw data and independent rerun |
| N-07 | [Discovery cycle 1](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/theory/DISCOVERY-CYCLE-1-20261121.md) | Referee corrected claims about exceptions and knowledge loss | Underlying scripts and newer controls |
| N-08 | [Build plan](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/BUILD-PLAN.md) | Existing tokenizer, corpus, laptop/TPU figures as agent-reported | Live TPU cost, corpus licenses and checksum |
| N-09 | [Nested registered protocol](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/PROTOCOL-NESTED.md) | Prespecified arms and five-seed rules | Full results; no selective pooling |
| N-10 | Uploaded `Nano_LM_Expanded_Master_Research_Blueprint_2026-11-21.md` | Prior 19k-word narrative research and engineering synthesis | It is a proposal, not ground-truth implementation status |

## 2. References relevant to architecture and expressivity

| ID | Research | Main point and scope (author-reported) | How Nano LM should use it |
|---|---|---|---|
| P-01 | [Qwen3.5 architecture documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) (2026) | 3:1 Gated DeltaNet / full attention is a real hybrid | Reference and code; not an optimal ratio proof |
| P-02 | [Mamba-3: Improved Sequence Modeling using State Space Principles](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html) (ICLR 2026) | Recurrence modifications and MIMO report gains in state tracking and LM | Benchmark optimized state mixer and decoding, not asymptotic estimates alone |
| P-03 | [Kimi Linear](https://arxiv.org/abs/2510.26692) (2025) | KDA+MLA hybrid shows author-reported advantages at 48B total/3B active | Implementation comparison; **not** a tested 3B total parameter backbone |
| P-04 | [A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention](https://proceedings.mlr.press/v306/ye26h.html) (ICML 2026) | Certain sequential-composition tasks need additional full attention under explicit bounds | Hard-negative family for hybrid claims; theorem is not universal |
| P-05 | [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers (LOTUS)](https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/) (2026) | Recurrent-depth latent reasoning plus gold intermediate CoT-token supervision, with reported 3B-scale results | Compare recurrence and supervision separately; charge gold-trace/teacher preparation and thought-phase versus end-to-end costs |
| P-06 | [Attention Residuals](https://arxiv.org/abs/2603.15031) (2026) | Author-reported depth-wise learned residual improvements | Measure added depth memory and training quality |
| P-07 | [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/) (ACL 2025) | Adaptive byte patches can be competitive with tokenizer models | Compare scored **bytes** and complete decoder costs |
| P-08 | [When Copying Is Hard / CopyGen](https://arxiv.org/abs/2610.28393) (2026) | Frozen backbone plus a trained copy-control plugin; the author repository documents plugin training | Augmented decoding comparator: count plugin parameters, training, calibration, source access and runtime |
| P-09 | [Memory Layers at Scale](https://proceedings.mlr.press/v267/berges25a.html) (ICML 2025) | Sparse memory capacity can improve some tasks per matched FLOPs | Include memory parameters and retrieval costs, don't infer lifelong learning |
| P-10 | [Titans: Learning to Memorize at Test Time](https://research.google/pubs/titans-learning-to-memorize-at-test-time-2/) (NeurIPS 2025) | Learnable neural memory with attention on long inputs | Prior art to any test-time memory claims |
| P-11 | [Nested Learning: The Illusion of Deep Learning Architectures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) (NeurIPS 2025) | Multiple timescale optimization framework | Compare actual retention, update cost and stability |

## 3. Continual learning, representation and plasticity

| ID | Primary reference | Evidence scope | Nano LM challenge |
|---|---|---|---|
| P-12 | [Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective](https://proceedings.mlr.press/v267/kim25p.html) (ICML 2025) | Measures representation discrepancy and forgetting | **Transport algebra is not novel**; test learnability and cost |
| P-13 | [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) (Nature 2024) | Plasticity loss in sequential tasks; continual backprop helps in tested settings | Distinguish ability to learn from retention |
| P-14 | [Disentangling the Causes of Plasticity Loss in Neural Networks](https://proceedings.mlr.press/v274/lyle25a.html) (2025) | Normalization + weight decay effective in several nonstationary settings | Must beat a modern simple plasticity control |
| P-15 | [Heads collapse, features stay: Why Replay needs big buffers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9b9521a6c8ead802f1b5dbd8cca34918-Abstract-Conference.html) (ICLR 2026) | Feature retention differs from classifier/output retention | Probes require unseen-task memorization controls |
| P-16 | [Provable Effects of Data Replay in Continual Learning](https://proceedings.mlr.press/v300/ding26b.html) (2026) | Full replay can still fail in a particular theoretical task family | Never describe replay as absolute guarantee |

## 4. Hardware, systems and evaluation references

| ID | Primary reference | Useful principle | Boundary |
|---|---|---|---|
| P-17 | [FlashAttention-4](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html) (MLSys 2026) | Physical scheduling and data movement yield speed | B200 measurements; not laptop/A100 speed claims |
| P-18 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1) (SOSP 2023) | KV paging can reduce fragmentation | Runtime throughput ≠ intrinsic model capability |
| P-19 | [SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) (NeurIPS 2024) | Prefix and execution reuse | Measure workload dependence and cache write cost |
| P-20 | [Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) (2026) | Contamination and flawed tests undermine conclusions | Fresh, independently checked tasks |
| P-21 | [Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) (2026) | SWE-bench Pro also contains many flawed tasks; earlier adoption advice retracted | Do not elevate Pro as automatically reliable |
| P-22 | [SmolLM3 training and data](https://huggingface.co/blog/smollm3) (2025) | Public compact-model pretraining uses very large token budgets | Do not equate 15M/125M pilots to frontier pretraining |
| P-23 | [Attention Residuals implementation](https://github.com/MoonshotAI/Attention-Residuals) | Concrete code and author tables | Check precise code license, weights, hardware |
| P-24 | [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | LM configuration inside a multimodal checkpoint; 4B is the language-model label | Pin the exact revision and inventory all tensors; the family name does not establish a strict 4B total-checkpoint limit |

## 5. Database, versioning and information transport analogies

- **Event sourcing / temporal databases:** use explicit revisions, causal time, provable `as-of` semantics and source provenance. This is a systems correctness tool, not a proof that the neural model knows a fact.
- **Solana parallel transaction scheduling:** the [Sealevel explainer](https://solana.com/id/news/sealevel---parallel-processing-thousands-of-smart-contracts) motivates dependency-aware parallel execution when read/write sets are known. The transfer to model updates is uncertain because gradient writes generally touch shared parameters and commutativity is not guaranteed.
- **Merkle DAG/content addressing:** can verify reproducible data/checkpoint custody, not semantic truth.
- **CRDTs:** defined conflict resolution may help concurrent exact-record edits, but it cannot resolve domain truth or intentional last-writer-wins updates without a policy.
- **Roofline performance:** computational advantage must be measured as actual FLOPs, bytes moved, power, runtime and quality, not performance claims borrowed from the original system.

## 6. Primary-source methodology and unfinished audits

**Historical repository review recorded by V2:** GitHub charter, active state, execution plan, PR #194, research branch ledger, investigation, discovery-cycle review, build plan and nested protocol. **Current primary-literature audit:** identities and stated narrow claims for P-01–P-24 plus added sources P-25–P-36, with source-specific inspection depth in [08](08_LITERATURE_AUDIT.md). This distinction preserves when repository evidence was inspected; the literature audit itself did not replay Nano training logs. **Not checked:** full mathematical derivations of every contemporary paper, all primary code implementations, license terms for every external model, hidden/restricted data sources, every local worktree, underlying raw training logs, precise current GPU quotas or billing. For the ICML hybrid theorem, rely only on its described assumptions and task family until the full appendix is analyzed.

**Research integrity checklist:** paper publication/title/date/authors; exact theorem hypotheses; baseline data/training/runtime; code and model licenses separately; benchmark contamination; relevance to 15M vs 3B scale; whether the published advantage needs teacher training or external retrieval; provenance of every claimed Nano result.

## 7. Source vs inference labeling

- **Repository-reported observation:** a claim recorded in the repository with a result location, possibly with unresolved execution/seed issues.
- **Published paper claim:** author-reported; not independent Nano LM replication.
- **Mathematical:** conditional result with an explicit proof; 'verified by numerical checks' means only that illustrative calculations succeeded.
- **Architecture hypothesis:** proposed mechanism and falsifiable outcomes, not evidence of effectiveness.
- **Research strategy:** managerial decision tied to current risks/resources; may be revised without changing past results.

## 8. 2026-11-21 v2 primary-source audit (newly checked)

This table retains the V2 aliases for historical continuity. Their public primary identities and narrow claims were rechecked in the current [source audit](08_LITERATURE_AUDIT.md). Most remain abstract-level evidence, and none was independently reproduced. Use the canonical IDs and alias mapping in the audit when adding new claims; a successful URL lookup is not a full methods review.

| ID | Primary source | Specific relevance | Limitation for Nano LM |
|---|---|---|---|
| V2-P01 | [Qwen3.5 official model-family documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5) and [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/main/README.md) | 3:1 Gated DeltaNet/full attention, actual vocabulary/width/layer configuration | Model-family configuration is not an optimal ratio or a strict total-checkpoint parameter bound |
| V2-P02 | [Ye et al., ICML 2026](https://proceedings.mlr.press/v306/ye26h.html) | Scope-limited formal expressiveness separation in sequential composition | Does not show every hybrid worse or address model training efficiency |
| V2-P03 | [Kim et al., ICML 2025](https://proceedings.mlr.press/v267/kim25p.html) | Representation-discrepancy measurements | Does not demonstrate a learned state-migration algorithm that improves Nano LM |
| V2-P04 | [Anthes et al., Lifelong Learning Agents 2025](https://proceedings.mlr.press/v274/anthes25a.html) | Task-relevant readout subspaces and plasticity/stability diagnostics | Direct prior art for linear/nonlinear subspace ideas, so no novelty presumption |
| V2-P05 | [Lyle et al., Lifelong Learning Agents 2025](https://proceedings.mlr.press/v274/lyle25a.html) | Multiple plasticity-loss causes; normalization + weight decay in their settings | Not a knowledge-retention cure |
| V2-P06 | [Zhang et al., Lifelong Learning Agents 2025](https://proceedings.mlr.press/v274/zhang25a.html) | Separate present, past and cross-task consolidation objectives | Different unsupervised continual learning setting |
| V2-P07 | [Berges et al., ICML 2025](https://proceedings.mlr.press/v267/berges25a.html) | Trainable sparse memory with capacity-per-FLOP advantages in tested conditions | Extra memory parameters count toward model size; no free sample efficiency claim |
| V2-P08 | [LOTUS preprint, 2026](https://arxiv.org/abs/2606.31779) | Looped latent reasoning with intermediate supervision | Can't attribute its result to looping alone |
| V2-P09 | [CopyGen preprint, 2026](https://arxiv.org/abs/2610.28393) | Selective exact span-copy controls using a trained plugin around a frozen backbone | Augmented-system comparator; include plugin training, capacity and runtime |
| V2-P10 | [Nested Learning, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) | Multiple learning timescales and learned update logic | Published hypothesis, needs exact implementation/compute comparison |
| V2-P11 | [Dohare et al., Nature 2024](https://www.nature.com/articles/s41586-024-07711-7) | Loss of plasticity under prolonged nonstationarity | Separate from factual forgetting and target-task transfer |

**Important historical evidence review:** [PR #194](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/194) contains a time-*calibrated* rather than perfectly time-equal 5M comparison; do not call 4.65 vs 5.02 a robust equal-clock-speed frontier result. [INVESTIGATION.md](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/INVESTIGATION.md) separately records one-seed memory saturation and three-seed write-time routing, which must not be pooled into one finding. [DISCOVERY-CYCLE-1](https://github.com/wwk5q8z6kk-bit/nano-lm/blob/research/ontogenesis-phase1/research/frontier/ontogenesis/theory/DISCOVERY-CYCLE-1-20261121.md) overturns the earlier broad exception-collateral story. A citation to one of these is not a replay of its raw results.


## 9. Direct prior art and sequential-editing controls added in this pass

The following additions close bibliography and comparison-design gaps. They do not promote an architecture or authorize a training run. Full source-by-source inspection status is in [08_LITERATURE_AUDIT.md](08_LITERATURE_AUDIT.md), with a [machine-readable index](results/literature_audit_20261121.json).

| ID | Primary reference | Relevance to Nano LM | Scope boundary |
|---|---|---|---|
| P-25 | Daniel Anthes, Sushrut Thorat, Peter König, Tim C Kietzmann (2025), [Keep moving: identifying task-relevant subspaces to maximise plasticity for newly learned tasks](https://proceedings.mlr.press/v274/anthes25a.html) | V2-P04/P04 source; studies readout-relevant versus null activation subspaces, linear interventions and nonlinear diagnostic approximation. | Activation-space diagnostics are not the same as exact parameter-Jacobian update projection. Cite OGD as direct precedent for the latter. |
| P-26 | Yipeng Zhang, Laurent Charlin, Richard Zemel, Mengye Ren (2025), [Integrating Present and Past in Unsupervised Continual Learning](https://proceedings.mlr.press/v274/zhang25a.html) | V2-P06/P06 source; distinguishes present/past/cross-task objectives and reports Osiris on unsupervised continual learning. | Useful objective separation; no demonstrated Nano LM forward transfer or a theorem about all continual learners. |
| P-27 | Mehrdad Farajtabar, Navid Azizan, Alex Mott, Ang Li (2020), [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html) | Direct prior art for projecting updates against prior-output gradients. Methods section explicitly uses local linearization and stored gradients as current-gradient proxies. | Add to theorem2 provenance. Old Jacobian drift and subset approximations matter; abstract language should not be promoted to global exact preservation. Only relevant methods sections inspected, not independent replication. |
| P-28 | Alex Graves (2016), [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983v6) | Introduces learned number of recurrent computation steps; publication revision in2017. | Direct prior art for adaptive computation. Abstract itself reports no large character-LM performance gain; stopping flexibility alone is not sufficient. |
| P-29 | Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser (2019), [Universal Transformers](https://arxiv.org/abs/1807.03819v3) | ICLR2019 paper identifies recurrent-depth attention and per-position halting. | Direct precedent for looped attention; conditional expressivity is distinct from finite-resource training efficiency.2018 preprint,2019 conference. |
| P-30 | Andrea Banino, Jan Balaguer, Charles Blundell (2021), [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407v2) | Learns adaptive computation under accuracy/cost/generalization objectives. | Relevant learned-halting baseline, not proof an optimal Bellman policy is available from current state or training labels. |
| P-31 | Solana Foundation (2019), [Sealevel — Parallel Processing Thousands of Smart Contracts](https://solana.com/id/news/sealevel---parallel-processing-thousands-of-smart-contracts) | Article explains parallel execution using declared transaction read/write sets. | Retain only as a dependency-scheduling analogy. It is not evidence that neural gradients commute or updates are local. |
| P-32 | Akshat Gupta, Anurag Rao, Gopala Anumanchipalli (2024), [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/) | Sequential ROME/MEMIT evaluation measures editability, earlier-edit forgetting and downstream capability degradation. | A successful edit or immediate locality check is not a durable-update result. The tested editors/models do not imply every future editor must fail. |
| P-33 | Xin Ma, Wei Chen, Qi Liu, Derong Xu, Zhi Zheng, Tong Xu, Enhong Chen (2026), [More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing](https://proceedings.mlr.press/v306/ma26f.html) | ICML2026 paper derives StableEdit from running-statistic normalization and ridge-regression analysis and reports long-horizon editing results. | Full theorem assumptions, source code and compute overhead were not independently audited. Lifelong normalization here is a method-specific update mechanism, not generic LayerNorm or universal lifelong-learning proof. |
| P-34 | Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov (2022), [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262v5) | NeurIPS2022 ROME paper describes rank-one FFN edits and evaluates specificity/generalization for factual associations. | A localized edit control is useful for update locality. Single-edit efficacy does not prove resistance to later edits or continued pretraining. |
| P-35 | Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau (2023), [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229v2) | MEMIT reports editing thousands of associations in GPT-J6B and GPT-NeoX20B. | Large-batch editing is not indefinite continual acquisition. Exact training statistics, model sizes and edit budgets matter; the retrieved v2 is dated2023 after the2022 initial preprint. |
| P-36 | Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, Tat-Seng Chua (2025), [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/html/2410.02355v4) | ICLR2025 paper provides direct nullspace-editing prior art; its ideal algebra preserves a linear layer's outputs on fixed protected keys. | Practical projection retains covariance eigenvalues below1e-2, not an exact nullspace in general. Estimated keys, later upstream drift and sequential layer edits prevent upgrading the fixed-key identity to a universal whole-LM guarantee. |

**Comparison design implied by this literature.** Test immediate edit acquisition, immediate locality, delayed edit durability, future acquisition and broad model quality separately. A replay control that protects other facts immediately does not establish persistence of the corrected fact. For a native model, use a feasible fine-tuning/replay arm and a localized or projected edit arm, counting their statistics, anchor data, preprocessing and update work. Treat modern methods such as StableEdit as candidates for replication after their exact assumptions and implementation are inspected.

**Novelty boundary.** Linear conjugacy, fixed-key nullspace identities, gradient projection, recurrent depth and learned halting all have established mathematical or algorithmic precedents. Any Nano contribution must lie in a specified implementation, a new proved result with explicit assumptions, or a measured improvement over the appropriate controls.

