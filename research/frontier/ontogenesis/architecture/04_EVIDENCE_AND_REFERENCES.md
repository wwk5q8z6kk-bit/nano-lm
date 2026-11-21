# 04 — Evidence register, primary references, boundaries and open verification work

**Cutoff:** 2026-11-21. This register distinguishes: **[N]** Nano LM repositories and uploaded work; **[P]** published/official literature; **[H]** newly stated hypotheses; **[M]** elementary mathematical propositions and numerical checks. Literature claims below have been checked against the accessible official abstracts, full abstracts, author/model documentation or primary pages; a complete paper-by-paper methods replication, source-code audit and dataset-license legal assessment **has not** been carried out. Do not treat abstracts as validated independent replication.

### New evidence — synthetic migration demonstration, 2026-11-21

| ID | Source | Finding and classification | Explicit limitation |
|---|---|---|---|
| N-11 | Companion package `checks/transport_anchor_pilot.py`, `results/transport_anchor_pilot_v0.json`; [Proposition A.1](01_ARCHITECTURE_AND_THEOREMS.md) | Constructed 16D orthogonal basis migration; five seeds; generic inverse estimator reached 100% top-1 agreement from full-rank 16-anchor set | **Synthetic linear experiment only;** no model training, no language evidence, no demonstrated runtime/energy advantage; arbitrary full linear map assumption |

## 1. Repository and historical data provenance

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
| P-05 | [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers (LOTUS)](https://www.microsoft.com/en-us/research/publication/bridging-the-gap-between-latent-and-explicit-reasoning-with-looped-transformers/) (2026) | Intermediate supervision can make looped latents useful, 3B-scale results | Require supervised vs unsupervised loop controls |
| P-06 | [Attention Residuals](https://arxiv.org/abs/2603.15031) (2026) | Author-reported depth-wise learned residual improvements | Measure added depth memory and training quality |
| P-07 | [Byte Latent Transformer](https://aclanthology.org/2025.acl-long.453/) (ACL 2025) | Adaptive byte patches can be competitive with tokenizer models | Compare scored **bytes** and complete decoder costs |
| P-08 | [When Copying Is Hard / CopyGen](https://arxiv.org/abs/2610.28393) (2026) | Conditional exact-copy start, length and stop improves authors' cases | Strong post-hoc decoding competitor to M21-inspired copy heads |
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
| P-18 | [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://doi.org/10.1145/3600006.3613165) (SOSP 2023) | KV paging can reduce fragmentation | Runtime throughput ≠ intrinsic model capability |
| P-19 | [SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) (NeurIPS 2024) | Prefix and execution reuse | Measure workload dependence and cache write cost |
| P-20 | [Why we no longer evaluate SWE-bench Verified](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) (2026) | Contamination and flawed tests undermine conclusions | Fresh, independently checked tasks |
| P-21 | [Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) (2026) | SWE-bench Pro also contains many flawed tasks; earlier adoption advice retracted | Do not elevate Pro as automatically reliable |
| P-22 | [SmolLM3 training and data](https://huggingface.co/blog/smollm3) (2025) | Public compact-model pretraining uses very large token budgets | Do not equate 15M/125M pilots to frontier pretraining |
| P-23 | [Attention Residuals implementation](https://github.com/MoonshotAI/Attention-Residuals) | Concrete code and author tables | Check precise code license, weights, hardware |
| P-24 | [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | Parameter/tokenizer details and config | Do not use exact size/vocab without loading model card |

## 5. Database, versioning and information transport analogies

- **Event sourcing / temporal databases:** use explicit revisions, causal time, provable `as-of` semantics and source provenance. This is a systems correctness tool, not a proof that the neural model knows a fact.
- **Solana parallel transaction scheduling:** the [Sealevel explainer](https://solana.com/id/news/sealevel---parallel-processing-thousands-of-smart-contracts) motivates dependency-aware parallel execution when read/write sets are known. The transfer to model updates is uncertain because gradient writes generally touch shared parameters and commutativity is not guaranteed.
- **Merkle DAG/content addressing:** can verify reproducible data/checkpoint custody, not semantic truth.
- **CRDTs:** defined conflict resolution may help concurrent exact-record edits, but it cannot resolve domain truth or intentional last-writer-wins updates without a policy.
- **Roofline performance:** computational advantage must be measured as actual FLOPs, bytes moved, power, runtime and quality, not performance claims borrowed from the original system.

## 6. Primary-source methodology and unfinished audits

**Checked in this research pass:** GitHub charter, active state, execution plan, PR #194, research branch ledger, investigation, discovery-cycle review, build plan, nested protocol; official pages for P-01 to P-24 where accessible through search. **Not checked:** full mathematical derivations of every contemporary paper, all primary code implementations, license terms for every external model, hidden/restricted data sources, every local worktree, underlying raw training logs, precise current GPU quotas or billing. For the ICML hybrid theorem, rely only on its described assumptions and task family until the full appendix is analyzed.

**Research integrity checklist:** paper publication/title/date/authors; exact theorem hypotheses; baseline data/training/runtime; code and model licenses separately; benchmark contamination; relevance to 15M vs 3B scale; whether the published advantage needs teacher training or external retrieval; provenance of every claimed Nano result.

## 7. Source vs inference labeling

- **Repository-reported observation:** a claim recorded in the repository with a result location, possibly with unresolved execution/seed issues.
- **Published paper claim:** author-reported; not independent Nano LM replication.
- **Mathematical:** conditional result with an explicit proof; 'verified by numerical checks' means only that illustrative calculations succeeded.
- **Architecture hypothesis:** proposed mechanism and falsifiable outcomes, not evidence of effectiveness.
- **Research strategy:** managerial decision tied to current risks/resources; may be revised without changing past results.