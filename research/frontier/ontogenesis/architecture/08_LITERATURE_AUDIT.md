# 08 — Primary-source audit and literature decision boundaries

**Audit date:** 2026-11-21, America/New_York. **Scope:** the P-01–P-24 register in `04_EVIDENCE_AND_REFERENCES.md`, aliases in `01_ARCHITECTURE_AND_THEOREMS.md` and `05_GAP_CLOSURE_AND_FORMAL_REVIEW.md`, plus missing direct prior art. The preserved V2 baseline was not modified. The working evidence register has been corrected to reflect this audit. This audit concerns bibliographic identity and whether each source supports the nearby narrow claim. It is not an experimental reproduction or full theorem-appendix review.

## Executive finding

All 24 existing primary-reference identities were recovered from primary pages, primary search results, or an author repository. The two added V2 continual-learning references (Anthes and Zhang) were also confirmed. The 2026 items — LOTUS, CopyGen, Ye's hybrid hierarchy, Mamba-3, Attention Residuals, FlashAttention-4, the replay papers and the two coding-benchmark audits — should **not** be removed as fabricated. The useful repairs are narrower scientific attribution, complete cost accounting, version pinning, and explicit inspection depth.

The most important additions are Orthogonal Gradient Descent, Adaptive Computation Time, Universal Transformers and PonderNet. They are direct precedents for projected updates, recurrent depth and learned halting. The existing register overemphasizes recent papers while omitting these closer foundations.

## Repairs that change decisions

| Finding | Why it matters | Concrete correction |
|---|---|---|
| CopyGen freezes its backbone but trains a plugin | A “frozen model” comparison can hide additional learned capacity and supervision | Include plugin parameters, training tokens, calibration and decoding work in the augmented-system budget; retain a plain frozen-backbone baseline |
| LOTUS uses gold intermediate reasoning supervision | A result cannot be attributed to looping alone when the training objective changes | Compare matched loop count, matched supervision and matched total cost. State that its reported advantage belongs to the full training recipe |
| Parameter-gradient projection has direct prior art in OGD | A new label does not make the nullspace construction new | Cite Farajtabar et al. AISTATS 2020 beside theorem 2, in addition to Anthes' activation-subspace analysis |
| Qwen's “4B” label describes the language component in a multimodal card | A nominal family label can conceal a larger full checkpoint | Inventory the exact checkpoint tensors and state whether vision, MTP, embeddings and auxiliary modules count; keep total and active parameter counts separate |
| Mutable documents mix architecture versions | Current Qwen documentation also discusses Qwen3.6 and multiple checkpoints | Pin package/config/repository revisions before implementing or benchmarking; a stable URL plus access date is not a content pin |
| Theory abstracts are not complete assumptions | Ye and Ding abstracts support narrow task-family conclusions | Keep theorem-family summaries; do not assert width/precision/unbounded-depth impossibilities until the full assumptions are extracted |
| Adaptive computation and recurrent depth already have established baselines | Choosing only recent papers weakens novelty and comparison claims | Add ACT, Universal Transformers and PonderNet as relevant prior art; choose realizable comparison arms, not merely citations |
| Evidence retrieval depth is uneven | “All checked” can conceal abstract-only or search-only access | Record an inspection-depth field per source and distinguish identity verification, methods review, implementation audit and independent reproduction |

These are research-design inferences from the cited sources and the Nano draft, not extra claims made by those papers.

## Primary reference register

Every entry below has **verified primary identity**. “Supported” means the stated narrow claim is present in the retrieved primary material. No entry has been independently reproduced by this audit.

| ID | Exact source and bibliographic identity | Inspection depth | Supported claim | Boundary or action |
|---|---|---|---|---|
| P-01 | Qwen Team / Hugging Face Transformers maintainers (2026). [Qwen3.5 model documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5) | official documentation | Verified 3:1 Gated DeltaNet/full attention stack in the current documentation. | Current docs also cover Qwen3.6 and several checkpoints; pin an exact package/config revision before implementation. A family ratio is not an optimum theorem. |
| P-02 | Lahoti et al. (2026). [Mamba-3: Improved Sequence Modeling using State Space Principles](https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html) | conference abstract | ICLR proceedings identify recurrence, complex state updates and MIMO; reported LM/state-tracking results exist. | Author-reported quality and latency are tested-scale/hardware results. No automatic advantage for a Nano implementation. |
| P-03 | Kimi Team (2025). [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692v2) | preprint abstract | KDA/MLA hybrid; abstract explicitly distinguishes 48B total from 3B active parameters. | Charge all weights, kernels, cache and training. It is not a 3B-total model. v2 is the retrieved revision. |
| P-04 | Xiaowei Ye, Xiaoyu He, Chao Liao, Chen Wu, Pinyan Lu (2026). [A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention](https://proceedings.mlr.press/v306/ye26h.html) | conference abstract | ICML2026 abstract describes a sequential function-composition separation between bounded-depth full and hybrid attention families. | Full theorem assumptions and appendix were not audited; PDF retrieval through the linked raw endpoint failed. Do not extrapolate to arbitrary task, width, precision or training cost. |
| P-05 | Ying Fan, Anej Svete, Kangwook Lee (2026). [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779v2) | preprint abstract | LOTUS uses recurrent-depth computation with gold intermediate CoT-token supervision and reports 3B-scale results. | This is not evidence for unsupervised looping alone. Teacher/gold-trace construction, supervision, padding and thought-phase latency must be accounted for separately. |
| P-06 | Kimi Team (2026). [Attention Residuals](https://arxiv.org/abs/2603.15031v1) | preprint abstract | Depth-wise attention over previous layer outputs; block form targets memory/communication cost. Large-scale report uses Kimi Linear. | Treat as a controlled residual-routing comparator. It does not establish small-model or hardware-independent superiority. |
| P-07 | Artidoro Pagnoni et al. (2025). [Byte Latent Transformer: Patches Scale Better Than Tokens](https://aclanthology.org/2025.acl-long.453/) | conference abstract | ACL paper describes entropy-based byte patches and FLOP-controlled scaling up to 8B parameters and 4T training bytes. | Compare the same scored byte strings and count all encoder, patcher and decoder work. Cross-tokenizer bits/token are not comparable. |
| P-08 | Jinghui Zhang, Lang Gao, Zongfang Liu, Ruihong Zeng, Zirui Song, Rui Yan, Kentaro Inui, Xiuying Chen (2026). [When Copying Is Hard: Copy-Constrained Decoding for Exact Span Reproduction](https://arxiv.org/abs/2610.28393) | primary search and author repository | Identity and abstract recovered by primary arXiv search; official CopyGen repository documents frozen backbone plus trained decoding plugin. | Direct arXiv open failed (DisabledError). Author repo https://github.com/mbzuai-nlp/CopyGen confirms plugin training and plugin.pt. Frozen backbone does not mean training-free system; include plugin parameters, training, calibration and runtime. |
| P-09 | Vincent-Pierre Berges, Barlas Oguz, Daniel Haziza, Wen-Tau Yih, Luke Zettlemoyer, Gargi Ghosh (2025). [Memory Layers at Scale](https://proceedings.mlr.press/v267/berges25a.html) | conference abstract | ICML paper reports trainable sparse key/value memory scaling and capacity/compute comparisons. | Reported experiments include up to 128B memory parameters. Sparse activation does not remove storage, retrieval or update costs; no lifelong-learning guarantee. |
| P-10 | Ali Behrouz, Peilin Zhong, Vahab Mirrokni (2025). [Titans: Learning to Memorize at Test Time](https://research.google/pubs/titans-learning-to-memorize-at-test-time-2/) | author publication page | Google publication page confirms NeurIPS2025 and neural test-time memory combined with attention. | Published long-context performance is not indefinite continual learning. Specify reset semantics and charge test-time updates. |
| P-11 | Ali Behrouz, Meisam Razaviyayn, Peilin Zhong, Vahab Mirrokni (2025). [Nested Learning: The Illusion of Deep Learning Architectures](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4309616aaed8e848009bc4a7ef73b493-Abstract-Conference.html) | conference abstract | NeurIPS abstract gives nested optimization, learned update rules and Hope as a tested module. | Framework and promising reported experiments are not a universal theorem, nor proof that Nano's current learner implements those mechanisms. |
| P-12 | Joonkyu Kim, Yejin Kim, Jy-Yong Sohn (2025). [Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective](https://proceedings.mlr.press/v267/kim25p.html) | conference abstract | Representation discrepancy metric and theoretical/empirical analysis on image continual-learning datasets. | Relevant to measuring representational change. It does not certify prediction-preserving migration of an LM's parameters, caches or optimizer state. |
| P-13 | Shibhansh Dohare, J. Fernando Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton (2024). [Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7) | publisher search and pdf excerpt | Publisher search/PDF recover Nature632:768–774, published21Aug2024; loss of ability to acquire differs from loss of old performance. | Publisher HTML open redirected to an inaccessible identity endpoint. Source findings are empirical within tested supervised/RL settings; no universal impossibility proof for gradient learners. |
| P-14 | Clare Lyle et al. (2025). [Disentangling the Causes of Plasticity Loss in Neural Networks](https://proceedings.mlr.press/v274/lyle25a.html) | conference abstract | PMLR274 paper reports multiple mechanisms and effective layer-normalization-plus-weight-decay controls. | Joint intervention does not isolate each factor. Maintaining learning speed is not factual retention. |
| P-15 | Giulia Lanzillotta, Damiano Meier, Thomas Hofmann (2026). [Heads collapse, features stay: Why Replay needs big buffers](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9b9521a6c8ead802f1b5dbd8cca34918-Abstract-Conference.html) | conference abstract | ICLR2026 source distinguishes representation separability from output/classifier forgetting. | Its asymptotic/theoretical assumptions are not fully reviewed here. Probe recoverability is not operational native-model retention. |
| P-16 | Meng Ding, Jinhui Xu, Kaiyi Ji (2026). [Provable Effects of Data Replay in Continual Learning: A Feature Learning Perspective](https://proceedings.mlr.press/v300/ding26b.html) | conference abstract | AISTATS2026 paper analyzes full replay in a multi-view task-incremental binary-classification model, including SNR and task order. | This is a scoped theoretical counterexample/analysis, not proof replay cannot work generally. Linked raw-PDF endpoint was not retrievable. |
| P-17 | Ted Zadouri, Markus Hoehnerbach, Jay Shah, Vijay Thakkar, Tri Dao (2026). [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html) | conference abstract | MLSys2026 source reports Blackwell-specific scheduling/pipelining and B200 BF16 comparisons. | Exact title should be retained. Speed numbers are kernel/hardware/workload specific, not a Nano laptop or A100 claim. |
| P-18 | Woosuk Kwon et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180v1) | preprint abstract | Original author preprint identifies KV memory fragmentation/sharing and vLLM serving results; SOSP2023 noted. | Original DOI returned403, so use accessible author preprint. Throughput depends on batch/workload and is distinct from model quality. |
| P-19 | Lianmin Zheng et al. (2024). [SGLang: Efficient Execution of Structured Language Model Programs](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html) | conference abstract | NeurIPS2024 source identifies RadixAttention cache reuse and structured generation optimizations. | Benchmark results are multi-call application throughput. Charge cache construction and report request distribution. |
| P-20 | OpenAI (2026). [Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | official research article | Feb23 article reports flawed tests and contamination in audited samples. | Use actual current article title. The59.4% figure concerns an audited27.6% subset selected for frequent failures, not the whole benchmark rate. |
| P-21 | OpenAI (2026). [Separating signal from noise in coding evaluations](https://openai.com/index/separating-signal-from-noise-coding-evaluations/) | official research article | Jul8 article reports agent/human audits of SWE-benchPro and explicitly retracts prior recommendation to adopt it. | 31-ish percent aggregate estimates are author-audit results, not every task invalid. Keep the later update with earlier reference; independently audit selected evaluation tasks. |
| P-22 | Hugging Face SmolLM team (2025). [SmolLM3: smol, multilingual, long-context reasoner](https://huggingface.co/blog/smollm3) | author technical blog | Jul8 2025 author release reports3B model and11T-token training. | Useful scale/cost context, not an estimate of minimum tokens required for Nano or proof all data rights are settled. |
| P-23 | MoonshotAI (2026). [Attention-Residuals implementation repository](https://github.com/MoonshotAI/Attention-Residuals) | author repository | Author repository exists and exposes implementation resources. | No immutable commit or license file was audited. Availability on GitHub is not license clearance or reproduction. |
| P-24 | Qwen Team (2026). [Qwen3.5-4B-Base model card](https://huggingface.co/Qwen/Qwen3.5-4B-Base) | official model card | Card gives LM4B, width2560, padded vocabulary248320,32layers,3:1 stack and tied LM output. | Card is multimodal; aggregate page displays5B Safetensors. Do not infer full checkpoint <4B total. Pin revision and inventory tensor counts. Generic card text describes pre/post-training despite Base name; resolve from exact release/config before claims. |
| P-25 | Daniel Anthes, Sushrut Thorat, Peter König, Tim C Kietzmann (2025). [Keep moving: identifying task-relevant subspaces to maximise plasticity for newly learned tasks](https://proceedings.mlr.press/v274/anthes25a.html) | conference abstract | V2-P04/P04 source; studies readout-relevant versus null activation subspaces, linear interventions and nonlinear diagnostic approximation. | Activation-space diagnostics are not the same as exact parameter-Jacobian update projection. Cite OGD as direct precedent for the latter. |
| P-26 | Yipeng Zhang, Laurent Charlin, Richard Zemel, Mengye Ren (2025). [Integrating Present and Past in Unsupervised Continual Learning](https://proceedings.mlr.press/v274/zhang25a.html) | conference abstract | V2-P06/P06 source; distinguishes present/past/cross-task objectives and reports Osiris on unsupervised continual learning. | Useful objective separation; no demonstrated Nano LM forward transfer or a theorem about all continual learners. |
| P-27 | Mehrdad Farajtabar, Navid Azizan, Alex Mott, Ang Li (2020). [Orthogonal Gradient Descent for Continual Learning](https://proceedings.mlr.press/v108/farajtabar20a.html) | abstract and methods excerpt | Direct prior art for projecting updates against prior-output gradients. Methods section explicitly uses local linearization and stored gradients as current-gradient proxies. | Add to theorem2 provenance. Old Jacobian drift and subset approximations matter; abstract language should not be promoted to global exact preservation. Only relevant methods sections inspected, not independent replication. |
| P-28 | Alex Graves (2016). [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983v6) | preprint abstract | Introduces learned number of recurrent computation steps; publication revision in2017. | Direct prior art for adaptive computation. Abstract itself reports no large character-LM performance gain; stopping flexibility alone is not sufficient. |
| P-29 | Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser (2019). [Universal Transformers](https://arxiv.org/abs/1807.03819v3) | preprint abstract | ICLR2019 paper identifies recurrent-depth attention and per-position halting. | Direct precedent for looped attention; conditional expressivity is distinct from finite-resource training efficiency.2018 preprint,2019 conference. |
| P-30 | Andrea Banino, Jan Balaguer, Charles Blundell (2021). [PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407v2) | preprint abstract | Learns adaptive computation under accuracy/cost/generalization objectives. | Relevant learned-halting baseline, not proof an optimal Bellman policy is available from current state or training labels. |
| P-31 | Solana Foundation (2019). [Sealevel — Parallel Processing Thousands of Smart Contracts](https://solana.com/id/news/sealevel---parallel-processing-thousands-of-smart-contracts) | official engineering article | Article explains parallel execution using declared transaction read/write sets. | Retain only as a dependency-scheduling analogy. It is not evidence that neural gradients commute or updates are local. |

## Reference alias reconciliation

Keep one canonical bibliography and use stable IDs across architecture, theory and ledger. The V2 aliases below are retained as a migration crosswalk; V3 uses canonical P-number labels in the current architecture and formal review.

| Architecture alias | Canonical ID |
|---|---|
| R01 | P-02 |
| R02 | P-03 |
| R03 | P-01 |
| R04 | P-04 |
| R05 | P-05 |
| R06 | P-10 |
| R07 | P-12 |
| R08 | P-14 |
| R09 | P-13 |
| R10 | P-11 |
| R11 | P-08 |
| R12 | P-07 |
| R13 | P-17 |
| R14 | P-20 and P-21 |

| V2 alias | Canonical ID |
|---|---|
| V2-P01 | P-01 and P-24 |
| V2-P02 | P-04 |
| V2-P03 | P-12 |
| V2-P04 | P-25 |
| V2-P05 | P-14 |
| V2-P06 | P-26 |
| V2-P07 | P-09 |
| V2-P08 | P-05 |
| V2-P09 | P-08 |
| V2-P10 | P-11 |
| V2-P11 | P-13 |

The `[P01]`–`[P11]` labels in file 05 map to the V2 alias list above, not the original `P-01`–`P-11` list. V3 normalized the current01/05 labels; this table documents their historical mapping.

## Copy-ready corrections

### Evidence-register opening

“Primary-source identity and the narrow claims summarized below were rechecked on 2026-11-21. The register records whether the retrieved evidence was an abstract, official implementation documentation, model card, author repository, selected methods text, or primary search/PDF excerpt. No literature result was independently reproduced by this audit. Most papers were not reviewed at full methods-and-appendix depth; no blanket code or data licensing conclusion is made.”

### CopyGen comparison

“CopyGen is an augmented decoding comparator with a frozen backbone and a learned plugin. Its official repository includes plugin training and checkpoint generation. Its training, calibration, additional parameters, storage and inference costs must be included in comparisons; freezing backbone weights does not make the complete system training-free.” [P-08]

### LOTUS comparison

“LOTUS reports a recurrent-depth latent reasoning system trained with gold intermediate CoT-token supervision. This motivates a controlled comparison of recurrence and supervision, not a claim that weight tying or unsupervised latent loops alone produce the reported improvement.” [P-05]

### Projection prior art

“The local parameter-Jacobian projection construction is closely related to Orthogonal Gradient Descent. OGD's methods discuss local changes and stored output gradients as proxies for gradients at later parameters; this is not a general finite-step preservation theorem. Anthes et al. provide complementary activation-subspace diagnostics.” [P-27, P-25]

### Adaptive computation prior art

“ACT, Universal Transformers and PonderNet establish prior art for learned computation depth and halting. The Nano hypothesis concerns a particular causal implementation and measured quality-cost trade-off; the existence of a halting rule is not novel, and no Bellman-optimal learned controller has yet been demonstrated.” [P-28–P-30]

### Qwen size accounting

“The Qwen3.5-4B-Base card lists a 4B language model within a multimodal checkpoint and a tied embedding/output vocabulary of 248,320 entries. Full checkpoint size must be determined from an exact revision and tensor inventory; the model name is not evidence of a strict four-billion-total-parameter limit.” [P-24]

## Specific depth limits and retrieval failures

- **CopyGen:** direct arXiv page opening returned a disabled-fetch error. Its primary arXiv search result supplied identity and abstract, and the author repository independently confirmed the project and trained-plugin interface. Treat this as verified identity plus abstract/repository inspection, not full-paper reading.
- **Dohare et al.:** the publisher HTML route redirected to an inaccessible identity endpoint. Publisher search and the publisher PDF excerpt recovered bibliographic details and the core plasticity/forgetting distinction. The original publisher URL remains valid as the reference.
- **PagedAttention:** the DOI endpoint returned HTTP403. The authors' arXiv version was available and explicitly labels SOSP2023.
- **Ye and Ding:** conference pages were retrieved; linked raw PDF endpoints returned unsupported content types. The source summaries therefore remain abstract-level. The exact assumptions are a documented pending methods-review task.
- **OGD:** the proceedings PDF was retrieved and the relevant local-gradient/proxy-gradient methods sections inspected. This does not imply a full independent proof audit, source-code reproduction or historical benchmark rerun.
- **Qwen and author repositories:** retrieved pages are mutable and were not pinned to immutable content hashes. Use exact release/config/commit IDs when an experiment depends on them.
- **Licenses:** CopyGen's repository declares MIT and Qwen's card displays Apache2.0. This audit did not inspect all underlying datasets, dependent code, weights or separate license files. Do not turn those page labels into a blanket permission conclusion.

## What this audit does and does not close

| Work item | Status after this audit | Evidence needed next |
|---|---|---|
| Bibliographic identity of existing P-01–P-24 | Closed at primary identity level | Recheck only if source changes or exact version matters |
| Identity of extra V2 Anthes and Zhang sources | Closed at primary identity level | Full methods needed for exact algorithm adoption |
| Narrow author-reported claims in existing register | Supported with the scope restrictions above | Independent reproduction for performance reliance |
| Missing projection/halting/recurrent-depth prior art | Four direct sources supplied | Add to canonical bibliography and matched baseline choices |
| Complete assumptions for hybrid hierarchy and replay theory | Open | Inspect complete paper/appendix; bind each formal claim to its assumptions |
| Exact license and source/checkpoint provenance | Open | Immutable version, license files, dataset permissions, tensor inventory |
| Native Nano LM quality or continual-learning superiority | Open; not tested here | Preregistered native-model experiments with paired seeds and full cost |
| Accuracy of Nano repository numerical results | Outside this subtask | Root's pinned repository/log audit, with raw run manifests |

The JSON companion contains one record per primary reference, source URL, inspection depth, retrieval reference identifiers, narrow supported claim and limitation. It is a machine-readable audit index, not a copy or snapshot of each source.



## Sequential editing: locality, acquisition and durability

- **P-32.** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli (2024). [Model Editing at Scale leads to Gradual and Catastrophic Forgetting](https://aclanthology.org/2024.findings-acl.902/). Inspection: conference abstract. Sequential ROME/MEMIT evaluation measures editability, earlier-edit forgetting and downstream capability degradation. A successful edit or immediate locality check is not a durable-update result. The tested editors/models do not imply every future editor must fail.

- **P-33.** Xin Ma, Wei Chen, Qi Liu, Derong Xu, Zhi Zheng, Tong Xu, Enhong Chen (2026). [More Edits, More Stable: Understanding the Lifelong Normalization in Sequential Model Editing](https://proceedings.mlr.press/v306/ma26f.html). Inspection: conference abstract. ICML2026 paper derives StableEdit from running-statistic normalization and ridge-regression analysis and reports long-horizon editing results. Full theorem assumptions, source code and compute overhead were not independently audited. Lifelong normalization here is a method-specific update mechanism, not generic LayerNorm or universal lifelong-learning proof.

- **P-34.** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262v5). Inspection: preprint abstract. NeurIPS2022 ROME paper describes rank-one FFN edits and evaluates specificity/generalization for factual associations. A localized edit control is useful for update locality. Single-edit efficacy does not prove resistance to later edits or continued pretraining.

- **P-35.** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau (2023). [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229v2). Inspection: preprint abstract. MEMIT reports editing thousands of associations in GPT-J6B and GPT-NeoX20B. Large-batch editing is not indefinite continual acquisition. Exact training statistics, model sizes and edit budgets matter; the retrieved v2 is dated2023 after the2022 initial preprint.

- **P-36.** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, Tat-Seng Chua (2025). [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](https://arxiv.org/html/2410.02355v4). Inspection: abstract and methods excerpt. ICLR2025 paper provides direct nullspace-editing prior art; its ideal algebra preserves a linear layer's outputs on fixed protected keys. Practical projection retains covariance eigenvalues below1e-2, not an exact nullspace in general. Estimated keys, later upstream drift and sequential layer edits prevent upgrading the fixed-key identity to a universal whole-LM guarantee.

For the Nano correction experiment, separate three outcomes: whether a correction is acquired immediately, whether unrelated same-entity and different-entity facts remain correct immediately, and whether the correction survives subsequent unrelated updates. A scoped replay success on the second outcome cannot close the third. The relevant native-model comparison should include a matched fine-tuning/replay control and an implementable localized or projected edit control, with their auxiliary statistics and preprocessing charged. ROME, MEMIT and AlphaEdit supply direct precedents; StableEdit supplies a contemporary durability comparator whose full assumptions and costs still need inspection.

**Fixed-key versus changing-model proof boundary.** The equality `(W + ΔP)K0 = WK0` follows exactly from `PK0 = 0` for the same protected key matrix. If keys subsequently become `K0 + E`, the layer's new-output difference includes `(W + ΔP)E`; fixed-key invariance alone cannot bound it. If a practical projector keeps nonzero covariance eigenvalues, `PK0` is generally nonzero and its residual must be measured. This boundary is an elementary inference from the model equations, not an additional theorem claimed for the complete Nano architecture.

**Reference status.** P-32–P-36 extend the canonical register to 36 sources. P-32/P-33 and ROME/MEMIT are abstract-level checks; AlphaEdit's relevant method sections and numerical projection threshold were inspected. None was executed or reproduced here.


Machine-readable companion: [literature audit index](results/literature_audit_20261121.json). The retrieval reference identifiers are session provenance, while the URLs and version suffixes identify the public sources. The audit index does not contain source-page snapshots.



## V4 component-method references P-37–P-40

These four additions document the exact training components now implemented. They add attribution and review boundaries; they do not validate model quality. The current register has 40 identities; the original 36-source audit remains a historical artifact.

| ID | Primary reference | Inspection depth and supported scope | Nano limitation |
|---|---|---|---|
| P-37 | Ilya Loshchilov and Frank Hutter (2019), [Decoupled Weight Decay Regularization](https://arxiv.org/html/1711.05101v3) | primary abstract and method. Decouples weight decay from loss-gradient moments in Adam-like adaptive optimization. | Implementation precedent, not proof of a good Nano training recipe or universal retention; initial preprint2017, v3Jan2019. |
| P-38 | Biao Zhang and Rico Sennrich (2019), [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467v1) | primary abstract. Introduces RMS-based normalization without LayerNorm recentering. | Nano epsilon, gain, dtype and derivative conventions are specified and checked locally; paper speed claims are not transplanted. |
| P-39 | Jianlin Su et al. (2021), [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5) | primary abstract. Uses position-dependent rotations to introduce relative-position dependence in attention. | Initial2021 preprint; retrieved v5Nov2023. It does not validate this implementation or context-length generalization. |
| P-40 | Noam Shazeer (2020), [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1) | primary abstract. Studies gated feed-forward variants in Transformers; direct precedent for the SwiGLU component family. | Author-reported benchmark gains do not show a Nano architecture advantage; local gradients are checked independently. |

Current machine index: [literature_audit_v4.json](results/literature_audit_v4.json). The earlier audit remains unchanged.
