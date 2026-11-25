# Causal information-transport feasibility audit — 2026-11-25

**Status:** new *synthetic implementation-contract evidence only*. No Nano LM run, learned-model effect, scientific validation of either external paper, protocol replacement, architecture selection, or merge. This is an additive review built on `research/ontogenesis-phase1`; other workstreams retain authority over their experiments.

## Repository, source and authority

- Connected GitHub repository `wwk5q8z6kk-bit/nano-lm`. Verified `master` at `3e36bf803a477f885a47614ca7f9d02429f85de4` and `research/ontogenesis-phase1` at `f9c6debc5b32e5a04568efc4288a1ea729e5bc3d`: both compare as **identical to the preceding review cutoff**, no new commits.
- Canonical `research/frontier/ontogenesis/LEDGER.md` has Git blob `f47542073938667a6d07bc2fae40d11168905618` on research. Reviews #210, #211 and #212 remain separate **draft, unmerged** proposals. PR #195 is **draft/unmerged** at `f356fb489b0c4a821ca8209a8c530609b5b2d50a`; its validation receipts are instrument checks, **not** a native generative model run. No new local training evidence or independent human scientific review was identified.
- Direct Nano sources: `research/frontier/ontogenesis/PROTOCOL-NESTED-LR.md`, `results/nested-lr/verdict.json`, `LEDGER.md`; `research/frontier/gate1_and_depth_routing/PROTOCOL.md`, `results/tournament/summary.json`, `score_tournament.py` on draft PR #203 at `7d0e6ee0926f614c41ee5b6e6f5e797fb82cff15`; PR #195 `architecture/16_GENERATIVE_REASONING_EVALUATION.md` and its `NOT_RUN/NOT_CHECKED` controls. All paths begin at repo root; external draft source locations are **not** present on the research branch.
- Primary literature: Zhao et al., [LoopRes arXiv:2611.31094v2](https://arxiv.org/html/2611.31094v2) (submitted Nov 23, revised Nov 24), especially §§2.4, 3.2, A.6; Xiong et al., [Attention Latent Feedback arXiv:2611.31154v1](https://arxiv.org/html/2611.31154v1) (submitted Nov 23), §§3.2–3.3, 5, A.2–3. Already reviewed separately in proposed PR #212. These are **external author-reported outcomes**.

## Decompose the noninterchangeable information paths

| Path | Source → destination | Information/cost boundary | What is still unknown |
|---|---|---|---|
| First-layer value residual (prior art, ResFormer family) | Attention value at early depth of **same token/position** → deeper attention value | Seven learned scalar mixture coefficients in the local 14M PR #203 tournament; coefficients do not determine actual information or norm share | Benefits beyond that dataset, baseline family, seeds and short schedule; matched prior-art/generic-skip controls |
| LoopRes (external) | Earlier loop iteration's block output for the **same token** → receiving loop's block, plus prelude and earlier current-loop blocks | Max `(W+1)B+1` sources per receiving sublayer with W earlier loops/B blocks, per paper; structural keys distinguish iterates | Whether its loop gains transfer from SFT of pretrained 1.7B–30B models to the native 14M–386M line or overcome measured wall-clock costs |
| ALF (external) | Deeper sublayer output at **position t-1** → shallower/current sublayer at **position t**, plus separate previous-final-state input | Previous-token cache of `2L` sublayer outputs and top state; source index `i >= l`. A fixed nearby-depth window bounds selection | Quality, speed, memory traffic, long-context prefill and true energy on Nano; teacher-forced parallel passes approximate exact sequential recurrence |
| Existing Nano nested reuse | Repeated whole block in toy BINDINGS task | `PROTOCOL-NESTED-LR.md` registered mixed decision | Does not validate LoopRes, ALF, or any combined architecture |

**Do not collapse** same-position depth transport, same-position iteration transport, and prior-position temporal transport into a single mechanism: they have different causal boundaries and bottlenecks. A fused design is an unregistered hypothesis, not the conclusion of these sources.

## Verification performed: 8 local CPU synthetic contract tests

New standalone `research/frontier/ontogenesis/reviews/transport_contracts.py` contains a deterministic standard-library toy scalar reference with no external library/code, learned weights, GPU, checkpoint or repository execution. It is intentionally **not a faithful implementation** of the LoopRes or ALF algorithms. Test source Git blob: `a861f5a95ddb0a7d6e6843f8b6c6a1a5af44224b`; SHA256 `9f6066a97b86a8c7b6db516a9516433fd277e469be335666c571f2e4fe345dc3`. The published GitHub blob was read back and matches the byte-identical locally tested file. Command: `python transport_contracts.py`. An isolated temporary container—not the user's worktree—ran **8 tests, 8 passed**.

| Test | Outcome / scope | Alternative, limitation or failure mode it screens |
|---|---|---|
| Loop source window and no-future rules | PASS: sources include only preceding W loops, prelude and earlier blocks of current loop, with no duplicates | Tests indexing, not learned routing, loop effectiveness or tensor shapes. Partial within-block sums are **omitted** in this toy |
| Structural identity tie control | PASS: distinct loop identity scores distinguish identical source content; zero identity query yields tied weights | A design can encode identity without learning beneficial selections; not evidence of accuracy |
| Value-residual scalar algebra | PASS: endpoints and 0.09 coefficient produce expected linear mixtures | A 91% coefficient on v0 is not a measured 91% information, norm or attention-output share |
| ALF prior-depth window | PASS: allowed previous-token depth reads are at/above receiving depth, at most W sources | Does not implement low-rank maps, token-input injection, Q/K softmax, actual attention or source-value normalization |
| Prefix causality | PASS: mutating a future token does not affect earlier outputs in exact or two-pass toy | Not an audit of real kernels, dropout or model training |
| Document reset | PASS: changing first document leaves outputs of second unchanged after reset; removing reset is a negative control and yields leakage | Production batching, packed masks and prefill need separate tests |
| Jacobi pass-depth equivalence | PASS: for six-token scalar toy, pass K matches exact sequential states for first K tokens, with full equality by pass 6; two passes are **not** exactly equivalent in general | Expected behavior; is **not** a failure of ALF, which optimizes an approximate two-pass training objective |
| Bad arguments | PASS: invalid source/window/pass arguments rejected | Does not guarantee numerical safety in real GPU kernels |

On the fixed synthetic six-token sequence, max absolute hidden-state difference against exact sequential execution is **0.404698323541** (pass 1), **0.135996004720** (pass 2), **0.021145201590** (pass 3), **0.006737117362** (pass 4), **0.002122351027** (pass 5), and **0** (pass 6). These are arbitrary toy state units, **not BPB, benchmark quality, training loss, memory bandwidth or evidence of model convergence**.

## Costs and interpretations: do not overclaim

- LoopRes reports a 29.1% **relative** improvement over SFT on an average of math/code tasks at 2× inference compute for a pretrained Qwen3-1.7B backbone; this is not a 29-point absolute gain, not an equal-end-to-end-energy result, and not a NanoLM run. The paper's fused-versus-unfused timing depends on specialized kernels.
- ALF reports up to about 1.38× equivalent baseline *training tokens at fixed size* for one scale, but total training FLOPs increase by roughly `1+(1+mu)/4 ≈ 1.253` and the author estimates roughly 1.27× overall training time. The paper itself notes that increasing backbone size at fixed training FLOPs can be competitive. A 1.1% decoding FLOP estimate **does not** measure latency, cache bandwidth or energy. Its evaluations often use several Jacobi passes; two-pass training does **not** give exact full-prompt sequential equivalence.
- Existing Nano toy reuse remains **mixed**; PR #203's slow depth routing remains **closed at 14M** under its recorded D1 and unconditional coda remains rejected under D2. Different mechanisms and objective/scales mean this new CPU contract has no authority to reopen or override those verdicts.
- The test harness checks invariants of a deliberately simple scalar recurrence, not production B/T/D layouts, exact masks, cross-device determinism, activation correctness, numerical precision, gradient flow, kernels, real tokenizers or measured runtime. The costs above are **source-described or analytic source bounds**, not new measured hardware results.

## Decisions and next cheapest decisive action

1. **Accept a narrow engineering feasibility result:** the tested abstract indexing/reset/prefix rules can be implemented with a deterministic CPU reference. **Do not** promote any learned architecture, routing mechanism or performance claim.
2. **P0 provenance** remains mandatory: resolve data/trainer/scorer/checkpoint hashes and PR #203 scorer guard defects as recorded in #211; independently review PR #195 and prior negative findings before any research-line integration.
3. **Next cheapest untrained check:** a second independent inspection of the *published authors' implementation* (source snapshots and licenses) and existing Nano code to pin specific shape/mask/reset semantics and identify whether the ALF prefill transition and LoopRes prior/current-loop ordering could be violated. No dataset, model download or GPU required. Do not mistake this toy proof for a faithful code parity test.
4. **Next genuinely decisive experiment (requires a separately approved preregistration and authorized compute):** compare independent LoopRes-like and ALF-like mechanisms **separately** at smallest defensible scale using tuned dense, tied/unrolled controls, matched token/compute AND wall-clock, at least 3 independent seeds and held-out contextual reasoning. Include document-boundary adversaries, ALF sequential vs one/two/three-pass prefill, fixed versus learned depth windows, prior-art/generic-shortcut controls, measured throughput, peak memory, memory traffic and ideally joules/token. Set effect/stop rules before results. Add a combined arm **only after** both isolated gains survive.
5. **Governance:** leave historical `LEDGER.md` rows unedited, V6 draft/unmerged, prior review PRs separate. This review's canonical ledger link is a **proposed append-only annotation**, not an approved experiment update.

No private owner data, PHI, licensed training material, cloud session, paid compute, external checkpoint, raw-result overwrite, running-protocol change or architecture merge was used.
