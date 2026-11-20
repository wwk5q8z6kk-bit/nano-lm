# ONTOGENESIS research ledger

The one authoritative status list. Updated 2026-11-20. Status values: Established (published elsewhere), Derived (mathematics checked here), Demonstrated (measured here under a committed protocol), Pilot (measured, not confirmatory), Unverified, Disproven, Abandoned, Open.

## Measured in this repository

| Item | Status | Evidence |
|---|---|---|
| Exact copying helps on code: M21 1.79 vs dense 2.15 bits/token on copyable targets | Demonstrated | `research/frontier/m21_dense_code/TIME_MATCHED_RESULTS.md` |
| Dense beats M21 at equal training time: 4.65 vs 5.02 bits/token | Demonstrated | same |
| M21 beats dense at equal tokens only through copying | Demonstrated | `research/frontier/m21_dense_code/RESULTS.md` |
| S2 hybrid edges dense at 350M: 0.86339 vs 0.86774 bits/byte, one seed, 0.48x speed | Pilot (one seed, below its own 1% rule) | handoff `antigravity-continuation/` |
| Pythia-410M beats all native models on public text and grammar | Demonstrated | `experiments/native_public_language`, `experiments/grammar_contrast` |
| ACE router original gain came from reading future tokens | Disproven (original claim) | `research/ace_x9/` |
| Wide rotation registers (U10) | Disproven (null) | `research/frontier/u10_registers/` |
| DEMOS energy-minus-price-times-mass is non-increasing | Derived, checked numerically | `check_lyapunov.py` |
| Sort gives latest-write lookup for all positions; (min,max) closure gives reachability at every prefix; symmetric closure equals union-find | Derived, checked numerically | `theory/prefix_structure_checks.py` |
| DEMOS learns one-hop lookup, 32 static facts: 99.7% (dense 100%) | Pilot | handoff `ontogenesis-pilots/` |
| Dense fails one-hop lookup with updates after 7 min: 20.4% (guess rate) | Pilot | same |

## Ideas

| Idea | Status | Note |
|---|---|---|
| Interaction mesh, space-time mesh, UII candidates | Disproven at toy scale | dense won |
| ERN surprise-driven recurrence | Disproven at 125M | 2/16 |
| Holonomy rotation registers | Abandoned | solves group composition; <1% on text at ~8x cost |
| Gödel max–min model | Pilot (synthetic only) | algebra reused by RELATOR |
| LEDGER, DYADIC, LEXICON | Abandoned | largely published; known failure modes |
| METRIPLEX | Abandoned | parallel form is linear attention |
| ALGEBRA-EQUILIBRIUM | Abandoned | cost and capacity inconsistent |
| DEMOS | Open (pilot running) | no expressivity-per-cost gain over looped sparse message passing; sample efficiency untested |
| KEYSTORE | Open | partly published; learned keys untested |
| RELATOR | Open, blocked | smooth training leaks future; needs hard causal training |
| FILTRA | Open, not built | not found inside a language model |
| Prefix-scannable models | Established | ICLR 2026 |
| Matrix prefix products for state tracking | Established | 2024–2026 line |
| Datalog + union-find unification | Established | egglog |
| Time-threshold (filtration) for causal relational memory | Derived | one-line corollary of provenance semirings; use in LM training not found |
| STRATA | Unverified | every component published; learning of structure is the open row |
| PATHWAY (unified design) | Abandoned as written | critic: algebra misnamed, linear star weaker than a looped layer, cost claim unsupported, components published; see section 11 of `theory/UNIFIED-ARCHITECTURE-20261120.md` |

## Open scientific questions, in order of information per cost

1. How large is the long-context gain on our text and code, and on which tokens? (L0)
2. Can discrete keys be learned from a prediction loss? (KEYSTORE learned keys)
3. Does iteration over a persistent graph beat a looped Transformer at equal compute?
4. Does any structured memory keep its advantage once the dense model grows?

## Duplicates and conflicts resolved

- Names Gen3/4/5, S2, M0/M2, Omega each mean several things; cite path and commit, never the bare name.
- Five competing 2B–3.9B "flagship" plans in the archive: none funded; all superseded by this ledger.
- CLOSURE is merged into RELATOR.

## Research tree (2026-11-20)

Root: one unified language model with better capability per parameter, token and unit of compute.

| Branch | Question | State | Next action | Where |
|---|---|---|---|---|
| B1 Learning speed on changing facts | Why did DEMOS learn updates when dense did not? | Running | tuned dense, looped, message passing, KEYSTORE; then ablations | `train.py`, handoff `ontogenesis-pilots/` |
| B2 Knowledge in the weights | Does two-way binding remove one-way fact storage? | Not started | critic review, then REVERSE | `MISSION.md` lead C |
| B3 Reusable operations | Does an operation store beat a meta-trained Transformer? | Not started | critic review, then REUSE | `MISSION.md` lead A |
| B4 Size of the long-context gain | How much, on which tokens, prose vs code? | Blocked | obtain a corpus of long documents | `theory/ONTOGENESIS-PHASE2-20261120.md` L0 |
| B5 Exact keyed memory in context | Can discrete keys be learned from a prediction loss? | Arm built | pilot queued | `structured.py` |
| B6 Relational closure | Hard causal training for RELATOR | Blocked | design training without the smooth semiring | round 2, section 9 |
| B7 Identity merging | FILTRA | Not built | after B5 | round 2 |
| B8 Copy and recurrent baselines | Port S2 and M21 to the harness | Not started | after B1 baselines | master `research/frontier/m21_dense_code`, `experiments/native_public_language/archived/gen3.py` |
| B9 Compute | TPU price and availability | Partly done | verify hourly price on the billing page before any run | handoff `tpu-probe/RESULT.md` |

A branch is closed when its decisive experiment is reported, pass or fail. New branches are added here before work starts.

## Archive, with where each item lives

Source: two read-only surveys on 2026-11-20 (repository and Antigravity records). Sub-readers opened most files; treat rows as Observed unless marked.

| Item | Mechanism | Largest scale | Result against dense | Location | Status |
|---|---|---|---|---|---|
| Gen2 hybrid (X4, X8) | recurrent delta blocks between attention blocks | 46M | 0.9575 vs 0.9645 bits/byte, two seeds | branch `nanolm/discovery-lab-v1`, `lab/arch/` (local only) | Pilot |
| S2 / Gen3 | Gen2 plus erase and differential attention | 350M | 0.86339 vs 0.86774, one seed | branch `nanolm/arch-evolution-v3`, `lab/arch/gen3.py`; archived copy on master | Pilot |
| S2 at 125M, second implementation | same name, different code | 125M | 0.91355, worse than dense | worktree `nanolm-v3` | Contradicts the row above |
| Gen5 / M2 | Gen3 plus physics-style attention | 350M | ahead historically, behind on public text | `gen5_foundations.py`; evals on master | Demonstrated (public text) |
| BOBO, Kalman landmarks | fast-weight mixer, surprise anchors | 125M | document claim only | `nanolm/arch-evolution-v3` | Unverified |
| ERN | surprise-driven constant-state recurrence | 125.8M | 2/16 in every arm | branch `nanolm/autonomous-discovery-v1` | Disproven |
| Holonomy / HSPA | rotation registers | 14M | 0.3751 vs 0.3784 at ~8x cost | PR #174 | Abandoned |
| "Omega-1.7B beats Qwen3-8B" | a Qwen LoRA, not a native model | — | 40 vs 39 of 56; 32 replies truncated | PRs #171, #174 | Disproven as stated |
| G3/G4 discovery backbone | Householder scan | 20.6M | 2.1112 vs 2.1188, within noise | PR #176 | Unverified |
| Tool-recovery 60/60 vs 17/60 | — | — | outcome was hard-coded | PR #176 | Disproven |
| Mechanism lab X03–X07 | recurrent cells | toy | mixed after correction | PR #175 | Pilot |
| M13–M22 zoo, M21 | rotation, slots, copy pointer | 5M | see measured table | branch `research/unified-discovery-loop-v1`; port on master | Demonstrated |
| Omega core, ACE-X9.2, IVE | router, memory tiers, verifier | 47M | ACE lost at scale; leak fixed in #191 | unpublished local commits on `codex/p1-h7-scribing-path` | Disproven (ACE) / Unverified (rest) |
| Interaction mesh E27, bitemporal E28 | signed node mesh | <65k | 0.258–0.516 vs 0.521; one tie | same local commits, `reports/E27`, `reports/E28` | Disproven at toy scale |
| UII candidates A–F | mesh, field, attractor, automaton | 9k | 0.573 vs 0.621 | same, `reports/UII` | Disproven at toy scale |
| V3 exact recall | copy pointer | toy | original contaminated; clean rerun seed-dependent | branch `research/v3-exact-recall`; #187 on master | Demonstrated (rerun) |
| V4 first principles | event-state operator | — | never run | untracked files in worktree `nanolm-v4-fp` | Open, at risk (uncommitted) |
| V5 Gödel model | max–min message passing | 19k | 0.845 vs 0.39 on 5-hop graphs | branch `research/v5-distributed-systems` | Pilot (synthetic) |
| FP1–FP4 | first-principles benchmark | 2.5M | FP4 zero-parameter arm 11/30; learned arm not run | branch `recovery/nanolm-cycle4-first-principles-20261119` | Open |
| 2B and 4B presets | configurations | 1.98B–3.93B | untrained | `nanolm/arch-evolution-v3`, #171 | Unverified |
| "1.54B NanoLM" | — | — | was unmodified Qwen2.5-1.5B | Antigravity audit note | Disproven |

Not yet audited: seven openresearch worktrees and the `orx/*` branches; 12 foundation documents on `origin/research/foundation-intelligence-program`; the holonomy publication draft. V4's work exists only as untracked files and should be committed by its owner before any cleanup.

## Pilot results added 2026-11-20 (evening)

One-step lookup with updates, 48 live facts, 7 minutes, one seed, guess rate 20.4% (`results/pilots/pilot2-*`, `pilot3-*`):

| Arm | Accuracy |
|---|---|
| KEYSTORE, given keys | 100% |
| Looped Transformer, lr 1e-3 with warm-up | 99.95% |
| Dense Transformer, lr 1e-3 with warm-up | 99.8% |
| DEMOS | 98.3% |
| Sparse message passing with forgetting | 98.0% |
| Dense, lr 3e-3 no warm-up; dense, lr 3e-4 | 20.4% |
| KEYSTORE, learned keys | 20.4% |

- DEMOS learning-speed advantage: **Disproven** (baseline was mis-tuned).
- Learned discrete keys by straight-through estimation: **Pilot, failed**.
- Mixed one- and two-step task: every arm run so far is at guess rate; under-budget, **Open**.
- DEMOS first-order hop step cannot remove the current key: **Derived**, overlap checked numerically (`MISSION.md` section 8).

PLASTICITY v0, 200 successive tasks of 256 facts, three seeds, predictions written before the run (`plasticity.py`, `results/pilots/plasticity-v0-summary.json`):

| Arm | Steps to 95%, first 50 tasks | Last 50 tasks | Retention of earlier tasks (chance 1.6%) |
|---|---|---|---|
| Plain MLP | 61–72 | never reached within 300 | 1.5% |
| MLP with layer normalisation and weight decay | 33–34 | 37–38 | 4% |
| Sparse key-value memory | 254–267 | never reached within 300 | 3.4% |

- P1 (a plain dense network loses its ability to learn): **Demonstrated at pilot scale**, all three seeds.
- P2 (a sparse key-value memory keeps its speed and retains more): **Disproven** in this implementation. It was slower from the start, lost plasticity, and retained nothing.
- A standard remedy (normalisation plus weight decay) keeps learning speed almost flat. It is the strongest simpler alternative for any plasticity claim.
- No arm retains earlier tasks. Plasticity and retention are separate problems here.
