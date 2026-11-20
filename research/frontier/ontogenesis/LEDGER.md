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
