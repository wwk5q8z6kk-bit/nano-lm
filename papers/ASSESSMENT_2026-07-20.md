# Program assessment of record (owner, 2026-07-20) — binding-and-coverage account

*Owner's full dissection + critical assessment + forward plan, condensed-faithful.
Supersedes narrative framings in earlier drafts where they conflict.*

## The account (named)
The held-out value copying failure is a **binding-and-coverage problem** — how reliably
a novel value is bound to its target slot under low type frequency and incomplete token
coverage — not a pure capacity or optimization deficit. Primary controllable factor:
**slot diversity** (66.7-pt pre-registered causal effect, monotonic, position innocent).
Secondary: **token novelty/subword coverage** (real, under-quantified). Training regime
modulates severity (worst corner: under-trained base × full-parameter adaptation; either
escape recovers a similar fraction; raw params and token volume secondary once diversity
and coverage are accounted for). **Allergy is the strongest instance of the mechanism,
not its definition** — de-emphasize accordingly in P2's narrative.

## Critical ledger (owner)
Solid: field localization (exact zeros); diversity effect; structured (not generic)
failure. Soft: interaction precision (few runs; one seed check; keep bands prominent);
token-coverage unquantified; Pythia↔own confound narrowed but the deepest adversarial
question stands: *are diversity+coverage downstream of a more basic residual-stream
binding failure?* Generality: one task/distribution.

## Corner decision rule — metric clarification (maintainer note)
The PRE-REGISTERED Q1 rule is on the DILUTED gap (≤4.5 / ≥6), fixed before the run; it
FIRES at seed-0's 4.2 ± 0.9 → own↔Pythia difference attributable to data+method,
tokenizer/architecture ~innocent. The owner's restatement above referenced the CLEAN
gap; under that lens the corner (17.7 ± 3.2) matches Pythia (14.7 ± 2.1) — same relative
conclusion — while exposing that BOTH branches hold: the cross-stack difference resolves
AND a shared hard floor (clean ~15–18 = the alg slot, the binding+coverage residual)
persists in both stacks. Phase C-1 therefore proceeds regardless of stack.

## Forward plan (owner)
- **Phase A (now):** finish corner seed-1 → freeze the factorial table at supported
  precision → write P2's empirical core around the binding-and-coverage account.
- **Phase B:** decision point (resolved as above: both branches partially fire).
- **Phase C (post-corner, priority order):**
  1. **Token-coverage continuum** — controlled novelty gradient at fixed slot diversity
     → effect size for the secondary factor.
  2. **Cross-slot generalization** — the diversity manipulation on other open-vocab
     slots: is 66.7 allergy-specific?
  3. **Minimal binding probe** — smallest synthetic task isolating value→slot binding
     under low type frequency; if it reproduces, the account is deeper than the scribe
     surface form.
  4. Higher-quality within-stack deconfounder — only if still load-bearing.
- **Phase D (parallel, lower urgency):** NanoScribe vNext architectural track, strictly
  separated; the binding-and-coverage failure becomes a permanent regression test for
  any factorized system. Do not let architecture dilute measurement discipline.

## Stance
Finish and absorb the corner; freeze P2 around binding-and-coverage (diversity primary,
coverage secondary, regime interaction); allergy = strongest instance; architecture
resumes only after the empirical core locks.

## Owner clarification (post-corner-seed-0): the residual is the central object

Both branches are true at once: the own↔Pythia gap was **largely eliminable** (strong
base + LoRA reaches the reference level; tokenizer/architecture largely innocent), AND a
**shared hard floor** (~15–18 clean points, alg-dominated) survives strong pretraining,
parameter-efficient adaptation, and cross-stack transfer — the true binding-and-coverage
residual. Updated causal table (owner):

| Layer | Status | Notes |
|---|---|---|
| Field localization | Solid | open-vocab only |
| Slot diversity | Strong causal | 66.7-pt effect |
| Token coverage | Secondary but real | leading candidate for the residual |
| Base quality × adaptation | Interaction real | explains most of the own-stack penalty |
| Tokenizer / architecture | Largely innocent | cross-stack gap mostly closed |
| **Residual floor** | **Real and shared** | ~15–18 clean pts, alg-dominated — **the target** |

Directive: Phase C-1 (token-coverage continuum) is the next experiment, stack-independent.
