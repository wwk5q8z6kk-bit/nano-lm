# NanoScribe — Master Plan (owner rewrite, 2026-07-20)

*Condensed-faithful capture of the owner's rewritten master plan; supersedes the
roadmap portions of NANOSCRIBE_VNEXT.md (whose architectural content it absorbs).
Crossing-note: the plan's ground-truth table was written concurrently with the C-1
dry-run falsification — "token coverage secondary (medium-strong)" is REVISED to
"lexical interference leading candidate; coverage refuted as driver (fraction and
weakest-link forms)" per PREREG_token_coverage.md AMENDMENT 1. Phase 2's residual
quantification slot is filled by C-1b (interference classes), not the coverage
continuum.*

## Vision (protected from drift)
NanoScribe is a **factorized, evidence-first, verification-gated cognitive system**
whose linguistic core is a small model; memory, routing, verification, tools, and claim
management are bounded, typed, auditable modules. nano-lm's empirical results are the
permanent foundation and regression suite. Primary goal: **indefinite functional
scalability** with the measured failure modes bounded and detectable.

## Ground truth (fixed reference points)
Field-localized failure (very strong) → verification/memory must be slot-aware.
Slot diversity causal, 66.7 pts (strong) → diversity pressure is a design variable.
Residual quantification: interference-leading (revised; see crossing-note).
Interaction: weak base × full FT (medium, N small) → adaptation method + base quality
are first-class controls. Shared residual floor ~15–18 clean (strong) → hard binding
residual. Per-type categorical competence (strong) → per-type state is first-class.
Allergy = strongest instance, not definition.

## The 12 anti-drift invariants (non-negotiable; violations rejected by default)
1 small model ≠ the intelligence · 2 no global context by default · 3 every claim
carries provenance+uncertainty · 4 contradictions are first-class · 5 memory typed and
versioned · 6 writes require authorization · 7 decisions traceable · 8 independent
module failure · 9 adding components must not raise context noise/communication ·
10 functional with a single small model · 11 measured failure modes = permanent
regression tests · 12 "solves/mitigates the failure" claims must be pre-registered and
measured on the existing instruments.

## Factorized state
S = I × K × R × M × P × V × E (intent, provenanced knowledge, ephemeral reasoning,
typed memory incl. per-slot diversity/coverage statistics, planning/routing under the
control kernel, verification able to detect the measured failures, sandboxed logged
execution). Core hierarchy: model proposes → evidence grounds (with slot metadata) →
verifier tests (against binding failures) → risk controller decides (calibrated
abstention) → human reviews residual → memory stores only validated state.

## Phases
- **Phase 0 — freeze empirical core: DONE 2026-07-20** (corner seeds landed |Δ|=0.00;
  factorial frozen; binding account named; papers carry it; regression suite = the
  existing instruments).
- **Phase 1 — minimal vertical slice (NEXT BUILD):** smallest end-to-end path on the
  existing scribe task — Intent → Control → nano-lm → Evidence Ledger (per-type
  coverage/diversity metadata) → Verifier (detects known copying failures) → Risk
  decision → typed Memory write. Success gate: equal-or-better clean-metric failure
  rates + full provenance per claim.
- **Phase 2 — residual first-class:** ledger tracks slot diversity + coverage; run
  C-1b (interference); verifier flags binding failures, not just surface mismatches.
- **Phase 3 — controlled expansion:** one bounded module at a time; each declares
  read/write contracts, passes the regression suite, shows no contamination increase.
- **Phase 4 — functional scalability dashboard:** clean gap (primary), unverifiable-
  claim rate, provenance completeness, contamination rate, failure recovery.

## Decision points
Corner ≤4.5 diluted → tokenizer/architecture innocent (FIRED). Module improves surface
metrics but worsens clean gap → reject. Router becomes a source of unverifiable claims
→ constrain toward deterministic.

## Two-track rule
Empirical track = ground truth, never diluted by architectural ambition; architectural
track must continuously pass the empirical tests.

## Immediate next actions (owner-ordered)
1. ✅ corner frozen (Phase A/0 closed).
2. → EMPIRICAL_FOUNDATION.md (the invariants lockfile) — companion doc, this commit.
3. → Phase 1 minimal vertical slice (next build unit).
4. C-1b + further modules only after the slice is measured and stable.
