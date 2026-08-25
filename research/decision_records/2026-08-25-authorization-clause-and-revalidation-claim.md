# Decision record — the authorization clause, and what the causalfix wave may claim

**Date:** 2026-08-25
**Decider:** owner
**Status:** in force

Two decisions, both settled against the permissive reading, plus one gap
recorded rather than closed.

---

## D1 — `$0` does not settle authorization

**Question raised.** `docs/ACTIVE_NOW.md` carries both
`local_zero_cost_exploratory_training = ALLOWED` and
`confirmatory_evidential_run = PREREG_PLUS_EXPERIMENT_SCOPED_AUTHORIZATION`.
E-DELIMIT is simultaneously \$0-on-local-MPS and pre-registered-confirmatory, so
it satisfies the first and triggers the second. Which governs?

**Decided: the confirmatory clause governs.** The document forecloses the
permissive reading in its own text, at line 47:

> The distinction is **cost / risk / evidential significance**, not "free local
> versus paid cloud."

That sentence exists specifically to stop `$0` from being treated as
self-authorizing. A pre-registered contrast carrying a kill condition is
evidentially significant whatever it costs to run.

**Scope note.** The NOT-AUTHORIZED list at line 77 prohibits *"confirmatory
evidential runs without required prereg."* A pre-registered confirmatory run is
therefore not forbidden — it is missing only the experiment-scoped authorization
half, which the owner grants directly. E-DELIMIT's grant is recorded verbatim in
`research/preregistrations/PREREG_E_DELIMIT.md`.

---

## D2 — Keep the causalfix conclusions; drop the revalidation claim

**Established, and staying established:**

- **The capability floor fired.** Pooled constrained coverage 54/1350 = 4.0%
  against a 10% threshold; unconstrained coverage 0/150 in 9/9 runs; every model
  abstains on 144 of 150 atoms. 30M at 1800 steps on a character-level hash
  tokenizer is below the capability floor for `p1_screening_eval_v1`, and the arm
  comparison is not used to rank architectures.
- **The arm split is seed noise.** `evidence_bottleneck` 6/18 pooled, per-seed
  (6, 0, 0); Wilson intervals overlap (0.1628 vs 0.1759); higher in 1 of 3 seeds.

**Marked PENDING:**

- **"The native30 wave ran clean under the integrity gate."** This cannot be
  asserted from `artifacts/campaign/reval_results_causalfix/`. That is the
  pre-assertion wave, which — per the session note that found it — *"has the code
  fixes but not the runtime gate, so it cannot prove it ran clean and does not
  satisfy the revalidation claim."* The claim is pending the gated re-run to
  `reval30_*_fixed_*`.

**Why the conclusions survive but the claim does not.** The integrity gate
guards against leakage, and leakage inflates results. Both conclusions above are
negative — a floor reading and a null. A missing anti-leakage gate cannot
manufacture 4% coverage or a seed-noise verdict, so the numbers are robust in the
direction that matters. But *"the wave ran clean under the gate"* is a statement
**about the gate**, and it cannot be made from a wave that did not have one.
Dropping it costs nothing that was actually in hand.

---

## G1 — Recorded gap: the eight-cell leakage ablation was not separately authorized

The eight-cell leakage ablation (`ddb5ce6`) was pre-registered, carried
CONFIRMED/REFUTED decision rules, and produced the program's headline span-port
result. By D1's reading it was a **confirmatory evidential run**, so the
zero-cost clause did not cover it and it needed experiment-scoped authorization
it does not have on the record. Earlier advice that the zero-cost clause covered
it, and that a stop on this point was a misapplied TSC #2, was wrong on the plain
text of line 47.

**This is recorded, not closed.** Retroactive ratification remains available to
the owner and would resolve it in one sentence. It matters because `ddb5ce6` is
load-bearing: the honest baseline, the C2-inert finding, the C1 UNRESOLVED
verdict, and the C3 void all rest on it.

Nothing about the ablation's *numbers* is in question here. The gap is procedural.
