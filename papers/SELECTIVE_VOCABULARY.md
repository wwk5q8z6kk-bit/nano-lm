# Selective-prediction vocabulary — one set of terms for three subsystems

**A0 deliverable, 2026-08-05.** Before any new metric lands, this fixes the
words. Three subsystems in this repo already measure abstention and each uses
different names; a fourth vocabulary would make the situation worse, so
`nano_ai/evaluation/selective.py` will use the canonical column below and
nothing else.

Standard terms are taken from the selective-prediction literature (El-Yaniv &
Wiener, JMLR 2010; Geifman & El-Yaniv, NeurIPS 2017), not invented here.

## The canonical quantities

For a system that may present a value or withhold it, over a set of *attempts*:

| canonical term | definition | why it matters |
|---|---|---|
| **attempts** | every field/item the system was asked to decide | the only denominator the system cannot shrink |
| **presented** | attempts on which it committed to a value | |
| **coverage** | `presented / attempts` | the axis that disappears when a gate divides by `presented` |
| **selective risk** | `wrong_presented / presented` | error *conditioned on committing* |
| **retained correct** | `presented − wrong_presented` | what the user actually gets |
| **abstention benefit** | withheld items that would have been wrong | the verifier earning its place |
| **abstention cost** | withheld items that would have been right | the number that must never be hidden |
| **correct abstention** | withheld where withholding was the right call | distinct from cost; needs gold |
| **over-abstention** | withheld where a supported answer existed | the realised failure in model + wedge |

Two invariants, which any implementation must satisfy:

```
attempts   = presented + withheld
withheld   = abstention_benefit + abstention_cost      (when gold is known)
```

Fabric satisfies the second exactly: `withheld = 2642 = caught_err + lost_correct = 2642 + 0`.

## The mapping

| canonical | `nano_ai/evaluation.py` | `fabric/slice.py` | `wedge_v1` |
|---|---|---|---|
| attempts | `total_fields` (`:584`), `raw_proposal_field_count` (`:662`) | `raw_pred` | `n` tasks (`eval/dogfood_utility.py`) |
| presented | `presented_count` | `presented` | `n − n_abstain` |
| coverage | **`coverage`** (`:584`, `:555`, `:662`) | *derivable, never named* | `1 − R` |
| selective risk | **`false_presented_rate`** (`:602`) | **`presented_error_rate`** | `E` — but `E = 1 − Q`, so it carries no independent information |
| selective accuracy | **`selective_accuracy`** (`:590`), `content_selective_accuracy` (`:586`) | `1 − presented_error_rate` | `Q` — conflated (see below) |
| abstention benefit | *derivable from* `state_confusion` (`:621`) | **`caught_err`** | — |
| abstention cost | *derivable from* `state_confusion` | **`lost_correct`** ← the only named instance in the repo | `OVER_ABSTENTION` count |
| correct abstention | *derivable from* `state_confusion` | `abstained + qualified − caught_err − lost_correct` | `CORRECT_ABSTENTION` count |
| abstention rate | **`intentional_abstention_rate`** (`:604`) | `(abstained + qualified) / raw_pred` | **`R`** |

## What each subsystem is missing (and what it uniquely has)

**`nano_ai/evaluation.py` — the richest, and the closest to canonical.** It
already names coverage, selective accuracy, false-presented rate, intentional
abstention, and ships a full 5×5 `state_confusion` that *determines*
over-abstention and correct-abstention. What it lacks is a **threshold
argument**: it reports one operating point. The curve is the gap.

**`fabric/slice.py` — the only subsystem that names abstention cost.**
`lost_correct` (`:235`) is the seed of the canonical term and should be adopted
verbatim. Its gaps: coverage is derivable but never surfaced, and its gate
(`:248`) divides by `presented`, which is exactly the denominator the system
controls. It also has **no score to threshold on** — abstention is categorical
(`:178-185`) — so fabric contributes *points*, not curves.

**`wedge_v1/eval/utility.py` — abstention is a term, but degenerately.**
`U = Q − 0.5E − 0.3R − …` with `E = 1 − Q`, so `U` collapses to
`(α+β)·Q − γ·R − const` and the α/β split is decorative. Worse, a correct
abstention is credited in `Q` (its row passes) *and* penalized in `R`, so the
metric cannot separate correct abstention from over-abstention — which is the
one distinction the product needs. Its labels (`OVER_ABSTENTION`,
`CORRECT_ABSTENTION`) are, however, the right *concepts* and map cleanly.

## Rules adopted

1. **Never report a selective risk without its coverage.** Any presented-error
   number is written as `risk @ coverage`, e.g. fabric v2 nano m1:
   `0.00% @ 81.5%`.
2. **Never denominate a gate in a quantity the gated system controls.** Pair
   every conditional-risk bar with a coverage floor denominated in *attempts* —
   the shape `scribe/gate_grounded.py:129-133` already uses
   (`flagged / total <= 0.25`), not the shape `fabric/slice.py:248` uses.
3. **`lost_correct` is the canonical name for abstention cost.** New code adopts
   it rather than minting a synonym.
4. **Coverage and risk are reported per epistemic state**, not just overall —
   the ternary structure (assert-value / assert-absence / abstain) is what H6's
   gates already encode, and an aggregate hides exactly the trade H6 made
   (uncertainty +11.2, absence −19.6, overall −0.2).
