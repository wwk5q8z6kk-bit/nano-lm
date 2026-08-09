# Decision — CUAD is the real-document corpus

**2026-08-06.** Resolves the corpus half of hurdle H-6. Licences verified against
primary sources; the data artifact was downloaded and parsed, not inferred.

## Decision: CUAD (Contract Understanding Atticus Dataset)

510 **real** commercial contracts × 41 clause types, with **character-offset
spans**. **CC BY 4.0**, confirmed on both the project page and the HF dataset-card
`license` field. No login, no DUA, data checked directly into the GitHub repo.

**Why it fits this project better than anything else surveyed:**

| Nano needs | CUAD provides |
|---|---|
| real documents, not synthetic | 510 real commercial contracts |
| a structured record over multiple fields | 41 clause types per contract |
| **spans with offsets**, since the metric is span-exact | `answer_start` character offsets, spot-checked as exact slices into `context` |
| **genuine unanswerable cases**, since the claim is calibrated abstention | **67.95%** of the 20,910 (contract, clause) pairs carry `is_impossible: true` with an empty `answers` array |

That abstention rate is **measured**, not quoted: the agent downloaded
`data.zip` (18,309,308 bytes) and parsed `CUADv1.json` — 20,910 pairs = 510 × 41
exactly.

**This is the property no synthetic corpus gave us.** A corpus with no genuinely
unanswerable items cannot test calibrated abstention at all; two thirds of CUAD
is unanswerable by construction of the real contracts, not by adversarial
authoring.

## What CUAD will not test

- **Ambiguous grounding.** Its negatives are clean absences, not hard-to-ground
  positives. Conflicting evidence — Nano's `conflicting` state — has no analogue.
- **One domain, one register.** Legal English, not clinical dialogue. A win here
  does not transfer to clinic notes.
- **No epistemic-state schema.** CUAD is present/absent. Nano's five states
  (`supported`/`absent`/`missing`/`uncertain`/`conflicting`) collapse to two.

So CUAD tests **evidence-bound extraction with abstention on real documents** —
the product's core claim — and not the epistemic-state taxonomy.

## Secondary picks

- **SQuAD 2.0** (CC BY-SA 4.0) — the standard abstention benchmark, ~34%
  adversarially-constructed unanswerable. Useful as a sanity check; not a
  structured record, and its unanswerability is adversarial rather than natural.
- **CORD** (CC BY 4.0) — real photographed receipts with bounding-box grounding;
  the only candidate testing *noisy scanned* documents. Its absent-field
  behaviour is unverified — check the schema before relying on it.

## Clinical bridge, if one is wanted

**ACI-Bench** and **PriMock57** (both CC BY 4.0, both role-played by clinicians
with no PHI, neither derived from credentialed sources) bridge from synthetic
clinic dialogue toward real speech. **MTS-Dialog** is CC BY 4.0 but may include
content derived from mtsamples.com — a separate copyright holder — so verify
before redistribution. **Asclepius** is CC-BY-NC-**SA** (non-commercial).

## Disqualified, and why — applied consistently

- **Licence unverifiable** → SciERC, FiQA, and notably **Natural Questions**,
  which has the best structural fit of anything surveyed (real full Wikipedia
  pages, naturally-occurring nulls) but whose data licence could not be pinned
  after four independent fetches. Disqualified under the same rule as every
  other ambiguous item rather than waved through on reputation.
- **Non-permissive but clear** → FUNSD (non-commercial, 18+ gate),
  Asclepius (CC-BY-NC-SA).
- **Access-gated** → DocVQA (portal account required before download).
- **Credentialed** → MIMIC, i2b2/n2c2 — never considered, per the standing rule.

## Next action

Build a CUAD adapter that maps `(contract, clause_type) → StateSpanProposal`
with `supported`/`missing`, and score it with the **unchanged** `_proposal_exact`
metric. That reuses the existing evaluation authority and tests route (b)'s
relocation on documents nobody in this project authored.
