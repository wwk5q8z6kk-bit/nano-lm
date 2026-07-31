# Claim glossary — forbidden and approved language

*Phase 0B deliverable. Use in reviews as default-reject rules.
Pairs with `papers/EVIDENCE_LEDGER.md`. 2026-07-31.*

## Forbidden claims (downgrade or delete)

| Forbidden | Why | Replace with |
|-----------|-----|--------------|
| “Transformers cannot do deterministic extraction” | Classical M1 extracts deterministically on this task; LM failure ≠ impossibility | “Under this distribution and metric, small LMs show a held-out copying gap” |
| “Zero hallucination” / “hallucination-free system” | Open-world overclaim; oracle \(R\) ≠ reality | “Zero accepted violations of verifier relation \(R\)” (only if soundness proven for that \(R\)) |
| “LoRA preserves geometry / copy circuits” | E2 unidentified; ban without causal ID | “LoRA changed measured held-out gap (behavioral); mechanism unknown” |
| “Copy circuit induced” / “we induced induction heads” | No circuit identification in α | “Behavior consistent with a shift from memorize/classify toward copy under higher slot diversity” |
| “Clinical deployment ready” / “production scribe” | Synthetic data; no clinical validation | “Synthetic structured-summarization benchmark” |
| “NanoScribe / fabric is the right architecture” | E1 KILL on this task | Omit; or “gated; substrate thesis falsified for this task under frozen \(U\)” |
| “Scale removes the failure” (unqualified) | Stack confound; within-stack flat under full FT | “Gap is much smaller under the tested Pythia *pipeline*; cause not identified as parameter count alone” |
| “Verification solves faithfulness” | Scoped demo only | “Under decidable \(R\) on this instrument, propose→verify→abstain achieved …” |
| “ρ = hallucination rate” (E1 \(U\)) | Contradicts PREREG + scorer | “ρ = review load (fraction of fields flagged)” |
| “M1 is a train-only lexicon baseline” | M1 uses rules-perfect `fabric._extract` templates for this synthetic world | “M1 is a hand-template / rules extractor (oracle-grade for this generator); M2 is train-lexicon + span” |
| “E2 is running” / fabricated E2 U | No `results_e2_*` | “E2 GATED/STOP; no RESULT” |
| “E3 human / clinician evaluation complete” | Agent-rubric only | “Agent-applied rubric audit; dual-clinician IAA open” |
| “R★ / E4 shows generative value” | No E4 measurement | “R★/E4 protocol only; E4 v1 executed KILL; further execute BLOCKED; no generative-value claim” |

## Approved hedges (prefer these)

| Pattern | Use when |
|---------|----------|
| “under this distribution” / “on this instrument” | Any quantitative gap or utility |
| “for this verifier relation \(R\)” | Any precision / abstention claim |
| “behavior consistent with” | Mechanism-flavored discussion without ID |
| “requires further causal identification” | LoRA, morphology, circuits |
| “exact-match construct survives agent-rubric audit; clinician IAA still open” | E3 limitation language |
| “KILL under frozen \(U\); a different utility could re-rank” | Substrate discussion |
| “allergy is an instance of low-diversity residual, not the definition” | Residual discussion |

## Review default-reject

Reject drafts that: assert forbidden rows; merge Paper α measurement with fabric product pitch; fund E2/residual/scaling without citing which **ledger row** the result could change.
