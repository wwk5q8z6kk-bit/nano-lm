# Claim glossary

Use these boundaries alongside `papers/EVIDENCE_LEDGER.md` when describing results.

## Forbidden claims (downgrade or delete)

| Forbidden | Why | Replace with |
|-----------|-----|--------------|
| “Transformers cannot do deterministic extraction” | Classical M1 extracts deterministically on this task; LM failure ≠ impossibility | “Under this distribution and metric, small LMs show a held-out copying gap” |
| “Zero hallucination” / “hallucination-free system” | Open-world overclaim; oracle \(R\) ≠ reality | “Zero accepted violations of verifier relation \(R\)” (only if soundness proven for that \(R\)) |
| “LoRA preserves geometry / copy circuits” | E2 unidentified; ban without causal ID | “LoRA changed measured held-out gap (behavioral); mechanism unknown” |
| “Copy circuit induced” / “we induced induction heads” | No circuit identification in α | “Behavior consistent with a shift from memorize/classify toward copy under higher slot diversity” |
| “Clinical deployment ready” / “production scribe” | Synthetic data; no clinical validation | “Synthetic structured-summarization benchmark” |
| “Fabric validates the NanoScribe architecture” | Fabric passed only a closed synthetic verifier relation; E1 rejected the tested generative substrate under frozen utility | “Fabric is a scoped verification regression harness; Nano's implementation remains to be validated” |
| “E1 proves Nano should not be a scribe” | E1 compared solvers on one frozen closed scribe task; it did not test whether the scribe intelligence should be built | “E1 rejects the tested generative solver as the preferred method for that task under frozen \(U\)” |
| “Scale removes the failure” (unqualified) | Stack confound; within-stack flat under full FT | “Gap is much smaller under the tested Pythia *pipeline*; cause not identified as parameter count alone” |
| “Verification solves faithfulness” | Scoped demo only | “Under decidable \(R\) on this instrument, propose→verify→abstain achieved …” |
| “ρ = hallucination rate” (E1 \(U\)) | Contradicts PREREG + scorer | “ρ = review load (fraction of fields flagged)” |
| “M1 is a train-only lexicon baseline” | M1 uses rules-perfect `fabric._extract` templates for this synthetic world | “M1 is a hand-template / rules extractor (oracle-grade for this generator); M2 is train-lexicon + span” |
| “E2 is running” / fabricated E2 U | No `results_e2_*` | “E2 GATED/STOP; no RESULT” |
| “E3 human / clinician evaluation complete” | Agent-rubric only | “Agent-applied rubric audit; dual-clinician IAA open” |
| “R★ / E4 shows generative value” | The recorded E4 comparison returned KILL | “On R★ v1, the tested generative reference did not beat the best classical method under frozen utility” |

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

Keep Paper α measurements, Fabric’s scoped verifier result, and AI direction
separate. Mechanism language requires a causal result, not a descriptive pattern.
