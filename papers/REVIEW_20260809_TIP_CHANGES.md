# Review — tip changes through the balanced transfer curve

**2026-08-09.** Review of the work landed on `codex/p5-measurement-integrity`
up through `16f89cc` (balanced transfer curve). Measurement-integrity theme:
several results were confounded by reporter-controlled denominators or label
mix; the surviving claims are narrower and better gated.

## What landed (newest first)

| Commit theme | Verdict | Survives? |
|---|---|---|
| Balanced transfer curve | Interpretable 4/5; both 1.5B and 1.7B clear frozen 75% bar; monotone among interpretable points | **Yes** — bracketed answer: sufficient base ∈ (0.5B, 1.5B] |
| `--balance` on LoRA builder | Confound repair, not a new experiment; bar/guard unchanged | **Yes** as method |
| Confounded first curve | Correctly retained; 4/5 uninterpretable; minority-class collapse | **Yes** as negative result |
| CUAD as H-6 real corpus | Licence + natural abstention rate measured on downloaded artifact | **Yes** as corpus decision; adapter not built |
| Span-bearing v2 dataset | Route (b) data fix; separate dir; `target_format` in provenance | **Partial** — built, but prompt still says "one word" while targets carry spans; **not balanced** |
| Span port → route (b) | Evidence-backed design decision with falsifier rates | **Yes** as decision; falsifier **unrun** |
| Wedge over-abstention audit | Claim was stale/overstated | **Yes** as correction |
| Prior-art check | Cycle findings mostly established elsewhere | Hygiene, not a gate |
| LoRA control (3B) | Pretraining carries transfer; state-only | **Yes**, with stated limits |
| Cross-model / surface / DP-1 cycle | Lexical unfamiliarity; seed instability; degeneracy guards | **Yes** as diagnostic chain |

## Measurement-integrity defects caught this cycle

1. Always-`DENIED` scorer (cross-model probe v1)
2. Guard fooled by skewed gold share (probe v2)
3. Fabric / DP-1 denominator degeneracy pattern
4. Transfer-curve minority-class collapse (5.6% `missing`)
5. Confounded "threshold near 1B" reading that would have included a 99.7% model that never emits `NOT_MENTIONED`
6. **Open:** span v2 prompt/target mismatch ("exactly one word" vs `STATED: "..."`)
7. **Open:** span v2 inherits natural label mix — same confound waiting for the next run

## What the evidence now authorises

- Stop from-scratch Nano as a product constraint; initialise from pretrained.
- Target base size **≥ ~1.5B** at 4-bit for local-first; 0.5B is insufficient on the balanced curve.
- Span contract port is **route (b): generate text, relocate by unique match**.
- Real-document dogfood corpus is **CUAD** (present/absent + spans + natural nulls).

## What it does not authorise

- Ranking Qwen2.5-1.5B over SmolLM2-1.7B (one seed; 31.5-point arm instability).
- Any claim that a pretrained base carries Nano's **span** contract.
- Clinical or open-world transfer.
- That balance alone (vs smaller N) caused the repair — size and balance co-vary.

## Next measurement (critical path)

Not another size sweep. Run the route-(b) falsifier on **Qwen2.5-1.5B** with a
**balanced, span-bearing** dataset and a prompt that matches the target format.
Report `no_match_rate` and `ambiguous_rate` separately, gated by the same
balanced control block. Prereg: `papers/PREREG_SPAN_PORT_B.md`.
