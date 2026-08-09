# Result — route (b) span port on Qwen2.5-1.5B is gray, not decided

**2026-08-09.** Reports `papers/PREREG_SPAN_PORT_B.md` against its frozen bars.
SCREENING. Artifact: `artifacts/nano_h6/span_port/SUMMARY.json`.

**Verdict: interpretable gray zone.** `no_match_rate = 0.217` sits between the
accept bar (≤ 0.10) and the falsify bar (> 0.25). Route (b) is neither accepted
nor falsified. Ambiguity is not the failure mode.

## 1. Against the frozen bars

| quantity | value | note |
|---|---|---|
| control worst-class recall | **0.733** | ≥ 0.20 → interpretable |
| emitted spans | 493 | denominator for relocation rates |
| `relocated_rate` | **0.783** | unique patient-turn match |
| `no_match_rate` | **0.217** | absent from Patient turns |
| `ambiguous_rate` | **0.000** | never multi-match |
| `missing_span_rate` | 0.033 | gold had a span; model emitted bare label |
| state held-out mean | **0.956** | joint target did not destroy state skill |
| decision | **gray_zone** | prereg §Bars |

Accept required ≤ 0.10. Falsify required > 0.25. **0.217 is gray.**

## 2. What this establishes

- A balanced, prompt-matched span-bearing LoRA on Qwen2.5-1.5B **does emit
  spans** (missing-span only 3.3%; unparsed fields 0).
- Most emitted spans **do relocate** through the unmodified
  `_locate_unique_patient_span` (78.3%).
- Failures are almost entirely **paraphrase / non-exact copy** (`no_match`),
  not ambiguity.
- State transfer survives the joint target (95.6% held-out denials) — so the
  run answers the span question rather than collapsing into a label-only model.

## 3. What this does not establish

- Not route-(b) acceptance. 21.7% of emitted spans cannot be grounded and would
  abstain under the binder — too high for the frozen accept bar.
- Not route-(b) falsification. Below 25%, and the prereg says the cheap next
  step is quote-forcing / constrained decoding, not an immediate pointer-head
  rewrite.
- Not a ranking vs SmolLM2-1.7B (not run).
- Not multi-seed confirmation; not CUAD / real documents.

## 4. Consequence (per prereg Asks default #3)

Stay on route (b). Next cheapest discriminating fix before architecture change:

1. **Quote-copy / constrained decode** — force the model to copy a contiguous
   patient span (grammar or constrained decoding), remeasure `no_match_rate`
   against the same bars without moving them.
2. Only if that stays gray or worsens: reconsider route (a).

## 5. Method note

The run used `data_spans_balanced` with `prompt_format=state_and_span_line`.
That repairs two prior defects at once: the minority-class collapse that
confounded the first transfer curve, and the v2 prompt/target mismatch
("exactly one word" vs `STATED: "..."`). Bars were not moved after seeing
results.
