# Choosing private or shared writes without an edit label: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-ROUTING.md](PROTOCOL-MEMRULE-ROUTING.md), committed in `6da758f` before any held-out run. Seeds 4101–4105, last 10 of 40 phases. Raw files: `results/memory_rule_routing/`. The labelled ceiling is the private 2,000-step arm of `results/memory_rule_privatefix/` (same seeds, same world). Conditions agreed with nano-lm-ad, which owns `INVESTIGATION.md`, Question 8. [Observed]

## Means over five held-out seeds

| Arm | Corrections | Inference, new entities | Acquisition | Retention | Fraction routed private | Regular new facts routed (false positives) | Exceptions routed | Corrections routed |
|---|---|---|---|---|---|---|---|---|
| No routing | 0.037 | 0.736 | 0.892 | 0.822 | — | — | — | — |
| Detector, τ = 0.1 | 0.810 | 0.701 | 0.968 | 0.880 | 0.205 | 0.089 | 0.703 | 0.907 |
| Random, matched to τ = 0.1 | 0.194 | 0.703 | 0.911 | 0.834 | 0.190 | 0.189 | 0.216 | 0.185 |
| Detector, τ = 0.2 | 0.499 | 0.728 | 0.920 | 0.836 | 0.078 | 0.010 | 0.308 | 0.559 |
| Random, matched to τ = 0.2 | 0.104 | 0.730 | 0.901 | 0.825 | 0.074 | 0.073 | 0.091 | 0.074 |
| Labelled ceiling (corrections only, given) | 0.891 | 0.714 | 0.887 | 0.881 | — | — | — | — |

For no routing, the "routed" columns would be what the detector *would* flag at τ = 0; nothing is routed.

## Against the registered predictions

- R1 (τ = 0.2: corrections ≥ 0.40, inference ≥ none − 0.03): **held**, five of five. Corrections 0.469–0.519; inference 0.003–0.011 below none.
- R2 (τ = 0.1: corrections ≥ 0.75, inference ≥ none − 0.05): **held**, five of five. Corrections 0.769–0.838; inference 0.029–0.038 below none.
- R3 (each detector beats its matched random router by ≥ 0.20 on corrections, **with inference no lower**): **did not hold** at either τ (two of five at 0.1, one of five at 0.2). The corrections clause holds on every seed by a wide margin: 0.59–0.64 at τ = 0.1, 0.38–0.43 at τ = 0.2. The inference clause fails: detector and random differ by −0.008 to +0.005, so the detector's inference is equal to random's, not higher.
- R4 (false positives ≤ 0.01 at τ = 0.2, ≤ 0.10 at τ = 0.1): **held at τ = 0.1** (four of five; 0.068–0.117) and **did not hold at τ = 0.2** (three of five; 0.006–0.016, mean 0.010).

## Reading

A label-free detector gets most of the labelled result. At τ = 0.1 it applies 0.81 of corrections against the labelled 0.89, with inference 0.70 against 0.71, acquisition 0.97 against 0.89 and retention 0.88 against 0.88. It does this without being told which facts are corrections, and the detected arm beats a router that sends the same number of facts private at random by about 0.6 on corrections.

R3's failure is informative. Its inference clause asked the detector to cost *less* inference than random routing. It costs the same. In this memory the later inference cost depends on how many facts are written privately with a large budget, not on which ones. What the detector buys is choosing the right ones: the corrections, and many exceptions.

The two gates trade as the development seed suggested. τ = 0.1 gets more corrections (0.81) for 0.035 inference; τ = 0.2 gets half the corrections (0.50) for 0.008. Neither dominates.

The detector also sends exceptions private (0.70 at τ = 0.1). In this world that helps acquisition: irregular facts stop competing with the rule in shared slots, which is why acquisition (0.97) exceeds the labelled ceiling (0.89, which routed corrections only).

## Development-seed finding: a disagreement detector cannot start from an empty memory

Development seed 4100 only, not confirmed on held-out seeds. With no confidence gate (τ = 0) the detector flagged 95% of facts and inference fell to chance (0.048). An empty memory disagrees with every fact in phase 0. Everything goes private, the shared path never learns the rule, and the loop does not recover. The matched random router at τ = 0 fell the same way (0.048), because it routes the same number of facts. As nano-lm-ad pointed out, `routing_loo.py` (`INVESTIGATION.md`, Question 8) avoids this because it judges each fact against the entity's other facts rather than against the memory's current prediction. A confidence gate is one fix; judging against other evidence is another.

## Limits

- **Stationary world.** No rule changes. nano-lm-d5's G04 (PR #204) shows that a router sending surprises to private slots holds an old rule after the rule changes; this detector would do the same.
- τ was chosen on one development seed, among six values.
- Type information is given; value tables, frozen addresses, synthetic world. The private table is per-fact storage, so this is an augmented-memory result. Not evidence about a language model.
- All twenty-five runs finished on the first attempt. Another session's two short runs overlapped the first few; only accuracy is measured, so this does not affect the results.

## Post-hoc diagnostic: what the confidence gate selects (development seed 4100, added after the results)

Measured after the held-out results were committed (`d5e5bab`), on the development seed only, with the `none` arm's memory, before each phase's training. The question came from nano-lm-d5 while calibrating its change detector.

- The memory's top softmax probability stays close to chance (1/32 = 0.031) for a long time and grows with phase: median 0.034 at phase 1, 0.050 at phase 10, 0.078 at phase 20, 0.169 at phase 39. Right and wrong answers overlap almost completely (phase 39 medians 0.169 and 0.131).
- In phases 30–39, corrections are the confident ones: median 0.216, and 95% are at or above 0.1. Regular new facts have median 0.131 (68% at or above 0.1); exceptions have median 0.117 (60%).

**Reading.** The gate works because a correction contradicts a value the memory has already stored with confidence. It is closer to "this contradicts something stored" than to a general surprise detector, which fits the detector routing 0.91 of corrections but only 0.70 of exceptions. Because the scale grows with training, τ also acts as a phase-dependent warm-up, so the same τ would flag more in a longer run. Neither point changes the registered results. Both narrow what they show: this is evidence for detecting contradictions of stored facts, not for detecting exceptions or rule changes in general.
