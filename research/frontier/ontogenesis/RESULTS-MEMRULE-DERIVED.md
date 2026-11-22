# Addresses built from the data instead of from given types: results

Run 2026-11-22 under [PROTOCOL-MEMRULE-DERIVED.md](PROTOCOL-MEMRULE-DERIVED.md), committed in `25acd5d` before any confirmatory run. Seeds 4101–4105, single shared path, learning rate 1e-3, 200 steps per phase. The typed and random arms are reused from `results/memory_rule_sharing/` (same seeds, same world). Raw files: `results/memory_rule_derived/`. [Observed]

## Means over five seeds

| Entity vectors | Acquisition | Inference, new entities | Inference, old entities | Retention | Corrections |
|---|---|---|---|---|---|
| typed (hidden type given) | 0.836 | 0.737 | 0.734 | 0.770 | 0.034 |
| derived from facts shown at creation | 0.893 | 0.302 | 0.306 | 0.822 | 0.034 |
| random | 0.622 | 0.106 | 0.109 | 0.584 | 0.196 |

## Against the registered predictions

- G1 (derived inference at least 0.15 above random and at least 0.25 below typed): **held**, five of five (0.278–0.328).
- G2 (derived acquisition at least typed): **held**, five of five (0.890–0.898 against 0.821–0.853).
- Recorded without a prediction: derived codes recover 0.27–0.34 of the typed–random inference gap (mean 0.31).

## Reading

Built from data alone, the memory's addresses recover about a third of the generalisation that the given types provided, and acquire and retain better than with given types. Every typed-memory result in this line should be read with that factor in mind: given types overstate what a learner that must discover them would get, by about a factor of three in inference here.

What limits it is how the codes are built, not how much information there is. With 32 values per relation and 3 shown facts, a type is almost always identifiable from an entity's shown facts. But a sum of per-fact vectors makes two same-type entities similar only on the relations they both had shown, about 1.1 of 3 on average. Recovering types properly needs inference across entities, and a greedy online version collapsed on the development seed (protocol, "What I had seen"): with frozen addresses, an early wrong grouping cannot be revised. A memory whose addresses can be revised as evidence accumulates is the natural next step. That is the learned-address problem (KEYSTORE in `LEDGER.md`), whose earlier straight-through pilot failed.

Corrections are as poor as with given types (0.034). They track how much addresses are shared, not where the sharing comes from.

## Limits

- One way of building codes, chosen for simplicity and not tuned.
- 3 of 8 relations shown per entity; more shown relations would make derived codes closer to types.
- Value table, frozen addresses, synthetic world. Not evidence about a language model.
- All five runs finished on the first attempt.
