# Medical Evaluation Protocol

## Layers

| Layer | What it validates | Sufficient for clinical claim? |
|-------|-------------------|-------------------------------|
| Unit / schema tests | Parsing, offsets, schema | No |
| Synthetic scribe gates | Faithfulness on template distribution | No |
| Fabric / verifier regression | Verifier relation on scoped tasks | No |
| External medical dialogue benchmark | OOD dialogue → record quality | Necessary, not sufficient |
| Blinded human clinician review | Edit burden, critical errors | **Required for P1 exit** |

## P1 exit evaluation (planned)

1. **Automatic metrics** on external held-out set (precision, omission, span correctness, attribution axes in [SCRIBING.md](SCRIBING.md))
2. **Blinded human evaluation** — clinician edit effort, critical-error severity, time to acceptable note
3. **Owner sign-off** on P1 exit record

## Forbidden claims

- "Clinically validated" from mock/synthetic benchmarks alone
- Deployment-ready medical product without human protocol completion
- Replacing E1/E4 scoped kills with broad "LM doesn't work" statements

## Historical instruments

- Scribe gates: `scribe/AUDIT.md`, `gate_sft.py` patterns
- Paper α copying instrument: `trajectory/`
- E1/E4 utility: frozen U functions on defined tasks/regimes

New P1 eval must be **preregistered** before execution ([research/EXPERIMENT_STRATEGY.md](../../research/EXPERIMENT_STRATEGY.md)).
