# Prior answer exposure and native alias recall

The native dual-copy head recovered the complete two-token answer in 487, 575 and 948 of 1,024 worlds when the scored alias was new to the query history. The generator scored zero in each seed and the identity-copy control scored nine. When the history supplied the query–answer pair, both copy models reached 1,024 in every seed. Repeated queries can therefore conceal a large, seed-dependent difference in recall on this task.

| Training seed | Dual copy, new | Dual copy, repeated | Generator, new / repeated | Identity copy, new / repeated |
|---|---:|---:|---:|---:|
| 31271 | 487 | 1,024 | 0 / 0 | 9 / 1,024 |
| 71339 | 575 | 1,024 | 0 / 0 | 9 / 1,024 |
| 91823 | 948 | 1,024 | 0 / 0 | 9 / 1,024 |

Each cell uses the same 1,024 worlds. Prior exposure converted 537, 449 and 76 dual-copy failures to successes. No paired world changed from success to failure. The two larger models have 167,660 and 167,590 parameters. The identity control has three. Reusing the same evaluation worlds across training seeds measures seed variation on that panel.

The target values came from a vocabulary range excluded during training. The records still included the target values in both conditions. This evidence supports testing associative recall on natural text. It does not measure broader language quality or isolate the contribution of each model component.

All nine fits completed on a single Google Vertex AI n2d-standard-4 CPU worker in us-central1 using the pinned PyTorch 2.4 image. The recorded job runtime was 18 minutes 8 seconds. At the preflight compute rate of $0.19433/hour, the running-time compute estimate is $0.059 before disk or billing adjustments. The protocol and executable were published at commit `13a39593b64c69e8870d3bccd161bd12206c4d8b` before the run. [Protocol and replay](PROTOCOL.md) · [Raw paired counts and world fingerprints](results.jsonl)
