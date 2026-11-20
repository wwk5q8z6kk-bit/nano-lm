# U10 register-width real-text screen

This exploratory comparison resumes Antigravity's Unified Discovery Loop at d178771125351e8be6dc02d076ac1950f0ef6803 and opt-in register implementation 90f3e18802e4896f7d67a2680e023d45a6f4184c. The three models learn the next token from initialization. The sole proposed change to M22 reallocates part of the feed-forward layer to widen the rotation mixer. M21 is the strongest U09 control on macro-average 512-token validation bits per byte.

| Arm | Mechanism | Registers | FFN width |
|---|---|---:|---:|
| m21 | M21_HSPA_ConceptSlot_EvidPtr | 4 | 460 |
| m22 | M22_HSPA_FullCore_U09 | 4 | 460 |
| m22_wide | M22_HSPA_FullCore_U09 | 11 | 357 |

Each arm receives exactly the same 1,000 batches within each of two seeds, using 8 sequences of length 128, AdamW and the original U09 schedule. Corresponding tensors with matching shapes start at identical values. Each arm sees 1,024,000 tokens per seed. Training runs serially on one backend. Results include all six cells, regardless of outcome.

**Primary decision.** Advance the wider mechanism to another exploratory experiment only if it improves the four-domain mean 512-token BPB against both controls on each seed, both depth-three centered effective ranks on code reach the U10 target of 12, and its copy-256 accuracy meets the prior 0.95 criterion on each seed. A directional win on two seeds describes this pilot only. The archived U09 scale gate also requires separate semantic-alias extrapolation and speed tests, so this experiment alone cannot authorize a scale-up.

**Other measurements.** Publish all four domains at evaluation lengths 128, 256 and 512, actual parameter counts, sampled-window hashes, measured training speed, depth-by-depth rank and repeated-sequence copy and multi-query associative recall probes. BPB uses the original per-domain corpus-average bytes/token conversion. The validation windows select identical deterministic offsets from existing validation streams. Exact-copy and retrieval probes are synthetic. Ordered run timing on shared hardware is descriptive. Reuse of historical validation windows makes this a development comparison. New runs preserve CPU-portable checkpoints locally for evaluation-only replay.
