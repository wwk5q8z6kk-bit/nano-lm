# Fabric Feature Boundary (Phase 7)

Tests run this freeze:
```
pytest fabric/test_fabric.py → 8 passed
```

| Capability | Implemented | Tested | Measured | Not implemented |
|------------|-------------|--------|----------|-----------------|
| Typed claims | Yes | Yes | Yes | — |
| Source spans required for VERIFIED/PRESENT | Yes | Yes (schemas/tests) | Yes | — |
| Contradiction state + counter-evidence | Yes (v2) | Yes | Yes | — |
| Abstention | Yes | Yes | Yes | — |
| Content-addressed IDs | Yes | Partial | Yes | — |
| Per-run ledger serialization | Yes | No dedicated | Yes (jsonl) | — |
| Append-only transactionality | **No** | No | No | Missing |
| Concurrency control | No | No | No | Missing |
| Schema migrations | No | No | No | Missing |
| Stale-write protection | No | No | No | Missing |
| Duplicate idempotency | No | No | No | Missing |
| Long-term memory | No | No | No | Missing |
| Control kernel | No | No | No | Missing |
| Task graph | No | No | No | Missing |
| Permissions | No | No | No | Missing |
| UI | No | No | No | Missing |
| Distributed execution | No | No | No | Missing |
| Semantic verifier | No | No | No | Missing |
| Calibrated risk controller | Minimal if/else | decide() only | Labels only | Calibration missing |

**What tests establish:** model-free pins that v1/v2 grounding, absence-never-from-silence,
wrong-speaker/wrong-slot behaviors, and decide() mapping hold on fixtures.
**What they do not establish:** e2e run_slice on checkpoints in CI; ledger durability; NanoScribe OS properties.
