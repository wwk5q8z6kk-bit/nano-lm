# Gate 0 — nano-lm Program 0

**Project:** nano-lm (empirical + evidence + fabric + benchmark Program 0)

**Not:** a multi-lab research institution.

```text
REGISTRY_SCHEMA = PASS
TASK_VALIDATION = PASS
SOURCE_DIGEST_BINDING = PASS
PER_ITEM_TRACEABILITY = PASS
REPEATED_RUN_REPRODUCIBILITY = PASS
FAILED_RUN_PRESERVATION = PASS
CLEAN_CLONE_EXECUTION = PASS
LAYER1_BOUNDARY = PASS
TAG_INTEGRITY = PASS
DEFAULT_TEST_SUITE = PASS
PROGRAM0_STATUS = INFRA_SMOKE_PASS
PROGRAM1_STATUS = BLOCKED
leaderboard_eligible = false
evidence_ledger_eligible = false
```

## Detail

```json
{
  "TASK_VALIDATION": [
    "Validating tasks: ['nano_held_value_sentinel']",
    "All tasks found and valid"
  ],
  "SOURCE_DIGEST_BINDING": {
    "sha256": "ed5e8171cf13a4e802ecc6635740e8ad3977064eea7e4149b331a20792dee0a2",
    "n": 4,
    "path": "benchmarks/adapters/lm_eval/fixtures/held_value_sentinel_n4.json"
  },
  "REPEATED_RUN_REPRODUCIBILITY": {
    "run_id": "67ad855dcdc37d490a5c2d8cc1d21531d1cfa9b2232558d39c05db6797230233",
    "exact_match": 1.0,
    "bench_hash": "1c00868cafe611e0dac3e59e249d67430a2269babd04bfe9fbad519853b555b5",
    "solver_hash": "06152ca6d2a07879e7907439ac10ffa33ecaba341284aceb71b4e4ef650904cd"
  },
  "LAYER1_BOUNDARY": "EVIDENCE_LEDGER/MANIFEST clean; charter deleted; Program 0 does not edit freeze",
  "TAG_INTEGRITY": {
    "paper-alpha-v1": "2eba7de09a6ee110f37c8fb9f128faefbafab379",
    "post-alpha-evidence-freeze-2026-07-31": "e916db8cf5f9815b07988265a3d26066fb5a51d1"
  },
  "DEFAULT_TEST_SUITE": [
    "...........................                                              [100%]"
  ]
}
```

## Organizational expansion check

No Laboratory Charter / Discovery Lab / AI Scientist Lab / Program Q / Hardware Co-Design Lab files remaining (forbid-list mentions in Constitution OK).

Disposition: `PROGRAM0_READY_FOR_OWNER_COMMIT`
