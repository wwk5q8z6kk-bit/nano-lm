# nano-lm

NanoLM is a research program building a general-purpose language model with frontier capability as its goal. The original 3.15M-parameter model was trained from scratch. Current work tests which architectures improve language modeling, reasoning and transfer at an affordable training and inference cost.

NanoScribe was an earlier medical application. Its implementation and experimental record remain available for reproduction. General-model development proceeds independently of its results.

## Current work

1. Recover native checkpoints and measure them on the same held-out text. Record tokenizer, training provenance, context length, loss, throughput and memory.
2. Compare promising dense and hybrid architectures with matched data and training budgets. Test whether improvements persist across sources and seeds before increasing scale.
3. Evaluate instruction following and reasoning separately, recording any contribution from adaptation, tools or external models.

Start with the [current experiments](docs/EXECUTION_PLAN.md), [machine-readable status](docs/ACTIVE_NOW.json) and [model research map](docs/research/MODEL_RESEARCH_PROGRAM.md). The [project charter](docs/PROJECT_CHARTER.md) defines the objective, and the [documentation index](docs/README.md) links the supporting material.

## Established findings (scoped)

| Result | Scoped finding |
|--------|----------------|
| Original Nano | Full 3.15M pretrain → alignment stack built from scratch |
| Scribe v1/v2 | Faithfulness failures exposed template/OOD limits |
| Stage A | Verification achieved measured presented precision under synthetic verifier relation |
| Stage S | Average task bars improved; held-out tail gap persisted |
| Paper α | Open-vocabulary held-out copying failure localized |
| E1 | Classical won on **old closed-task** utility (not program kill) |
| E4 | Classical won on **tested R★** |
| H6 | One-shot development evaluation **REJECTED** the exact coupling: copying passed, abstention calibration failed |

## Repository map

| Path | Role |
|------|------|
| `docs/` | Current program truth |
| `papers/` | Science and evidence |
| `trajectory/` | Experimental records |
| `pretrain/`, `sft/`, `scribe/` | Original pretraining, alignment and historical scribing experiments |
| `nano_ai/` | Learned Nano transcript-to-structure research through frozen H6 (integrated) |
| `nano/` | Domain-general contracts, evidence, state, retrieval, and runtime (integrated) |
| `nanoscribe/` | Historical P1 encounter proposal, constrained evidence binding, and evaluation |
| `artifacts/nano_h6/` | H6 terminal decision and authenticated evaluation output |
| `fabric/` | Verification harness (integrated) |
| `wedge_v1/` | Verified document intelligence (integrated) |
| `frontier/` | Branch notes only |

## Frozen evidence snapshot — 2026-07-31

**E1 (one sentence):** Under frozen E1 utility on the **old closed scribe task**, classical beat official LoRA-160M (**KILL** scoped to that task).

| | |
|---|---|
| **Freeze tag** | `post-alpha-evidence-freeze-2026-07-31` |
| **Ledger** | [`papers/EVIDENCE_LEDGER.md`](papers/EVIDENCE_LEDGER.md) |
| **Outsider summary** | [`papers/PUBLIC_ONE_PAGER.md`](papers/PUBLIC_ONE_PAGER.md) |

Full table: [`audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`](audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md)

## Quick repository verification

```bash
python scripts/check_active_now.py
python scripts/check_docs_integrity.py
pytest fabric/test_fabric.py trajectory/test_recompute_c3.py -q
```

Full reproduction of papers/training: [`trajectory/REPRODUCIBILITY.md`](trajectory/REPRODUCIBILITY.md)

## License

MIT — [`LICENSE`](LICENSE)
