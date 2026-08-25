# nano-lm

**Nano is a research and engineering program for building compact, reliable intelligence systems.**

We begin with **medical scribing** because reliably transforming a messy conversation into an evidence-grounded record forces a system to solve exact information transport, uncertainty, temporality, structured state, coherent generation, and verification in one consequential domain.

The program then advances from scribing → summarization → longitudinal charting → synthesis → reasoning → planning → action → adaptation.

The original **3.15M from-scratch language model** remains the project's experimental foundation — not its final architecture or size target.

## Macro-phases

```text
FOUNDATION I — P1 Master Scribing (faithful capture)

FOUNDATION II — P2 Summarization + P3 Charting (compression + longitudinal state)

INTELLIGENCE EXPANSION — P4–P9 (synthesis → reasoning → planning → action → adaptation)
```

## Canonical documentation

**Start here:** [`docs/README.md`](docs/README.md) · Authority: [`docs/PROJECT_AUTHORITY.md`](docs/PROJECT_AUTHORITY.md) (typed — empirical evidence ≠ program mission)

| Document | Purpose |
|----------|---------|
| [PROJECT_CHARTER](docs/PROJECT_CHARTER.md) | Mission, macro-phases, naming |
| [CAPABILITY_LADDER](docs/CAPABILITY_LADDER.md) | P1–P9 programs |
| [SYSTEM_ARCHITECTURE](docs/SYSTEM_ARCHITECTURE.md) | Nano Core + DomainPacks |
| [ACTIVE_NOW](docs/ACTIVE_NOW.md) | Current gate and policy |
| [EXECUTION_PLAN](docs/EXECUTION_PLAN.md) | Executable tasks |

`papers/` = **science**. `docs/` = **current program truth** (must not contradict tagged evidence).

## Product naming (medical DomainPack)

```text
Nano Core + Medical DomainPack + P1  =  NanoScribe

Nano Core + Medical DomainPack + P1–P3  =  longitudinal medical documentation intelligence (not yet earned)

P4–P9  =  broader synthesis, reasoning, planning, action, adaptation
```

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
| H6 / span-port lineage | Pretraining/transfer research exists — **cross-branch, not yet integrated** in this tree |
| P1 foundation (master) | Encounter v0, evidence transport, Qwen harness — **integrated** in `nanoscribe/` (#37–#41) |
| Accelerated campaign v2 | Tool calling, agent platform, native/student tracks — **active on frontier branch** |

## Repository map

| Path | Role |
|------|------|
| `docs/` | Current program truth |
| `papers/` | Science and evidence |
| `trajectory/` | Experimental records |
| `pretrain/`, `sft/`, `scribe/` | Mechanism / training (integrated) |
| `nanoscribe/` | P1 NanoScribe — encounter v0, adapters, harness, campaign |
| `artifacts/campaign/` | Manifest-gated paid experiment artifacts |
| `fabric/` | Verification harness (integrated) |
| `wedge_v1/` | Verified document intelligence (integrated) |
| `frontier/` | Branch-local campaign configs + [NEXT.md](frontier/NEXT.md) |

## Current program status — 2026-08-23

- **Frontier:** P1 Master Scribing — [`docs/ACTIVE_NOW.md`](docs/ACTIVE_NOW.md)
- **Active branch:** `frontier/accelerated-research-campaign-v2` — tool calling, agent platform, multi-track campaign ([`docs/research/ACCELERATED_CAMPAIGN.md`](docs/research/ACCELERATED_CAMPAIGN.md))
- **Master foundation:** `origin/master` @ `c4822b9` — encounter v0 + harness (#37–#41)
- **Compute:** RunPod active; routine training **ALLOWED_WITHIN_ACTIVE_EXPERIMENT_BUDGET**; materially costly **EXPERIMENT_SCOPED**; confirmatory **PREREG + EXPERIMENT_SCOPED**
- **Data:** no PHI / private owner material in current Nano experiments

### Compute evolution

```text
Original Nano
→ trained from scratch locally on Apple Silicon

Current Nano research
→ local development + RunPod GPU training

Future deployment
→ chosen independently from training infrastructure
```

Training venue ≠ deployment venue. Compact/local/private deployment remains a long-term optimization axis. Details: [`docs/infrastructure/RUNPOD.md`](docs/infrastructure/RUNPOD.md).
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
pytest fabric/test_fabric.py trajectory/test_recompute_c3.py nanoscribe/test_encounter_v0.py nanoscribe/test_tool_calling.py -q
```

Full reproduction of papers/training: [`trajectory/REPRODUCIBILITY.md`](trajectory/REPRODUCIBILITY.md)

## License

MIT — [`LICENSE`](LICENSE)
