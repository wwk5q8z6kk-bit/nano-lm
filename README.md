# nano-lm

**Nano is a research and engineering program for building compact, reliable intelligence systems.**

We begin with **medical scribing** because reliably transforming a messy conversation into an evidence-grounded record forces a system to solve exact information transport, uncertainty, temporality, structured state, coherent generation, and verification in one consequential domain.

The program then advances from scribing → summarization → longitudinal charting → synthesis → reasoning → planning → action → adaptation.

The original **3.15M from-scratch language model** remains the project's experimental foundation — not its final architecture or size target.

## Canonical documentation

**Start here:** [`docs/README.md`](docs/README.md)

| Document | Purpose |
|----------|---------|
| [PROJECT_CHARTER](docs/PROJECT_CHARTER.md) | Mission and optimization target |
| [CAPABILITY_LADDER](docs/CAPABILITY_LADDER.md) | P1–P9 capability sequence |
| [SYSTEM_ARCHITECTURE](docs/SYSTEM_ARCHITECTURE.md) | Nano Core + DomainPacks |
| [ROADMAP](docs/ROADMAP.md) | Historical arc → current program |
| [ACTIVE_NOW](docs/ACTIVE_NOW.md) | Current gate and work |
| [EXECUTION_PLAN](docs/EXECUTION_PLAN.md) | Executable tasks |

`papers/` holds **science** (preregistrations, results, manuscripts). `docs/` holds **current program truth**.

## What Nano is building

```text
Nano Core  +  Medical DomainPack  =  NanoScribe / Medical Intelligence
```

**Nano Core** (domain-general): representation, memory, evidence, state, retrieval, synthesis, planning, generation, verification.

**Medical DomainPack** (first proving ground): clinical schema, semantics, evaluation — medicine is not the permanent boundary.

**Governing rule:** Do not ask the model to learn what software solves more reliably; do not maintain brittle software for what learned representations solve more generally. Use the **smallest sufficient solver**.

## Capability ladder (summary)

```text
P1 Scribing → P2 Summarization → P3 Charting → P4 Synthesis
→ P5 Questioning → P6 Reasoning → P7 Planning → P8 Tools/Action → P9 Adaptation
```

**Current frontier:** P1 Master Scribing — see [domains/medical/SCRIBING.md](docs/domains/medical/SCRIBING.md).

## Developmental arc (one story)

```text
3.15M from scratch → SFT / DPO / RLVR → medical scribing gates
→ held-out copying failure (Paper α) → scale / diversity / adaptation
→ verification (Fabric, Stage G/A) → utility kill gates (E1, E4 on tested regimes)
→ Wedge (local verified document intelligence) → pretrained transfer / span work
→ Nano Core capability ladder (this program)
```

Historical kills (E1 on the old closed task; E4 on tested R★) are **scoped routing evidence** — not a kill of the full program.

## Repository map

| Path | Role |
|------|------|
| `docs/` | Current project truth and architecture |
| `papers/` | Science — evidence, preregistrations, manuscripts |
| `trajectory/` | Experimental records and reproducibility |
| `artifacts/` | Machine evidence bundles |
| `pretrain/`, `sft/`, `scribe/` | Mechanism / training experiments |
| `fabric/` | Verification regression harness |
| `wedge_v1/` | Local verified document intelligence ([subsystem doc](docs/subsystems/WEDGE.md)) |
| `nano_ai/` | Model / intelligence core (when present) |
| `frontier/` | Branch-local notes only — not canonical authority |

## Evidence and publication (protected)

- Paper α: tag `paper-alpha-v1` — [`papers/`](papers/)
- Evidence ledger: [`papers/EVIDENCE_LEDGER.md`](papers/EVIDENCE_LEDGER.md)
- Empirical foundation: [`papers/EMPIRICAL_FOUNDATION.md`](papers/EMPIRICAL_FOUNDATION.md)
- Outsider summary: [`papers/PUBLIC_ONE_PAGER.md`](papers/PUBLIC_ONE_PAGER.md)

Do not move or rewrite freeze manifests, tagged result JSONs, or ledger files as part of ordinary doc edits.

## Public status (evidence-scoped, 2026-07-31)

**E1 (one sentence):** Under the frozen E1 utility on the **old closed scribe task**, classical/rules methods beat official LoRA-160M (**KILL**; not a statement about the full Nano capability program).

| | |
|---|---|
| **Freeze tag** | `post-alpha-evidence-freeze-2026-07-31` |
| **Fabric** | Verification slice — ≠ NanoScribe architecture |
| **E3** | Agent-applied rubric — not human/clinician evaluation |

Full status table: [`audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`](audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md)

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_active_now.py
pytest fabric/test_fabric.py -q
```

Training stack: `requirements-ml.txt` · Details: [`trajectory/REPRODUCIBILITY.md`](trajectory/REPRODUCIBILITY.md)

## License

MIT — see [`LICENSE`](LICENSE).
