# Council Hybrid Closeout

*Manual execution of Council-of-Five hybrid (2026-07-31). Autonomous path blocked (Claude credit balance too low).*
*Generated 2026-07-31T18:00Z*

## One-sentence claim (Translator)

> Under the frozen E1 utility, classical/rules methods beat official LoRA-160M on this closed scribe task (**KILL**; M1 U≈0.999 vs official M0 U≈0.925, δ=0.05).

## Verify card (Auditor)

| Check | Result |
|---|---|
| HEAD | `2e03e0df564008cf51c4309e9dbdf01a59c3c7b5` |
| `pytest -q` | PASS (`...........................                                              [100%]`; exit 0) |
| E1 verdict | **KILL** |
| Official M0 | `M0_pythia160m_lora` U=0.9252173639550433 |
| Best non-LM | `M1_template` U=0.9989993963311425 |
| margin / delta | 0.07378203237609926 / 0.05 |
| venue | `runpod-cuda` |
| `results_e1_utility.json` SHA256 | `a5117d2cad25ca53df7d7f3cdb25c563b36cab5cf1ac63be3867b53db76b760f` |
| Active RunPod pods | none expected under freeze |

## Quarantine (Deprecation Prophet) — do not expand

These paths stay **gated / non-authoritative** until a *written* re-scope (new utility or problem) exists:

| Path / track | Status |
|---|---|
| `trajectory/e2/` | GATED_STOP — no LoRA-mechanism claims |
| Fabric V2 / NanoScribe control plane | GATED — harness ≠ product architecture |
| Residual continua / Stage M curiosity | GATED |
| AAEA P2 eng sprint | Optional only with explicit owner authorize — **not** started by this closeout |
| E4 / R★ further runs | No curiosity reopen; revision only under written budget |

## Soft-freeze teeth (Saboteur)

Reopening any gated track requires **all** of:

1. New or amended prereg with explicit utility/problem statement
2. Owner authorization string in-session (`OWNER_RESCOPE_OK` or stronger)
3. No “just one offline CUDA check” without (1)–(2)

## Deletionist constraint

This closeout adds **documentation only**. No AAEA P2 implementation. No new experiments. No tag/push in this step.

## Tag gate

Proposed reconciled freeze tag remains **owner-only** (see `EVIDENCE_CURRENT.md`).  
**Not created** by this hybrid run. Existing immutable tags:

- `paper-alpha-v1` — do not move
- `post-alpha-evidence-freeze-2026-07-31` — do not move

## Program state after this file

`IDLE_AFTER_HYBRID_DOCS` — waiting on owner for optional `OWNER_TAG_OK` / commit of this card; otherwise remain idle.
