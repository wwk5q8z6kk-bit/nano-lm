# Wedge v1 — Verified local document intelligence

**Implementation:** `wedge_v1/`  
**Role:** Supporting subsystem under Nano Core — **not** the entire Nano program.

## Purpose

Local-first **research document intelligence**:

> Ingest a private corpus of technical/biomedical notes; answer structured questions with **evidence spans**; abstain when unsupported; never default to an unverified generative substrate.

## Design principles

| Principle | Implementation |
|-----------|----------------|
| Classical-first | exact find → BM25 → E-class plugins → optional escalate on ABSTAIN only |
| Verifiable output | claims with `doc_id`, char spans, `SUPPORTED` / `CONTRADICTED` / `ABSTAIN` |
| Smallest sufficient solver | LM only if \(\Delta U > \delta\) on this workflow (currently **not indicated**) |
| Local / private | default offline; owner corpus via env, not in git |

## Utility (draft)

\[
U = Q - 0.5\,E - 0.3\,R - 0.02\,L - 0.05\,C
\]

Freeze weights before comparative scoring.

## CLI surface (representative)

```bash
python -m wedge_v1 ask "question" --corpus /path/to/docs
python -m wedge_v1 compare "term" --corpus /path/to/docs
python -m wedge_v1 owner-ready --demo
python -m wedge_v1 habit --list
python -m wedge_v1 review --interactive
```

## Relation to P1 scribing

Wedge exercises **retrieval, evidence binding, abstention, and review** on documents — skills required for P1 encounter records and longitudinal charting. It is **not** a substitute for medical scribe evaluation or clinical validation.

## Historical spec

Full task pack (40 tasks T01–T40) and phase history: `papers/WEDGE_V1.md` (superseded stub → this file).

## Evidence artifacts

- Classical baseline: `wedge_v1/results_wedge_v1_classical.json`
- Phase 3 E-class: `wedge_v1/results_wedge_v1_phase3_eclass.json`
- Dogfood: `wedge_v1/results_wedge_v1_dogfood.json` (gitignored runtime)

## Non-goals

General chatbot; NanoScribe clinical deployment; old `OLD_TASK_U` scribe template world as primary product.
