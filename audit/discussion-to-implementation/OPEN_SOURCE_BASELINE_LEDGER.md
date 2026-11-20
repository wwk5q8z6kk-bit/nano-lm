# Open-Source Baseline Ledger

*Search date: 2026-07-31. No installs performed.*

## Search method

Repository-wide ripgrep for: LangExtract, llguidance, Guidance, Outlines, Instructor, RAGChecker, OpenEvals, Sigstore, model-transparency.

**Result: zero matches** in code/docs/deps.

Also inspected `requirements.txt`, `requirements-ml.txt`, `pyproject.toml`, `environment.yml` — no such deps.

## Per-item ledger

| Item | Discussed in repo? | Adapter? | Benchmark? | Tests? | Pinned dep? | Recommendation |
|------|--------------------|----------|------------|--------|-------------|----------------|
| LangExtract | No hits | No | No | No | No | DEFER |
| llguidance | No hits | No | No | No | No | DEFER |
| Guidance | No hits | No | No | No | No | DEFER |
| Outlines | No hits | No | No | No | No | DEFER |
| Instructor | No hits | No | No | No | No | DEFER |
| RAGChecker | No hits | No | No | No | No | DEFER |
| OpenEvals | No hits | No | No | No | No | DEFER |
| Sigstore / model-transparency | No hits | No | No | No | No | DEFER (release hygiene later) |
| Stronger open-model baselines | Pythia used | Partial | Yes (Stage T) | Limited | transformers/peft in ml reqs | ALREADY_IMPLEMENTED (Pythia) / BENCHMARK_ONLY for new families |
| SSM comparisons | No | No | No | No | No | REJECT for now |
| Pointer-generator (See et al.) | Optional M6 in E1 prereg | No | No | No | No | DEFER |
| In-house constrained decode (M4) | Yes | Yes (`e1/methods.py`) | E1 | No unit tests | N/A | ALREADY_IMPLEMENTED (custom) |
| In-house template/dict (M1/M2) | Yes | Yes | E1 | No | N/A | ALREADY_IMPLEMENTED |

## Justification

External constrained-decoding / extraction libraries do not currently add decision value over E1 in-house M1–M5 on the closed task (already KILL). Integrating them now expands surface area without a live kill-gate that needs them. If Stage 4 (E4) is authorized, BENCHMARK_ONLY adoption of one structured-decoding library as an additional classical/constrained reference could be reconsidered — still not product integration.
