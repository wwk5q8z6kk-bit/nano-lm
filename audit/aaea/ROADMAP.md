# AAEA Roadmap — nano-lm

## P0 — Critical / immediate
- [x] Fix pytest collection GPU false-failure (`pytest.ini`)
- [x] Expand `.gitignore`
- [ ] Owner: push freeze commits + authorize tag `post-alpha-evidence-freeze-2026-07-31`

## P1 — High / this sprint (freeze-compatible)
- [ ] Offline E1 U recompute pytest from committed JSON
- [ ] Offline E3 normalize pytest (0/486 invariant)
- [ ] Restore packaging files (`requirements.txt` / `pyproject.toml`)
- [ ] Document E1 latency/cost normalization schema

## P2 — Medium / next sprint
- [ ] `main()` guards for import-unsafe training scripts
- [ ] Remove stage_m auto-pip; fail closed
- [ ] Fabric slice docstring boundary alignment

## P3 — Low / backlog
- [ ] Broader scorer unit tests
- [ ] Installable package layout
- [ ] Any NanoScribe implementation only under explicit new owner scope
