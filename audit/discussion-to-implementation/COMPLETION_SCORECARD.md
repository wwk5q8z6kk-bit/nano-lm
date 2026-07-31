# Completion Scorecard

*Weights declared before scoring. Tracks scored separately — do not average into one headline %.*

Scoring levels: `c=0` absent · `0.25` discussed · `0.5` partial · `0.75` implemented unverified · `1.0` implemented and verified.

Completion = Σ w_i c_i / Σ w_i.

## Empirical track

| Component | Weight w | c | Note |
|-----------|----------|---|------|
| Preregistration corpus | 1.0 | 0.75 | Many preregs; RESULT sections often stale |
| Data/pools/eval generators | 1.0 | 0.75 | Generators exist; some pools present |
| Kernels / runners | 1.2 | 0.75 | Extensive; E2/StageM incomplete |
| Raw artifacts (JSONL/checkpoints) | 1.2 | 0.50 | JSONL local+gitignored; checkpoints sparse |
| Deterministic recomputation | 1.0 | 0.75 | C3 yes; not universal |
| Audit / decision rules applied | 1.0 | 0.75 | E1/C3/C-1b strong; E3 human partial |
| Documentation sync | 0.8 | 0.50 | Widespread status drift |
| Manuscript integration (α) | 1.0 | 0.75 | Tagged paper; limitation text lag |
| Reproducibility packaging (git) | 1.0 | 0.50 | Post-α locks/results largely untracked |

**Empirical Completion ≈ 0.67** (computed below).

## Fabric track

| Component | Weight w | c | Note |
|-----------|----------|---|------|
| Schemas / typed packets | 1.2 | 1.00 | Verified by tests |
| Verifier v1/v2 | 1.2 | 1.00 | Tests + measured JSON |
| Risk controller | 1.0 | 0.50 | Minimal decide() |
| Ledger | 1.0 | 0.50 | JSONL truncate; hashes real |
| Telemetry | 0.8 | 0.50 | Aggregate only |
| Unit tests | 1.0 | 1.00 | 8/8 pass |
| Adversarial tests (full list) | 1.0 | 0.50 | Partial coverage |
| Calibration | 0.8 | 0.25 | Discussed |
| Integrations / adapters | 0.8 | 0.00 | Absent |
| Documentation accuracy | 1.0 | 0.50 | Overclaims Intent/append-only |

**Fabric Completion ≈ 0.61**.

## NanoScribe architecture track

| Component | Weight w | c | Note |
|-----------|----------|---|------|
| Specification docs | 1.0 | 0.75 | Rich specs; STOP gated |
| Control plane / kernel | 1.2 | 0.25 | Discussed |
| Memory | 1.0 | 0.25 | Discussed |
| Routing | 1.0 | 0.25 | Discussed |
| Tools | 0.8 | 0.25 | Discussed |
| Permissions | 0.8 | 0.25 | Discussed |
| Distribution | 0.8 | 0.00 | Absent |
| Observability | 0.8 | 0.50 | Minimal |
| UI | 0.6 | 0.00 | Absent |
| Release readiness (product) | 1.0 | 0.00 | E1 KILL blocks |

**Architecture Completion ≈ 0.26**.

## Computed values

| Track | Completion |
|-------|------------|
| Empirical | **0.67** |
| Fabric | **0.61** |
| NanoScribe architecture | **0.26** |

Interpretation: empirical science core is substantially real but packaging/repro lag; fabric is a verified thin slice with over-documented aspirations; NanoScribe architecture is mostly paper.
