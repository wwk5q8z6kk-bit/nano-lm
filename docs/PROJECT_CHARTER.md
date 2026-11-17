# Project Charter — NanoLM

## Mission

> **Build the most capable, reliable, and efficient general-purpose foundation language model achievable within a strict 2.0–4.0 billion parameter budget—maximizing useful intelligence per parameter, training token, unit of compute, and byte of inference memory.**

NanoLM is **the foundation model itself**—not a domain-specific Scribe engine, clinical workflow, DevX feature, or hardcoded agent wrapper. Earlier medical scribing and 3.15M–160M mechanism studies remain preserved as historical empirical records ([`archive/legacy/PROJECT_CHARTER_SCRIBE_ERA_20261012.md`](archive/legacy/PROJECT_CHARTER_SCRIBE_ERA_20261012.md), [`PROJECT_AUTHORITY.md`](PROJECT_AUTHORITY.md)), not architectural or roadmap constraints.

## Hard Parameter & Efficiency Constraints

| Constraint | Specification |
|------------|---------------|
| Preferred target | **~2.0B** deployed parameters (`nanolm-2b-*` at `2.04B`) |
| Permitted range | **2.0B – 4.0B** deployed parameters |
| Hard maximum | **4,000,000,000** deployed parameters (never exceeded; enforced in [`nanolm/config.py`](../nanolm/config.py)) |
| Accounting rule | Counts embeddings, output projections, all routed + shared experts, routers, norms, and deployed MTP heads |

## Core Capability Objectives

1. **Software Engineering & Programming** — Multi-language generation, debugging, refactoring, algorithms, and repository-level comprehension verified by executable test suites.
2. **Mathematical & Logical Reasoning** — Multi-step quantitative, deductive, constraint-satisfaction, and scientific reasoning with genuine out-of-distribution generalization.
3. **Information Processing & Summarization** — Long-document synthesis and compression preserving exact metrics, relationships, qualifications, and exceptions.
4. **Structured Generation, Charts, and Diagrams** — Native reasoning into strict JSON schemas, chart specifications (valid chart types, axes, units, values), and Mermaid diagrams (`flowchart`, `sequenceDiagram`, `stateDiagram`) with verified topology and directionality.
5. **General Language & Knowledge** — High-clarity instruction following, technical communication, and broad factual synthesis.
6. **Long-Context Intelligence** — Accurate multi-needle retrieval, cross-document connection, and distant contradiction detection across context positions.

## Canonical NanoLM Documentation

- [Official NanoLM Charter](nanolm/CHARTER.md)
- [Phase 1 Ground-Truth Audit](nanolm/PHASE1_AUDIT.md)
- [Architecture, Data & Training Specification](nanolm/ARCHITECTURE_AND_TRAINING.md)
- [Active Program Status](ACTIVE_NOW.md) · [RunPod Compute Policy](infrastructure/RUNPOD.md)
