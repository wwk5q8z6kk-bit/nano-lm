# P2 Summarization & P3 Charting (Future Contract)

**Status:** Specified — not current implementation priority.

P1 interfaces must not block these stages. Serious P2/P3 engineering waits for [SCRIBING.md](SCRIBING.md) exit criteria.

## P2 — Summarization

**Question:** What matters?

Hierarchical compression over **verified source state** — not lossy prose-on-prose.

```text
source → atomic facts/events → encounter record
→ section summaries → encounter summary → multi-encounter summary
```

Measures: factual precision, critical omission, salience, compression, provenance, contradiction/uncertainty retention.

## P3 — Charting

**Question:** What is the state, and how has it changed?

```text
encounters + labs + meds + diagnoses + procedures + messages
→ entity resolution → event graph → temporal ordering
→ state transitions → supersession / contradiction → current state → longitudinal chart
```

Distinguish: event vs persistent fact vs historical vs current vs resolved vs superseded vs uncertain vs contradicted.

## Required record fields (design now)

```text
entity_id · event_id · encounter_id · source_id
timestamp / interval · state_transition
supersedes · contradicts · supports · derived_from
confidence · evidence spans
```

## Nano Core vs Medical pack

- **Core:** identity, temporality, provenance graph primitives
- **Medical pack:** clinical ontology, section semantics, clinical eval protocols
