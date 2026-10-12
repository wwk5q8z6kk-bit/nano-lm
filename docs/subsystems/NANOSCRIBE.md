# NanoScribe (P1 subsystem)

**Nano Core + Medical DomainPack + P1** — faithful encounter capture. Not full medical intelligence; that is earned through P2–P9.

## Truth object

```text
immutable source (transcript)
        ↓
Encounter Representation v0  (nano.encounter.v0)
        ↓
CandidateAtom proposals  (model + software)
        ↓
ConstrainedSelector + verification
        ↓
verified encounter record
        ↓
note rendering (view — not primary truth)
```

Canonical schema: `nanoscribe/encounter.py` · tests: `nanoscribe/test_encounter_v0.py`

## Software layers

| Layer | Path | Role |
|-------|------|------|
| Encounter schema | `nanoscribe/encounter.py` | Typed entities, events, evidence spans, invariants |
| Adaptation | `nanoscribe/adapt.py` | `ModelCandidate`, `CandidateAtom`, gold alignment |
| Inference — structured JSON | `nanoscribe/structured_inference.py` | Default campaign path (`response_format: json_object`) |
| Inference — tool calling | `nanoscribe/tool_inference.py` | `submit_candidate_atoms` for vLLM/SGLang + Qwen3 coder parser |
| Adapters | `nanoscribe/adapters.py` | Fixture, compact, serverless strong control, Kimi/API teachers |
| Harness | `nanoscribe/harness.py` | Three-track (and extended) evaluation harness |
| Native Nano | `nanoscribe/native/` | Scratch architecture screening (~30M–100M) |
| Campaign | `nanoscribe/campaign.py`, `campaign_fanout_lib.py` | Manifest-driven fan-out, wallet gates |
| Agent platform | `nanoscribe/agent_canary.py`, `coding_tools.py` | Sandboxed agent tools for campaign automation |
| Capabilities | `nanoscribe/capabilities/` | Registry for scribing, summarize, table (P1+ hooks) |

## Model interface

Primary contract: **structured `CandidateAtom` JSON** (tool-call path is equivalent after parse).

```text
ModelInput (transcript + atom specs)
        → adapter.propose()
        → ModelCandidate
        → adapt() → PredictedEncounter
        → evaluator / verifier
```

See [infrastructure/TOOL_CALLING.md](../infrastructure/TOOL_CALLING.md) for structured vs tool modes.

## Evaluation tracks (harness)

Integrated in `nanoscribe/tracks.py`:

| Track | Role | Cost class |
|-------|------|------------|
| **Fixture** | Deterministic CI | zero local |
| **Compact** | Qwen2.5-1.5B historical continuity | routine RunPod |
| **Serverless strong control** | Qwen3.8-27B span-port ceiling | serverless burst |
| **Frontier teacher** | Kimi / managed API capability ceiling | experiment-scoped |
| **Native** | Scratch vNext screening | campaign budget |

Qwen adapters are **control paths**, not Nano itself. See [research/ACCELERATED_CAMPAIGN.md](../research/ACCELERATED_CAMPAIGN.md).

## Relation to legacy `scribe/`

`pretrain/`, `sft/`, `scribe/` = mechanism-model era (Paper α, Stage A/S, E1 task).  
`nanoscribe/` = P1 product frontier with encounter representation v0 and evidence transport.

Historical failures inform design — [FAILURE_TO_ARCHITECTURE.md](../FAILURE_TO_ARCHITECTURE.md).

## Medical domain contract

Exit gate, human eval requirements: [domains/medical/SCRIBING.md](../domains/medical/SCRIBING.md).

## Verify

```bash
python3 -m pytest nanoscribe/test_encounter_v0.py nanoscribe/test_evidence_transport.py -q
python3 -m pytest nanoscribe/test_tool_calling.py -q
```
