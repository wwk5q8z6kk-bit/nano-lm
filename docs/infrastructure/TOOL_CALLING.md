# Tool calling for P1 scribing and coding-agent workflows.

Nano supports two parallel first-class inference paths for CandidateAtom extraction:

| Mode | OpenAI API | Default |
|------|------------|---------|
| `structured` | `response_format: json_object` | **yes** |
| `tool` | `tools` + `tool_choice` → `submit_candidate_atoms` | opt-in |

Structured JSON remains the campaign default. Tool calling is for vLLM/SGLang backends with `qwen3_coder` parser support (Qwen3.8, managed refs, student endpoints).

## Architecture

```text
prompt (transcript + atom slots)
  → OpenAI chat.completions
      structured path: response_format json_object
      tool path:       tools=[submit_candidate_atoms] + tool_choice
  → ToolCallParser
      tool_calls → validate → ModelCandidate.from_dict
      content JSON → JSON_FALLBACK (backward compat)
  → adapt() → PredictedEncounter
```

Core modules:

- `nanoscribe/candidate_schema.py` — JSON Schema derived from `CandidateAtom` contract
- `nanoscribe/tool_calling.py` — `ToolDefinition`, `ToolCall`, `ToolCallResult`, `ToolCallParser`
- `nanoscribe/tools.py` — OpenAI tool exports + vLLM env constants
- `nanoscribe/tool_inference.py` — tool-calling inference adapter
- `nanoscribe/coding_tools.py` — sandboxed coding-agent tool foundation
- `nanoscribe/structured_inference.py` — unchanged default JSON path (uses shared parser)

## vLLM serverless configuration

Deploy with tool-calling env vars (also in campaign manifests `vllm_env`):

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOOL_CALL_PARSER` | `qwen3_coder` | vLLM tool-call parser |
| `ENABLE_AUTO_TOOL_CHOICE` | `true` | Allow model auto tool selection |

**Precedence:** process env → `deploy_env` → manifest `vllm_env` → `DEFAULT_VLLM_TOOL_ENV`.

```bash
scripts/p1_serverless_launch.sh deploy
# or explicitly:
TOOL_CALL_PARSER=qwen3_coder ENABLE_AUTO_TOOL_CHOICE=true scripts/p1_serverless_launch.sh deploy
```

## Python usage

### Structured JSON (default)

```python
from nanoscribe.adapters import ServerlessQwen38StructuredAdapter

adapter = ServerlessQwen38StructuredAdapter(endpoint_id="your-endpoint")
batch = adapter.propose(model_input, atom_specs)
```

### Tool calling

```python
from nanoscribe.adapters import ServerlessQwen38ToolAdapter

adapter = ServerlessQwen38ToolAdapter(endpoint_id="your-endpoint")
batch = adapter.propose(model_input, atom_specs)
```

### Campaign fan-out

```bash
python3 scripts/campaign_fanout.py orchestrate --modes structured,tool --endpoint YOUR_ENDPOINT
```

### Parser harness (fixtures)

```bash
python3 scripts/tool_call_harness.py
python3 scripts/tool_call_harness.py --fixture valid_tool_calls.json
```

## Coding-agent tools

`nanoscribe/coding_tools.py` provides typed, sandboxed tools for agentic campaign scripts:

- `read_file`, `list_directory`, `search_code`, `apply_patch` (dry-run default), `run_command` (allowlisted binaries only)

Paths are confined to a sandbox root; no unrestricted shell.

## Tests

```bash
python3 -m pytest nanoscribe/test_tool_calling.py -q
python3 -m pytest nanoscribe/ -q
```

Fixtures live in `fixtures/tool_calls/`.

## Logging

Tool inference emits structured JSON log lines via `log_tool_call_event` (no secrets, no transcript text). Enable with standard Python logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```
