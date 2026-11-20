# nano-lm discovery evidence register

Companion to the [strategic disposition report](strategic-disposition-report.md). Investigation date: **2026-10-22 America/New_York** (some tool timestamps fall on October 23 UTC).

## Scope, authority and evidence classes

Repository observations refer to `/Users/mac/Projects/nano-lm` at `61a06524c9d1a9998f64c90d1035831b575713eb`, branch `codex/p1-h7-scribing-path`, unless explicitly identified as remote or historical. Line numbers refer to that snapshot. Relative links resolve from this directory. Reading a retained experiment is not executing it. A source's claim about its product is not an independently measured outcome.

- **Verified fact:** source/configuration inspected, artifact read/recomputed, or command executed in this investigation. State which kind.
- **Inference:** a conclusion drawn from those observations; alternative explanations remain possible.
- **Estimate:** prospective effort/cost judgment, not an observed quantity or commitment.
- **Hypothesis:** benefit, demand or generalization requiring a new test.

Confidence describes support for the exact scoped claim. High confidence in an artifact's contents does not confer high confidence in external validity. Typed authority is [PROJECT_AUTHORITY.md](../PROJECT_AUTHORITY.md): empirical artifacts over narrative, current execution sources over stale plans, code/tests over implementation prose. These reports do not change any gate.

## Repository evidence

| ID | Significant claim and class | Supporting source/observation | Confidence, limits and contradictions |
| --- | --- | --- | --- |
| E00 | **Verified:** feature checkout and cached remote differ from live master; starting tree had only a checkpoint symlink untracked | `git status --short`, `git rev-parse HEAD`, `git log origin/master..HEAD`, remote URL; live GitHub commit API and [commit dbbeb84](https://github.com/wwk5q8z6kk-bit/nano-lm/commit/dbbeb84eaaaeaa128a1cc206f1adf02eb172678c). HEAD `61a06524c9d1a9998f64c90d1035831b575713eb`; cached base `d6ed212c32384c1d50089e50260918028af88bfd`; nine feature commits. Live master's parent is that cached base; only changed file removes unused `json` import in `scribe/build_scribe_data_v2.py` | High at observation time; no fetch/integration. Preserved `artifacts/native_checkpoints_fixed` points to `/Volumes/Express1M2/Offload/nano-lm/native_checkpoints`; target integrity not audited. Local findings must not be relabeled mainline behavior |
| E01 | **Verified:** broad faithful-intelligence mission, typed authority, historical 3.15M foundation | [PROJECT_AUTHORITY.md](../PROJECT_AUTHORITY.md):9–23,53–75; [PROJECT_CHARTER.md](../PROJECT_CHARTER.md):11–27,42–81; [CAPABILITY_LADDER.md](../CAPABILITY_LADDER.md):24–38 | High for intended purpose, not achievement. E1/E4 and old STOP states have scoped/historical meanings |
| E02 | **Verified:** P1 active; H7 consumed with invalid gold; separate-data semantic/state/conflict work next | [ACTIVE_NOW.json](../ACTIVE_NOW.json):4–16,29–52; [EXECUTION_PLAN.md](../EXECUTION_PLAN.md):28–37; [MODEL_RESEARCH_PROGRAM.md](../research/MODEL_RESEARCH_PROGRAM.md):27–43 | High. Model-research H7 row is stale relative to ACTIVE_NOW. Span-port/P4/CUAD marked cross-branch; other branches not inspected |
| E03 | **Verified implementation + execution:** transcript-to-checked-draft path connected, limited to five fields and review/abstain | [scribe.py](../../nanoscribe/scribe.py):30–90,93–132,135–180; [SCRIBING_PATH.md](../../nanoscribe/SCRIBING_PATH.md):16–39,50–129; T08–T10 below | High for source-tree execution. Unknown temporality and review-only disposition; no end-to-end P2–P9, audio/EHR/UI or automatic longitudinal persistence demonstrated |
| E04 | **Verified:** strong proposal/source/record separation and explicit failures; **inference:** promising independently reusable boundary | [contract.py](../../nano_ai/contract.py):30–52,55–75,93–97,197–228,305–319; [solver.py](../../nano_ai/solver.py):66–111,202–212; [adapt.py](../../nanoscribe/adapt.py):41–55; [check_note.py](../../nanoscribe/check_note.py):187–285; [encounter.py](../../nanoscribe/encounter.py):1–5 | High for invariants, moderate for reuse, low for adoption. Contract v0 permits ABSENT only for medication/allergy: 22 admissible field/state cells, although quota tables include all 25 combinations with zero for inadmissible cells. Source substring membership is not semantic entailment |
| E05 | **Verified:** optional compact route is local and bounded; semantic checks share assumptions | [compact.py](../../nanoscribe/compact.py):70–101,132–275; [context.py](../../nanoscribe/context.py):15–77; `scribe.py`:32 and `check_note.py`:198 both use `assess_fields`; [pilot report](../../artifacts/p1_pilot_v1/REPORT.md):38–56 | High. Lazy local loading with `trust_remote_code=False`; checkpoint directory alone does not verify full manifest. Finite Patient-source candidate trie and shared classifier are not independent semantic assurance |
| E06 | **Verified:** Core is executable research code but not the scribing runtime or durable memory | [runtime.py](../../nano/runtime.py):1–42,99–108,180–215; [pipeline.py](../../nano/pipeline.py):1–12; [retrieval.py](../../nano/retrieval.py):1–31; [contracts.py](../../nano/contracts.py):392–444; [kernel.py](../../nano/kernel.py):29–67,281–316; [architecture.py](../../nano/architecture.py):112–117; `scribe.py`:16–27 | High. In-memory mutable ledger, bounded questions, fixed capability costs and deterministic stand-ins. Identity primitives/metadata do not prove enforcement, durability, tamper resistance or production PHI controls. No comprehensive security audit |
| E07 | **Verified:** Wedge is classical local document/span retrieval | [wedge_v1/runtime.py](../../wedge_v1/runtime.py):1–5,21–28,41–71,132–214; [README](../../wedge_v1/README.md):5–15; T07 | High for implementation. Historical 8/8 dogfood is narrow, not evidence of general RAG/reasoning or current clinical benefit |
| E08 | **Verified:** infrastructure has useful evidence/benchmark contracts; limited task-world assumptions | [fabric/slice.py](../../fabric/slice.py):85–120 imports world-generator templates; [trajectory/README.md](../../trajectory/README.md):3–26; [benchmarks/README.md](../../benchmarks/README.md):3–20; [REGISTRY.yaml](../../benchmarks/REGISTRY.yaml):1–11; T01–T07 | High. Closed-world/reference checks and operational sentinels do not establish open-world semantics or benchmark superiority |
| E09 | **Verified retained result:** E1 KILL is scoped to the old closed task under frozen U | [results_e1_utility.json](../../trajectory/results_e1_utility.json):699–725; [EMPIRICAL_FOUNDATION.md](../../papers/EMPIRICAL_FOUNDATION.md):262–282 | High for stored result. M1 utility `0.9989993963`, official M0 `0.9252173640`, margin `+0.0737820324`; interpret using that experiment's utility and decision rule, not generic accuracy. Clean-clone E1 L/C reproduction is partial. No E1 rerun here |
| E10 | **Verified retained result:** E4 KILL applies to tested R-star regime | [results_e4_utility.json](../../trajectory/results_e4_utility.json):217–231,397–411,442–461; [EMPIRICAL_FOUNDATION.md](../../papers/EMPIRICAL_FOUNDATION.md):318–323 | High for stored result: 220 items/1,100 fields, best classical utility `0.6379553791`, Gref `-1.6232979175`, delta `-2.2612532966`. This contradicts speculative positive R-star narrative, not the possibility of every future Nano program. No rerun |
| E11 | **Verified retained results:** H6 rejected; H7 consumed but gold invalid | [H6 terminal](../../artifacts/nano_h6/H6_TERMINAL_DECISION.json):13–84; [H7 terminal](../../artifacts/nano_h7/H7_TERMINAL_DECISION.json):5–58; [H7 diagnosis](../research/H7_EVALUATOR_DIAGNOSIS.md):9–50 | High for artifacts/diagnosis. H6 3,901/5,000 overall, 2,277/2,987 held-value, absence 199/413, conflict 143/250, uncertain 190/250, missing 250/250; adaptive development, no successful downstream sealed claim. H7's 2 seeds consumed 4,017.465 seconds at recorded $0, but every arm's 5,000 gold fields were missing/empty instead of 250 missing/4,750 nonmissing. Recorded REJECT is not a valid comparative model-quality verdict. Neither evaluation can be rerun/relabelled |
| E12 | **Verified retained development result:** improved pilot remains exposed, hybrid and narrow | [pilot REPORT.md](../../artifacts/p1_pilot_v1/REPORT.md):3–16,21–56,73–104,122–144; [run-09 checked evaluation](../../artifacts/p1_pilot_v1/run-09/checked_evaluation.json):11–167; [annotation review](../../artifacts/p1_pilot_v1/annotation_review.json):2–26 | High for saved outputs, low for generalization. 20 dialogues = 10 paired worlds; quotas 66 supported/14 absent/12 missing/4 uncertain/4 conflict. Raw contract-valid 18/20, raw exact 85/100; checked exact 95/100, 78/78 presented fields correct. Final 14 review/6 abstain, 56 claims. All five checked errors are medication; 2/4 conflicts retained. Same-agent annotation review, not independent/human IAA. No clinical interpretation of these percentages |
| E13 | **Verified execution:** saved pilot proposals reproduce current drafts exactly; **inference:** reproducible code seam, not new quality evidence | [current_path_audit.json](../../artifacts/p1_pilot_v1/current_path_audit.json), [audit source](../../artifacts/p1_pilot_v1/audit_current_path.py); independent read-only T09 below | High for 20 exact equalities, 14 REVIEW/6 ABSTAIN and 56 claims. Six recorded code hashes and annotation/spec hashes checked against source by audit agent. No new model inference. Do not count replay as a new sample |
| E14 | **Verified retained comparisons:** historical compact/ORX/C2 answer different questions and cannot be pooled | [comparison REPORT.md](../../artifacts/p1_comparison/REPORT.md):9–37,41–72,76–89 | High for recorded numbers. Compact three-seed two-field synthetic min-arm state floors 100/83.33/76.67%; overlap-inclusive binding agreement 100/99.79/100%, not all exact/general P1. ORX: 24 fresh synthetic inputs, 384 repeated slots; prompt D grounded 17/240 present vs P1 3/240, correct absent 21/144 vs 24/144. Corrected C2: 128 cases; state 109/149,118/150,129/146 and exact 13/149,13/150,13/146 across arms. Heterogeneous denominators; corrected student delta -5.51pp supersedes invalid -32pp. Adaptation remains locked |
| E15 | **Verified:** annotation transport and controls improved; stale narrative still exists | [P1_ANNOTATION_TRANSPORT.md](../research/P1_ANNOTATION_TRANSPORT.md):5–27,35–38,68–70; [comparison REPORT.md](../../artifacts/p1_comparison/REPORT.md):126–128; [pilot report](../../artifacts/p1_pilot_v1/REPORT.md):21–34; [ACTIVE_NOW.md](../ACTIVE_NOW.md):68–81 | High. Pilot oracle 100%, all-missing 12%, but neither proves semantic annotation correctness. Old 2-review/4-abstain/14-failure descriptions superseded by 14/6/0. H6 terminal's calibration headline is stronger than its unassessed-cause qualification; recomputed rows separate 241 state-only, 771 binding-only and 87 joint errors. No causal calibration conclusion warranted |
| E16 | **Verified configuration/environment:** tests run but default/CI coverage differs; fresh install unverified | [pytest.ini](../../pytest.ini):5–7; [pyproject.toml](../../pyproject.toml):1–15; [ci.yml](../../.github/workflows/ci.yml):16–56; [requirements.txt](../../requirements.txt); [requirements-bench-test.txt](../../requirements-bench-test.txt):4–8; execution section | High for observed setup. `pytest.ini` takes precedence and defaults to fabric/trajectory. Full nano/nano_ai omitted from CI; some learned tests included. Bench-test pytest pin 9.0.2 vs main 9.0.3; installed Python/ML versions differ from CI/pins. No claim that a clean install fails |
| E17 | **Verified observation:** advertised MIT license text not found in tracked repository | [README.md](../../README.md):133–135 links `LICENSE`; file absent; `git ls-files '*LICENSE*' '*COPYING*'` returned no paths | High for this snapshot; redistribution terms unresolved. Not a legal conclusion; dependencies/model/data licenses separate |
| E18 | **Inference:** code is a useful research source tree, not a validated distributable SDK or clinical service | E03–E08, E16–E17; pyproject empty `py-modules`, old measurement description, dependencies elsewhere; no inspected installed entry point | Moderate. Running source-tree imports worked; absence of install verification is not proof packaging is broken. No maintained service/API or production security certification established |
| E19 | **Verified governing requirement:** P1 exit is still external/human/owner gated | [medical SCRIBING.md](../domains/medical/SCRIBING.md):38–46; [PROJECT_CHARTER.md](../PROJECT_CHARTER.md):59–68; ACTIVE_NOW.json:13–16 | High. External held-out medical evaluation, blinded human usefulness/safety assessment and owner-signed exit remain necessary. Synthetic development success and this report cannot substitute |

### Preserved terminal identities

Read-only SHA-256 checks were performed on these terminal artifacts:

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/nano_h6/H6_TERMINAL_DECISION.json` | `297a7b7e914bd9be938479a726a8e690c56f1946a62bd02e6806ed243d3471e7` |
| `artifacts/nano_h7/H7_TERMINAL_DECISION.json` | `14cfb96158c30d117ed635a765f1bc2d2c1d6969100dfb6f38b40485d81fad60` |

These identify what was read, not a new scientific endorsement. Both files also match `git show HEAD:<path>` byte-for-byte. No frozen artifact, tag or terminal decision was changed. The separately checked evaluation files pinned by these decisions have hashes `cb612410bd9530062147c1ed36bdcd606f9292a77b750376baf7a9860639e11a` (H6 `kaggle/eval-20260805/results/development_evaluation.json`) and `9d1b8035b87b8af9419ead92a0dffd66806df1e77720abdf6a1d1deb98951d49` (H7 `kaggle/h7-boundary-margin-v1/h7_screen_evaluation.json`); those are evaluation-file identities, not terminal-JSON hashes.

## Observed execution

All commands ran from `/Users/mac/Projects/nano-lm` in the existing environment. Python was **3.14.7**, executable `/opt/homebrew/opt/python@3.14/bin/python3.14`; observed packages included Torch **2.11.0**, Transformers **5.5.3**, pytest **9.0.2**, NumPy **2.5.0**, datasets **4.8.4**. CI uses Python 3.11; the pinned requirements were not freshly installed.

For the pytest groups and replay, the environment prefix was:

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

The commands below show the test invocation after that prefix unless explicitly stated. `-p no:cacheprovider` suppressed pytest cache writes. Captured logs were temporary files under `/tmp`, not repository deliverables. For T04–T06 the existing pytest `addopts=-q` and explicit `-q` suppressed a summary; counts were checked from progress markers and collection, not inferred merely from exit status.

| ID | Exact invocation | Observed outcome and interpretation |
| --- | --- | --- |
| T01 | `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_active_now.py` and `PYTHONDONTWRITEBYTECODE=1 python3 scripts/check_docs_integrity.py` | Exit 0: `ACTIVE_NOW_OK`, `DOCS_INTEGRITY_OK`. Configuration/document consistency, not empirical correctness |
| T02 | `PYTHONDONTWRITEBYTECODE=1 python3 fabric/test_fabric.py` | 8/8 pass. Also covered within T07; not additional unique test count |
| T03 | `PYTHONDONTWRITEBYTECODE=1 python3 trajectory/test_recompute_c3.py` | 7/7 pass. Also covered within T07; not additional unique test count |
| T04 | `python3 -m pytest -p no:cacheprovider -q nanoscribe nano scripts/test_reconcile_p1_comparison.py scripts/test_run_h7_kaggle.py scripts/test_run_p1_pilot.py > /tmp/nano-discovery-scribe-nano.log 2>&1` | Exit 0; **583 passed**, no skips. No new campaign/model training or experiment run |
| T05 | `python3 -m pytest -p no:cacheprovider -q nano_ai/tests > /tmp/nano-discovery-learned.log 2>&1` | Exit 0; **499 passed, 1 skipped**; 500 collected |
| T06 | `python3 -m pytest -p no:cacheprovider -q benchmarks/ > /tmp/nano-discovery-benchmarks.log 2>&1` | Exit 0; **500 passed, 1 skipped**; 501 collected. Tests validate instruments, not external benchmark capability |
| T07 | `python3 -m pytest -p no:cacheprovider -o addopts='' -q fabric trajectory wedge_v1 > /tmp/nano-discovery-foundations.log 2>&1` | **32 passed in 0.43s**, exit 0. T04–T07 total **1,614 passed, 2 skipped** without adding overlapping direct checks |
| T08 | `PYTHONDONTWRITEBYTECODE=1 python3 nanoscribe/benchmark.py` | Four fixture cases: exact_gold_span 4/4, correct state 4/4, coverage/character F1 1.0, zero critical/state errors, three direct/one review_required. **Fixture/software smoke, no model weights**, not clinical accuracy |
| T09 | Read-only Python replay shown below | 20/20 full drafts equal stored run-09 drafts, 14 review and six abstain, 56 supported/absent claims. No new inference; outputs not rewritten |
| T10 | `python3 -m nanoscribe.scribe --transcript nanoscribe/examples/synthetic_encounter.txt --format text` | Exit 0, review draft with headache, 3 days, mild, acetaminophen and explicit “No allergies.” evidence links. Reference solver, not live pretrained-model verification |
| T11 | `python3 -m pytest -p no:cacheprovider -o addopts='' -q -rs nano_ai/tests/test_admit_evidence_query_h6.py::test_installed_guard_builds_a_spec_accepted_by_admission benchmarks/adapters/lm_eval/tests/test_program0.py::test_lm_eval_validate_if_available` | **2 skipped in 0.29s**. Installed Nano RunPod guard unavailable (`test_admit_evidence_query_h6.py`:577); `lm_eval` import unavailable (`test_program0.py`:169). Repeated skips clarify T05/T06, not extra collected tests |

T05/T06 skip identities were additionally resolved with `python3 -m pytest -p no:cacheprovider -o addopts='' --collect-only -q nano_ai/tests` and the analogous `benchmarks/` collection; these collected 500 and 501 respectively and did not execute another experiment.

### T09 read-only reproduction

Executed as `python3 - <<'PY'` with the offline/no-bytecode prefix above. The repository's audit script was inspected, but not invoked because it writes an artifact. This variant performs assertions and prints counts only:

```python
import collections
import json
from pathlib import Path

from nanoscribe.compact import compact_solver
from nanoscribe.scribe import scribe

base = Path("artifacts/p1_pilot_v1")
rows = json.loads((base / "dialogues.json").read_text())
saved = json.loads((base / "run-09/encounters.json").read_text())
provenance = json.loads((base / "run-09/provenance.json").read_text())
lookup = {
    row["transcript"]: rec["generation"]["raw"]
    for row, rec in zip(rows, saved, strict=True)
}
solver = compact_solver(
    lookup.__getitem__,
    version=provenance["checkpoint_revision"],
    binder="context",
)
counts = collections.Counter()
claims = 0
for row, rec in zip(rows, saved, strict=True):
    draft = scribe(
        row["transcript"],
        encounter_id=row["example_id"],
        solver=solver,
        context_policy="context",
    )
    assert draft == rec["draft"], row["example_id"]
    counts[(draft["status"], draft["disposition"])] += 1
    claims += sum(
        claim["state"] in ("supported", "absent")
        for claim in (draft.get("note") or {}).get("claims", [])
    )
print(len(rows), counts, claims)
```

### Not performed

No fresh dependency installation, deployment, new pretrained-model inference, H6/H7 rerun, model download, paid compute, dataset ingestion, PHI/private-data upload, clinician study, user interview, formal security audit, cross-branch migration or checkpoint/archive integrity assessment was performed. Unit tests exercised tiny model/code fixtures; that is not a new campaign training result. Preserved pilot latency (316.130 seconds for 20 cases including loading/pipeline, with concurrent local work) is not a controlled hardware benchmark or deployment SLO.

## External evidence

All sources below were accessed **2026-10-22 EDT**. These are primary sources. Vendor capabilities/prices are verified as **published claims**, not independently observed product quality. Code-license descriptions are repository observations, not legal advice or an audit of all bundled assets. URLs and offerings may change; recheck before procurement or ingestion.

| ID | Source and supported facts | Classification, confidence and limits |
| --- | --- | --- |
| X01 | [Google product page](https://workspace.google.com/products/gemini-notebook/) (NotebookLM URL redirects here), [NotebookLM plans](https://support.google.com/notebooklm/answer/16213268?hl=en): source-grounded questions/summaries/references; Standard listed 50 sources per notebook | Published capabilities, high confidence in overlap with generic assistant concept; no independent grounding-quality measurement or demand evidence |
| X02 | [Docling repository](https://github.com/docling-project/docling), [project site](https://docling.ai/): multi-format document conversion, layout/OCR/table handling, local execution; MIT code | Verified documentation/license observation. Parsing makes source structure available; it does not certify semantic claims. Model/data licenses separate |
| X03 | [Unstructured repository](https://github.com/Unstructured-IO/unstructured), [pricing](https://unstructured.io/pricing): Apache-2.0 OSS partitioning; separate commercial platform, initial 10,000 free pages then $0.015/page advertised | Published offering/price, time-sensitive. OSS and platform features/costs must not be conflated; no service used |
| X04 | [LlamaExtract configuration](https://developers.llamaindex.ai/llamaparse/extract/guides/configuring-extract/), [developer pricing](https://developers.llamaindex.ai/llamaparse/general/pricing/), [plans](https://www.llamaindex.ai/pricing): schema extraction, mode-dependent citations/confidence; spreadsheet extraction lacks citations/confidence, Turbo citations are text-level. Listed default combined parsing-plus-extraction totals 8/25/60 credits/page and $1.25/1,000-credit reference rate | Published capabilities/prices. Derived reference costs $0.010/$0.03125/$0.075/page; text/preparsed inputs can use extraction-only rates. Plan/add-on terms and tiers require recheck. No validation of medical state fidelity or extraction confidence calibration |
| X05 | [Haystack evaluation docs](https://docs.haystack.deepset.ai/docs/evaluation), [repository](https://github.com/deepset-ai/haystack): pipeline evaluation including statistical/model-based approaches and integrations; Apache-2.0 | Verified documentation. Supports avoiding redundant generic orchestration work; no claim generic metrics suffice for P1 |
| X06 | [ContextualAI extract-bench](https://github.com/ContextualAI/extract-bench): 35 human-validated PDF/JSON tasks, five schemas/four domains; multiple matching/judging methods; MIT code | Author-reported dataset/method facts. Nonclinical external reference; semantic judges and data licensing need task-specific review |
| X07 | [run-llama ExtractBench](https://github.com/run-llama/ExtractBench), [license](https://github.com/run-llama/ExtractBench/blob/main/LICENSE): distinct project, 370 documents/4,869 pages, extraction/completeness/grounding scoring; Apache-2.0 code | Author-reported coverage, not independent vendor leaderboard validation. Dataset rights separate; do not confuse with X06 or casually run a costly full benchmark |
| X08 | [vLLM](https://github.com/vllm-project/vllm): distributed serving, structured output and adapter features; Apache-2.0 | Published implementation features. Supports “integrate serving” judgment; no local deployment, capacity or latency verified |
| X09 | [llama.cpp](https://github.com/ggml-org/llama.cpp): local CPU/GPU and Apple Silicon inference, quantization; MIT | Published implementation features. Hardware fit/model compatibility remain per-workload questions |
| X10 | [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat): nanoGPT marked deprecated in favor of nanochat; existing model training/inference/evaluation stack; MIT code | Verified repository status at access time. Author cost examples are not a budget for nano-lm's data/task/hardware; training infrastructure alone is not differentiation |
| X11 | [Heidi pricing/features](https://support.heidihealth.com/en/articles/8885030-heidi-s-pricing-plans-cost-and-features): free unlimited transcription/notes with standard templates; paid workflow features; cited research queries | Published offering, not independent accuracy. Literature citations do not establish transcript claim entailment. No evidence here about recurring customer complaints or suitability for Nano |
| X12 | [Abridge contextual reasoning engine](https://www.abridge.com/abridge-contextual-reasoning-engine): documentation describes encounter/history/context integration, linked source evidence and clinician-reviewed workflow outputs | Vendor claim, not independently reproduced quality, price or clinical outcome. Establishes feature competition only |
| X13 | [ACI-Bench author repository](https://github.com/wyim/aci-bench), [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2024/10/s41597-023-02487-3.pdf): role-played dialogue/note corpus; 207 encounters; author repository states CC BY 4.0 and known swapped speaker tags. Paper describes synthetic/imaginary context and review of unsupported note content | Primary research/data documentation, high confidence in relevance as a candidate. No data downloaded. Requires pinned transcript variant, provenance/license review and new field/state/span annotations; paired note is not input-source evidence. Selected data still must meet the current no-PHI scope |

### Research limitations and alternative explanations

- Available alternatives demonstrate that broad reading, retrieval, extraction and citation features are already supplied. They do not prove those systems outperform Nano on its five-state task; that comparison remains proposed.
- No first-hand user study or independently corroborated recurring complaint analysis was completed. Vendor pages are not evidence of unmet demand, adoption, willingness to pay, or clinical reliability.
- The external review sampled relevant systems rather than exhaustively benchmarking the market. No commercial service trial or live hosted-model comparison was run.
- Differentiation could lie in auditability, local operation or epistemic-state precision, but competitors also offer provenance/local tools. Only task-specific benefit can establish meaningful advantage.

## Decision inferences, estimates and hypotheses

| ID | Conclusion | Type/support | Confidence and what could reverse it |
| --- | --- | --- | --- |
| D01 | Bounded repair is the best next use of effort if a short independent trial is affordable | **Inference**, E02–E08, E12–E18, T01–T11; existing seam and concrete failure targets make a reversible test possible | Moderate. No reviewers, no user task, or an equally useful simpler baseline favors archival/adoption instead |
| D02 | Broad transformation or a new runtime/model stack is premature | **Inference**, E03/E06/E12, X01–X12; much new scope, established tools, absent demand/quality advantage | Moderate-high for current evidence. A narrow demonstrated unmet need and successful baseline trial could reverse it |
| D03 | A full rewrite is not justified by an observed architectural blocker | **Inference**, replaceable solver seam E03/E04 versus unintegrated Core E06 | Moderate. New requirements may reveal coupling not covered by this audit; prove the blocker before rebuilding |
| D04 | The combined checker can improve useful faithfulness per dollar/minute | **Hypothesis**, exposed pilot supports plausibility but not generalization or reviewer benefit | Low. Matched-coverage independent baseline study can disprove it |
| D05 | Researchers/developers will adopt a bounded workflow or reusable verifier | **Hypothesis**, no adoption study performed | Low. Five-user screen is directional; durable demand requires repeated real use |
| D06 | Repair decision takes roughly 1–2 engineering weeks; bounded evolve 3–6 further weeks; scoped rebuild 4–8 weeks; broad product months | **Estimate**, one familiar engineer, existing tools, reviewers available; scope table in report §7 | Low-moderate. Reviewer recruitment/annotation, installation, legal/data questions and unknown integration work may dominate; not commitments |
| D07 | Proposed 100-case, five-user, cost/coverage/latency thresholds can make the next decision falsifiable | **Proposed design**, report §§13–14; not current experimental authority or measured capability | Moderate as screening design, low as statistical assurance. Finalize prospectively; small/stratified cohorts cannot establish clinical prevalence or safety |
| D08 | Preserve/archive is preferable if the benefit premise fails | **Decision rule/inference**, opportunity cost and X01–X12; preserve E09–E15 regardless | Moderate. Positive independent quality and actual user benefit support evolution instead; sunk cost is not a reason to continue |

## Contradictions resolved without rewriting evidence

| Apparent contradiction | Resolution used in the report |
| --- | --- |
| Project has P1–P9 types, so P1–P9 must work | Code path and capability ladder distinguish interfaces/research mechanisms from delivered application behavior |
| E1/E4 KILL means stop the entire project | Preserve the negative scoped verdicts; current charter/authority does not make them a program-wide stop |
| H7 terminal says REJECT, so it proves model quality | Gold transport was invalid; preserve the consumed result and diagnosis without drawing a valid comparative quality conclusion |
| 95/100 checked fields means 95% note completeness | Checked representation and final note are different denominators: 78 presented fields versus 56 final claims, 14 review/6 abstain |
| Zero false presented claims proves safety | Exposed small paired examples, shared semantic checks and no independent clinical labels cannot establish population safety |
| Passing tests means installable/released/clinically ready | Tests ran in an existing environment with two skips; packaging, fresh install, generalization and clinical evaluation remain separate |
| A cited product feature proves a market gap or unmet need | It establishes available competition; demand and comparative benefit remain hypotheses |
| Local branch work represents current master | Remote master was separately verified; feature commits remain distinct until normal integration |

## Report-only validation

After report creation, active-state and documentation-integrity checks returned `ACTIVE_NOW_OK` and `DOCS_INTEGRITY_OK`. All 80 local Markdown file links resolved. Decision-matrix totals and sensitivity scores were independently calculated from the documented vectors. Git showed only the two new discovery reports and the pre-existing untracked checkpoint symlink; no tracked source/evidence changes. Terminal JSON bytes matched HEAD. These are report/integrity checks, not new scientific evaluations.
