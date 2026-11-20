# nano-lm strategic disposition report

Investigation date: **2026-10-22, America/New_York**. This is a decision report, not a replacement charter, execution plan, experiment authorization, or P1 exit decision. Only this report and its [evidence register](evidence-register.md) were created. Application code, evidence, checkpoints, Git references, and existing user work were preserved.

The two supplied requests differed on implementation. After the owner delegated the scope choice, this investigation followed the request for strategic analysis and two reports. Proposed work below has **not** been implemented.

## 1. Executive decision summary

**Repair and stabilize the existing bounded P1 path, with a short, falsifiable continuation test before further expansion.** The evidence supports continuing a focused research and developer workflow; it does not yet support launching a clinical product, building a general information assistant, or funding a new model/runtime platform.

The asset worth testing is a connected pipeline that turns a transcript into an evidence-linked encounter and a checked review draft. Its explicit missing/absent/uncertain/conflicting states, source binding, observable failures, and conservative rendering are useful. Its semantic generality and user benefit remain unproved. This distinction makes a limited repair preferable to either a rewrite or an unconditional roadmap commitment. [E03–E08, E12–E15](evidence-register.md#repository-evidence)

The most favorable preserved development result is 95/100 checked fields on 20 exposed, paired synthetic dialogues. That is not 95% note completeness: the final path produced 14 review drafts, six abstentions, and 56 supported/absent claims. Medication-state and conflict failures remain. Replaying saved proposals reproduced all 20 drafts in this investigation; it did not test new model generations or independent data. Broad local verification passed **1,614 tests, with two explicitly skipped dependencies**. These are strong software/reproducibility observations, not clinical efficacy. [E12–E14, T01–T11](evidence-register.md#observed-execution)

**Confidence:** moderate in the repair recommendation; high that the current repository does not justify broad product or clinical claims; low in any forecast of demand or commercial differentiation. There were no user interviews, customer commitments, independent clinician evaluations, fresh model comparisons, or clean-environment installation trials.

**Fallback:** preserve the research record and stop active product development if independent testing finds no advantage over simpler baselines, or if intended users do not benefit. Extract a small verifier/library only if a real downstream use is demonstrated. Starting an unrelated product is a separate decision, not an automatic consequence of archiving this one.

## 2. Verified description of nano-lm today

The project began as a small from-scratch language-model and closed-world measurement effort; the protected 3.15M model work and E1/E4 results remain its empirical foundation. Its current charter seeks useful, reliable intelligence across evidence capture, compression, temporal state, reasoning, and action. It does not define success by a particular parameter count. NanoScribe is the medical proving application, not the whole program, and Qwen is a compact control/adapter path rather than Nano's identity. [E01, E09–E11](evidence-register.md#repository-evidence)

The authoritative ladder is P1 scribing → P2 summarization → P3 longitudinal charting → P4 synthesis → P5 questioning → P6 reasoning → P7 planning → P8 tools/action → P9 adaptation. P1 is the current frontier; P2/P3 have specifications/interfaces; P4–P9 are architectural requirements. Here, **P3 charting means longitudinal records**, not plotting charts. Interface names and capability catalogs are not demonstrations of these capabilities. [E02, E06](evidence-register.md#repository-evidence)

The inspected checkout was `codex/p1-h7-scribing-path` at `61a06524c9d1a9998f64c90d1035831b575713eb`. It had nine commits beyond cached `origin/master` at `d6ed212c32384c1d50089e50260918028af88bfd`. Live GitHub master was separately verified at `dbbeb84eaaaeaa128a1cc206f1adf02eb172678c`; that one newer mainline commit removes an unused import in `scribe/build_scribe_data_v2.py`. No fetch, checkout, merge, commit, or push was performed. **The current feature-branch P1 results must not be described as already integrated on live master.** [E00](evidence-register.md#repository-evidence)

The initial untracked `artifacts/native_checkpoints_fixed` symlink was left untouched. The working tree was otherwise clean before these reports. Inspection covered the active source paths, canonical docs, configuration, tests, representative historical artifacts, and relevant history. It was not an exhaustive line-by-line audit of all 1,151 tracked files or of unintegrated branches.

## 3. Current capabilities and technical condition

### The actual executing path

```text
UTF-8 transcript
  → NanoInput → NanoSolver.infer
  → typed five-field output with evidence/state
  → encounter representation
  → deterministic note rendering
  → original-source/record/note checks
  → REVIEW draft, ABSTAIN, or explicit inference failure
```

The connected fields are chief complaint, duration, severity, medication, and allergy. Contract v0 permits explicit `absent` only for medication/allergy; the other three fields support `supported`, `missing`, `uncertain` and `conflicting`. The bridge preserves evidence and unresolved information but assigns unknown temporality and requires review. It neither captures audio nor integrates a clinician interface, EHR, durable patient store, or autonomous actions. The JSON output can be saved by a caller; that is not an automatic longitudinal persistence system. [E03–E05](evidence-register.md#repository-evidence)

The reference path is deterministic. The optional compact path uses locally available Hugging Face/PyTorch model files; the preserved pilot used base Qwen2.5-1.5B-Instruct, not the historical trained adapter or Native Nano. Constrained decoding offers finite exact Patient-source spans through a token trie. Bounded question/reply grammars help assign context and reject unsupported states. The decoder, binder, and note checker share a context classifier: their agreement is **not independent semantic verification**. Exact quotation alone cannot establish the correct subject, time, field, denial, or resolution of a conflict. [E04–E05, E12](evidence-register.md#repository-evidence)

### Capability and condition table

| Area | What exists | Practical limit |
| --- | --- | --- |
| Faithful extraction/rendering | Working CLI and solver seam; source spans; five states; deterministic checked draft | Narrow vocabulary/fields; omission and conflict handling remain central problems |
| Learned experiments | Implemented contracts, adapters, training/evaluation machinery and preserved results | H6 rejected; H7 scoring instrument invalid; neither establishes a successful general learned scribe |
| Nano Core | Executable symbolic state, ledger, WorkSlice, bounded questions and verification | Scribe imports its disposition enum, not the complete runtime; ledger is in memory |
| Wedge | Local lexical document/span search | Useful supporting retrieval; no general comprehension or LM reasoning |
| P2–P9 | Types, interfaces and some separate synthetic research mechanisms | No demonstrated end-to-end application progression beyond P1 |
| Evaluation infrastructure | Many contract, negative-case, lineage and benchmark tests | Passing instrument tests is not winning an external benchmark |
| Distribution | Source-tree execution worked in the existing environment | Fresh install, packaged SDK and supported runtime matrix remain unvalidated |
| Privacy/security | Local inference option; source identity and provenance primitives | No demonstrated production authentication, retention, durable isolation, or PHI readiness |

Evidence: [E03–E08, E16–E18](evidence-register.md#repository-evidence).

### Verification and reproducibility

The full explicit offline suite set passed: 583 NanoScribe/Core/script tests, 499 learned-path tests, 500 benchmark tests, and 32 foundation tests; the learned and benchmark groups each skipped one unavailable optional dependency. Fabric/C3 direct regressions also passed, but overlap the foundation count. Documentation checks passed. The reference CLI and fixture benchmark ran successfully. Exact commands, environment and skip reasons are in [T01–T11](evidence-register.md#observed-execution).

These observations establish execution in the **existing** Python 3.14.7 environment. Installed Torch 2.11.0 and Transformers 5.5.3 differ from requirements pins 2.13.0 and 5.10.1; CI specifies Python 3.11. No clean installation, performance benchmark, deployment or new live-model evaluation was attempted. Bare `pytest` selects only `fabric` and `trajectory`; CI does not run the complete `nano` and `nano_ai` suites. README claims MIT licensing, but a tracked license text was not found. These are maintenance/release uncertainties, not proof that installation is impossible or a legal determination. [E16–E18](evidence-register.md#repository-evidence)

## 4. Existing assets and independent value

The highest-value asset is the combined solver/contract/evidence/render/check seam. It can compare learned and deterministic proposals without trusting model-invented offsets or collapsing inference failure into a valid abstention. It could remain useful even if the broader intelligence program stops. Whether anyone outside this project needs it is a hypothesis requiring a small adoption test.

Other retained value includes the explicit epistemic-state taxonomy; exact-source regression cases; annotated transport and negative controls; source/proposal/output lineage; and preserved unsuccessful experiments. The H7 failure demonstrates why evaluator schema and gold transport require validation before compute is spent. The pilot demonstrates why syntax constraints do not solve semantic-state selection. Those lessons survive a decision to discard implementation. [E04, E09–E15](evidence-register.md#repository-evidence)

The broad architecture, dataclasses, content IDs, token tries, model wrappers and generic benchmark registries are replaceable engineering choices, not demonstrated novelty. The in-memory ledger is not durable, tamper-proof storage. The large quantity of experimental artifacts is valuable for reproducibility but creates review and maintenance overhead. Historical effort and file count are not arguments for continued investment. [E06–E08](evidence-register.md#repository-evidence)

## 5. Research into plausible future directions

External sources were accessed **2026-10-22 EDT**. They are project documentation, vendor documentation, repositories, and a research paper; advertised capabilities are not independent accuracy measurements. No recurring user complaint pattern, willingness to pay, or market gap was independently established. Detailed sources and limitations are in [X01–X13](evidence-register.md#external-evidence).

| Existing alternative | What a user can already obtain | Implication for nano-lm |
| --- | --- | --- |
| [Google NotebookLM / Gemini Notebook](https://workspace.google.com/products/gemini-notebook/) | Source-grounded questions, summaries and source references | A general “read documents and answer questions” proposition has substantial overlap; citations alone do not differentiate |
| [Docling](https://github.com/docling-project/docling), [Unstructured](https://github.com/Unstructured-IO/unstructured) | Established document conversion/partitioning, including complex formats | Integrate parsing if needed; building OCR/layout ingestion is a poor first investment |
| [LlamaExtract](https://developers.llamaindex.ai/llamaparse/extract/guides/configuring-extract/) | Schema-based extraction with configurable citations/confidence; capabilities vary by mode | Exact state/conflict behavior needs a comparison; structured extraction and provenance are already available |
| [Haystack](https://docs.haystack.deepset.ai/docs/evaluation) | Pipeline orchestration and component/system evaluation | Generic orchestration/evaluation frameworks are not an attractive independent runtime niche |
| [ContextualAI Extract Bench](https://github.com/ContextualAI/extract-bench) and [run-llama ExtractBench](https://github.com/run-llama/ExtractBench) | Two distinct public extraction evaluation projects | Reuse task/evaluation ideas; do not confuse their datasets, licenses, or metrics with each other or with P1 |
| [vLLM](https://github.com/vllm-project/vllm), [llama.cpp](https://github.com/ggml-org/llama.cpp) | Established GPU/distributed and local/Apple Silicon inference respectively | A new inference engine has little demonstrated rationale here |
| [nanoGPT](https://github.com/karpathy/nanoGPT), [nanochat](https://github.com/karpathy/nanochat) | Training/experimentation reference implementations; nanoGPT now directs users to nanochat | Small-model research needs a specific scientific question, not a generic training-stack deliverable |
| [Heidi](https://support.heidihealth.com/en/articles/8885030-heidi-s-pricing-plans-cost-and-features), [Abridge](https://www.abridge.com/abridge-contextual-reasoning-engine) | Clinical documentation offerings, with different workflow/context/source-link features | Five-field note generation alone is weak differentiation; vendor pages establish offerings, not comparative safety |

Docling's code is MIT, Unstructured and Haystack are Apache-2.0, vLLM is Apache-2.0, and llama.cpp is MIT according to their repositories. Model weights, data, service agreements and deployment obligations remain separate checks. Local parsing/inference can avoid transmitting inputs, but does not by itself supply access control or secure storage. Hosted services trade infrastructure work for usage fees and data-handling constraints. [X02–X05, X08–X09](evidence-register.md#external-evidence)

Price observations illustrate why API-first comparisons can be cheaper than owning model infrastructure; they are not procurement estimates. Unstructured listed an initial 10,000 free pages and then $0.015/page. LlamaIndex documented default combined parsing-plus-extraction totals of 8/25/60 credits per page and a $1.25/1,000-credit reference rate, approximately $0.010/$0.03125/$0.075 per page before plan-specific terms; text/preparsed inputs can use extraction-only rates. Verify current tiers, included processing and limits before any purchase. No paid service was used. [X03–X04](evidence-register.md#external-evidence)

## 6. Comparison of competing product concepts

All demand and differentiation claims in this section are **hypotheses**. “Essential” describes the minimum coherent workflow if selected; it does not add requirements to the current plan. Effort estimates assume one engineer familiar with the code, available reviewers, and existing pretrained tooling. They exclude clinical validation, customer procurement and unknown data-acquisition delays.

| Concept, user and important problem | Essential workflow and output; optional scope | Reuse versus new work | Differentiation test and decision |
| --- | --- | --- | --- |
| **A. Bounded P1 research/developer workflow**: builders evaluating faithful extraction need to find unsupported assertions and omissions, not just parse JSON | Transcript → interchangeable proposal → state/source checks → review draft plus errors and coverage. Essential: reproducibility, explicit failure, independently scored source evidence. Optional: API packaging and extra fields | High reuse of current solver, contracts, checkers and tests. Repair semantics and evaluation independence; pretrained models supply proposals | Compare against deterministic and prompted-model baselines at matched coverage and reviewer effort. Best immediate fit; roughly 1–2 engineering weeks for a decision, not a clinical release |
| **B. Narrow document/evidence comparison**: researchers checking two versions or competing claims need an auditable difference table | Parse → identify claims → align sources → show agreement, contradiction, missing support and a qualified brief. Essential: provenance and source-aware comparison. Optional: broad QA, diagrams | Medium reuse of evidence/state concepts; medical schema and context grammar need replacement. Use Docling plus an existing model; build document alignment and numeric/entity checks | Must catch consequential changes better or faster than NotebookLM and schema extraction. Test 10 public/synthetic pairs manually before code. Several weeks for a bounded prototype; demand unknown |
| **C. General information-understanding assistant**: analysts/learners want documents, explanations, summaries, QA, extraction and charts in one place | Ingest → structured source map → task-specific retrieval/model reasoning → checked summary/table/chart → correction. Optional tools/actions | Low-to-medium conceptual reuse, low direct product reuse. Needs parsing, retrieval, multimodal layout, broad schemas, UI, sessions, permissions, persistence and chart/data processing | Competes with established assistants and extraction stacks. Only coherent around one repeated decision workflow; feature aggregation is insufficient. Multiple months and ongoing service maintenance; reject now |
| **D. Small model/training research**: researchers testing a particular mechanism need causal, reproducible comparisons | Freeze one falsifiable hypothesis and dataset, train/control models, measure quality-cost tradeoff. Optional distillation only after a teacher baseline | Reuse lineage, contracts and experiment controls; replace generic training infrastructure with established tooling where sensible | Existing negative findings are useful, but no evidence new training beats pretrained proposals. A mechanism experiment may take days/weeks; product-quality model development is unbounded. Retain as conditional research, not the lead product |
| **E. Inference/model experimentation toolkit**: developers need efficient, reproducible model serving and evaluation | Model adapter → existing runtime → batch evaluation and artifacts. Essential adapters/provenance; optional hosting | Reuse solver/evaluator seam; serving, quantization and scheduling already exist in vLLM/llama.cpp/Haystack | Generic runtime differentiation is weak. A small adapter extraction may take days; competing runtime months+. Prefer contributing/integrating over a standalone platform |
| **F. Clinical documentation product**: clinicians want less review/documentation burden without losing clinical meaning | Audio/transcript → useful note → source review → chart workflow. Essential domain coverage, human safety evaluation, deployment/privacy controls. Optional longitudinal support only after P1 exit | Reuse P1 seam, but most clinical workflow and operational controls are new; Heidi/Abridge already serve the broad workflow | Requires independent clinician benefit/safety and external held-out evaluation, not synthetic score gains. Many months plus governance/data access; defer rather than advertise current clinical readiness |

### Does an integrated information workflow make sense?

Yes, **within a defined decision**, such as comparing two policy versions and explaining changed obligations. Reading, extraction, comparison and a qualified summary then share one source model and one output the user can verify. A universal assistant spanning arbitrary datasets, medical records and planning has no demonstrated coherent user need here.

If such a direction later passes a narrow demand test, models should propose semantic interpretations and candidate claims; retrieval should select relevant passages without certifying them; deterministic code should bind source IDs, enforce schemas, compute units/aggregates and validate citations; a structured data engine should calculate table values; standard plotting libraries should render verified data. A chart must retain its data/filter/aggregation/unit lineage. Fluency, a rendered figure, or a citation is not evidence of comprehension or numerical correctness.

The present P1 path lacks document-layout interpretation, broad contextual semantics and quantitative/chart evaluation. A narrow transcript workflow requires fewer independent failure-prone components and is cheaper to falsify. This is why concept A leads, while B is a possible small comparison experiment and C is not an implementation recommendation. [E03–E08; X01–X09](evidence-register.md)

## 7. Disposition analysis

The following estimates concern engineering effort to a **useful testable result**, not a production or clinical release. No funding is authorized by this report.

| Outcome | Benefit, assets and first result | Costs, risks and replacements | Prefer when / reject when |
| --- | --- | --- | --- |
| **Keep substantially as is** | Preserve all research, CLI and experiments; document a frozen research snapshot in 1–2 days | Known semantic failures, runtime/license ambiguity and limited CI remain; no new user value; low operating cost but dependency drift | Prefer if the objective is archival reference or teaching. Reject as an active product strategy if users need dependable broader coverage |
| **Repair/stabilize** | Keep current seams and frozen evidence; repair release reproducibility and state/conflict handling against separate data. First decision in about 1–2 engineering weeks plus reviewers | Risk of polishing a narrow grammar without transferable benefit. Moderate test/data maintenance; no broad infra required | Prefer if independent checks can cheaply test incremental value. Reject if no independent reviewers/data or simpler baselines match benefit |
| **Incrementally evolve** | Extend useful P1 coverage after repair; later connect Core only where a measured need exists. Preserve most P1 assets | Approximately 3–6 additional weeks for one bounded nonclinical extension after a passing gate; semantic scope, labels and regression costs grow. No automatic P2/P3 advancement | Prefer after demand and independent quality gains. Reject while exposed fixtures are the only positive evidence |
| **Transform** | Could target document comparison, research assistance or analysis using evidence principles | Months for a broad product, new ingestion/UI/data/privacy stack, low code reuse, established alternatives and unknown demand. Domain schema/grammar largely replaced | Prefer only after an identified workflow gap and a cheap comparative trial. Reject a general feature list or a speculative market |
| **Selectively rebuild** | Retain contracts, evidence, regression cases and verdicts; replace a specifically failing semantic or storage component | Roughly 4–8 weeks for a scoped replacement prototype, potentially much more for a product. Migration/compatibility risk; no evidence current solver boundary blocks progress | Prefer if adapters cannot support a demonstrated requirement without pervasive coupling. Reject an aesthetic rewrite or unmeasured “clean architecture” rationale |
| **Archive/abandon active product work** | Preserve knowledge, artifacts, code and reproducibility instructions; stop ongoing product opportunity cost in 1–2 days | Forego uncertain upside; frozen dependencies still need archival instructions. No new product automatically follows; no deletion needed | Prefer if time/budget is scarce, no user values outputs, or baseline trials remove the distinctive value. Reject only if inexpensive, independently evidenced benefit warrants continuation |

The leading hybrid is **repair → explicit continue/stop decision → conditional incremental evolution**. It does not mean doing all three regardless of results. Its case rests on reversibility and cheap information gain, not sunk investment. A full rebuild is inferior now because the inspected solver seam already permits replacement of proposals without discarding source binding and evaluation contracts. [E03–E08, E12–E18](evidence-register.md#repository-evidence)

## 8. Transparent decision matrix and sensitivity

Criteria were defined before scores. Scores represent current decision judgments, not measured ROI, probabilities, or statistical evidence. A high reuse score is deliberately only 5% of the total, so sunk implementation cannot dominate.

| Criterion | Weight | Meaning of a higher score |
| --- | ---: | --- |
| User value/problem significance (V) | 20% | More credible useful outcome for a specific user |
| Technical feasibility (F) | 15% | Fewer unproved technical dependencies |
| Differentiation (D) | 10% | Stronger plausible advantage over available alternatives |
| Useful asset reuse (R) | 5% | More reuse that reduces future work, not historical effort |
| Time to testable value (T) | 10% | Faster meaningful result |
| Development/operating cost (C) | 10% | Lower total future cost |
| Reliability/privacy/maintainability (Q) | 15% | More tractable assurance and ongoing support |
| Incremental improvement potential (I) | 5% | Clear bounded path to useful learning/improvement |
| Evidence strength (E) | 10% | Fewer material assumptions unsupported by observations |

Scale: **1** unfavorable/mostly unsupported; **2** substantial gaps; **3** plausible/mixed; **4** favorable with bounded gaps; **5** strongly favorable for this option's scoped outcome. Confidence is high for current implementation facts, moderate for effort/feasibility, and low for demand/differentiation across every option.

| Option | V | F | D | R | T | C | Q | I | E | Weighted /5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Keep | 2 | 5 | 2 | 5 | 5 | 5 | 2 | 1 | 4 | 3.35 |
| Repair | 3 | 4 | 3 | 5 | 4 | 4 | 4 | 4 | 4 | **3.75** |
| Evolve | 4 | 3 | 3 | 5 | 3 | 3 | 3 | 5 | 2 | 3.30 |
| Transform | 3 | 2 | 1 | 2 | 1 | 2 | 2 | 4 | 1 | 2.00 |
| Selectively rebuild | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 4 | 2 | 2.85 |
| Archive active product | 1 | 5 | 1 | 3 | 5 | 5 | 4 | 1 | 3 | 3.15 |

Formula: `sum(weight × score) / 100`. The decimals make arithmetic auditable; their precision does not imply precise preference measurement.

**Score rationale:** Keep's V2/D2/Q2/I1 reflect narrow benefit, generic alternatives, unresolved defects and no improvement; F5/R5/T5/C5 reflect an already runnable snapshot; E4 applies to preserving that known state, not efficacy. Repair's V3/D3 remain unproved, F4/T4/C4/Q4 reflect a bounded existing seam and correctable maintenance gaps, R5 retains relevant assets, I4 permits targeted tests, and E4 reflects directly observed repair targets. Evolve's V4/I5 are greater potential, F3/T3/C3/Q3 account for new semantic scope, D3/R5 retain the same tentative advantage/assets, and E2 penalizes absent generalization/demand. [E03–E18](evidence-register.md#repository-evidence)

Transform's V3/I4 allow a plausible coherent workflow; F2/T1/C2/Q2 reflect major new infrastructure; D1 reflects crowded alternatives; R2 limited direct reuse; E1 no demand/performance evidence. Rebuild's V3/D3/I4 are conditional potential, F3/C3/Q3 a manageable but unnecessary replacement, R3 partial preservation, T2 migration overhead, E2 no demonstrated architectural blocker. Archive's V1/D1/I1 mean no new product benefit, F5/T5/C5 easy cessation and low ongoing cost, R3 preserves research but does not exploit active code, Q4 reduces exposure while retaining archival upkeep, E3 reflects strong reasons for restraint but incomplete evidence that continuation is worthless. [E06–E18; X01–X12](evidence-register.md)

### Sensitivity and decisive trade-offs

| Changed priority/evidence | Explicit change | Result and interpretation |
| --- | --- | --- |
| Severe time/cash constraint | Weights V/F/D/R/T/C/Q/I/E = 10/10/5/5/25/25/10/5/5; unchanged scores | Keep 4.00, repair 3.90, archive 3.90, evolve 3.25, rebuild 2.75, transform 1.85. A frozen research snapshot is preferable if even the small trial displaces more valuable work |
| Much stronger emphasis on user value | Weights = 35/10/10/5/5/5/10/10/10; unchanged scores | Repair 3.60, evolve 3.55. Weighting an unproved benefit more heavily does not create evidence |
| Independently validated demand **and** quality | Previous weights; hypothetical evolve F/D/Q/E rise to 4/4/4/4, others unchanged | Evolve becomes 4.05. This is a conditional scenario, not a claimed result; passing the proposed tests could justify expansion |
| More/larger GPUs | No score change without workload measurements | Hardware may reduce runtime; it does not repair invalid labels, semantic scope or demand. Memory/serving fit still requires measurement |
| Less hardware or no paid budget | Use existing local reference/replay, annotation and manual comparisons first | Repair can start without training; if model comparison cannot run locally, narrow the trial or preserve/archive. Do not exceed campaign constraints |
| Existing tool matches quality and review burden | Repair D/V fall; absent unique benefit becomes decisive | Prefer adopting the tool and preserving research; extract only components with demonstrated independent use |

No numerical ranking overrides the stop conditions. Repair wins only as a bounded information-gathering investment. If there is no identifiable user or access to independent annotation, its central premise fails.

## 9. Component-level retention and retirement recommendations

These are prospective dispositions, not changes made during this task.

| Component | Classification | Evidence, dependencies and reason |
| --- | --- | --- |
| `nanoscribe/scribe.py`, encounter/adapt/check modules | **Preserve and improve** | Connected seam works; improve partial usefulness/conflict handling without weakening source checks. Depends on solver contracts and independent semantic fixtures [E03–E05, E12] |
| `nano_ai/contract.py`, solver interface | **Preserve and improve** | Strong typed proposal and explicit-failure boundary; preserve compatibility/source pins for consumed experiments [E04, E11] |
| Development compact/context modules | **Refactor only as evidence requires** | Bounded vocabulary/shared checker and grammar can hide correlated failures. Keep replaceable solver seam; compare semantic alternatives before replacing it [E05, E12] |
| Frozen H6/H7 adapters, preregistrations, terminal artifacts and Paper α | **Preserve unchanged** | Reproducibility and negative/invalid outcomes are evidence; no rerun, relabeling, freeze changes or retrospective success [E09–E11] |
| Pilot raw proposals, annotations, failed/interrupted runs and challenges | **Preserve unchanged** | Development evidence/regressions; never promote exposed cases into independent confirmation [E12–E15] |
| `fabric/`, `trajectory/` | **Preserve unchanged as research evidence; improve only unfrozen tooling under existing rules** | Closed-world verifier assumptions and lineage useful; cannot establish open-world clinical faithfulness [E08–E10] |
| `nano/` contracts/runtime/kernel | **Investigate before integration** | Separate executable research substrate; in-memory ledger and clinical coupling. Integrate only when a demonstrated P1 requirement benefits [E06] |
| `wedge_v1/` | **Preserve as optional reference utility** | Classical local text/span search; no reason to make it mandatory for transcript extraction [E07] |
| `benchmarks/` | **Preserve and improve selectively** | Useful metrics/adapter contracts; prioritize tests serving active decisions, avoid treating registry breadth as capability [E08, T06] |
| Evaluation/checking API subset | **Extract only after a downstream use is shown** | Possible independent asset, but packaging before demand adds support burden; license and runtime resolution prerequisites [E04, E16–E18] |
| Future document OCR/parsing, serving engine, chart engine | **Replace proposed custom builds with existing tools** | Not an instruction to delete current modules; these broad implementations are absent from the active path. Integrate Docling/runtime/plotting tools if a future experiment needs them [X02–X09] |
| Packaging/configuration/CI | **Preserve and improve** | Resolve supported runtime, test discovery, pin mismatch, license text and installation evidence before distribution [E16–E18] |
| Superseded status claims and duplicate future scaffolding | **Retire from active guidance after normal review; preserve history** | ACTIVE_NOW outranks stale H7/pilot narrative; no historical evidence deletion [E01, E15] |
| Span-port/P4/CUAD branches and external checkpoint archive | **Investigate further before deciding** | No cross-branch implementation audit or archive integrity audit here; do not infer redundancy or delete [E00, E02] |

## 10. Recommended product concept, if any

Keep nano-lm's identity and mission. The proposed near-term deliverable is its existing **bounded P1 research/developer workflow**: a reproducible way to compare transcript extraction proposals, preserve exact support and unresolved states, and inspect a checked draft together with what it omitted or refused.

The value hypothesis is that this reduces unsupported assertions **without paying for correctness through excessive omission or review effort**. The result is useful beyond a chatbot only if a builder can reproduce decisions, inspect original support and errors, substitute a solver, and measure completeness and abstention together. None of that establishes a market by itself.

This is a scoped validation vehicle for the current program, not new branding, a replacement intelligence architecture, or a declaration that Nano should remain a five-field note tool forever.

## 11. Intended users and problem to solve

The initial intended users are researchers and developers building faithful extraction systems with public/synthetic material. They need to distinguish a syntactically valid answer from a semantically correct, adequately complete representation, and to know whether a proposed checker actually helps.

Clinicians are prospective domain evaluators and eventual users, not validated current customers. The repository's P1 exit still requires external held-out medical evaluation, blinded human usefulness/safety assessment and an owner-signed exit. This report neither replaces nor satisfies those requirements. [E02, E19](evidence-register.md#repository-evidence)

There is no observed willingness to adopt or pay. A small counterbalanced task study is more informative now than a polished interface or a large training run.

## 12. What it should and should not do

The minimum workflow should accept a bounded, explicitly supported transcript; run a pinned interchangeable solver; distinguish supported, absent, missing, uncertain and conflicting states; retain exact source/attribution; expose malformed input and inference failure; render review-only output; and report omissions and abstention alongside correctness. An out-of-scope transcript should have a visible reason, not an implied successful medical interpretation.

The next phase should not promise comprehensive medical notes, autonomous chart writes, diagnosis, general comprehension, audio transcription, durable longitudinal memory, document ingestion, chart generation or P2–P9 completion. Those require distinct evidence and are outside this recommended trial. No PHI/private owner material is authorized. Model-provided confidence, fluent explanations and source offsets should not be relabeled independent truth.

## 13. Proposed minimum viable release and measurable success criteria

This would be a **nonclinical research release** after a successful trial. The criteria below are proposed decision thresholds, not existing frozen thresholds, observed results, or a P1 exit gate. Finalize them prospectively with the independent reviewers before scoring any new comparison.

| Dimension | Proposed acceptance and measurement |
| --- | --- |
| Reproducibility | Fresh documented environment runs the CLI, explicit relevant suites and saved-proposal replay; publish exact code/model/data/runtime identities and known skipped optional features. Resolve license/distribution uncertainty before redistribution |
| Instrument validity | Independently double-annotate fresh examples; adjudicate every disagreement; report pre-adjudication state agreement and span agreement separately. Require state agreement ≥0.90 and exact-span agreement ≥0.85 for a screening instrument; if not, revise annotation rules using separate development cases before evaluating |
| Faithfulness and grounding | Zero observed critical unsupported/incorrect final claims in the new screening cohort; all presented evidence must bind to original source and correct subject/time. Report denominators and interval estimates clustered by encounter; zero observed errors is not proof of zero risk |
| Coverage and utility | At matched error tolerance, ≥10 percentage-point useful-claim coverage improvement over the strongest simpler baseline, **or** ≥20% median review-time reduction at matched coverage with no critical-error increase. Predefine the primary endpoint and comparison; exploratory alternatives stay labeled exploratory |
| Conflict/uncertainty | ≥90% correct conflict and uncertainty retention on their designated strata; report every medication/denial error and every unnecessary whole-note abstention |
| Latency/resource | Publish cold/warm p50/p95, token/candidate counts, RAM/VRAM and retries on a named runtime. Provisional interactive target p95 ≤30 seconds for the declared short-transcript envelope; if it fails, assess batch usefulness explicitly rather than conceal latency. Report total cost per accepted useful draft including review |
| User usefulness | Five prospective users perform matched synthetic review tasks; at least three prefer the workflow, with the predefined time/quality criterion met. This is directional screening, not market validation |

Define useful-claim coverage as correctly presented final-note fields divided by **all gold supported/absent fields across every case**. Whole-note abstentions, inference failures, incorrect states and omitted eligible fields contribute zero to the numerator and remain in the denominator. Measure uncertainty/conflict retention separately rather than treating those states as asserted facts. Before annotation, define critical errors as unsupported presented clinical facts, wrong subject/time attribution, reversed denials, or unresolved conflicts presented as resolved; reviewers must finalize any additional severity rules before scoring. Ordinary omissions remain visible in coverage and review burden.

Evaluate a prospective **100-encounter/500-field** cohort with independently written, nonpaired synthetic cases covering the **22 admissible field/state cells**. For chief complaint, duration and severity, allocate 25 instances to each of supported/missing/uncertain/conflicting; for medication/allergy, allocate 20 to each of all five states. This totals 500 fields while respecting contract v0. Test out-of-contract denials separately for safe rejection, not as impossible gold targets. This artificial distribution measures failure modes, not clinical prevalence. Use independent reviewers with relevant clinical-language expertise, separate development data and authoring templates from the test cohort, and freeze code/prompts/scoring rules before testing. A small, provenance-reviewed adaptation of ACI-Bench can later test longer external dialogue. Its paired note is not evidence for content absent from the input, and speaker-tag errors require explicit handling. Neither dataset automatically meets the clinical exit. [E04; X13](evidence-register.md)

Baselines should include the current deterministic reference, an existing compact or hosted model with a schema prompt, that same model plus simple exact-source validation, and the full Nano checked path. Hold source cases, field definitions, model identities, exposure and scoring constant. Do not pool this with H6/H7, C2, historical compact results or the exposed pilot. Use raw, checked and final-note metrics separately.

Failure cases must include speaker swaps; historical/future/third-party statements; negation scope; conflicting medication use; uncertainty versus missingness; repeated source text; corrections; compound questions; malformed/truncated/Unicode inputs; multiple entities; and accurate claims lost when another field fails. Include unsupported units/numeric paraphrases and explicit out-of-scope inputs even though numeric analysis is not a current capability.

If a future information-understanding concept is selected, add distinct tests for summary proposition coverage, contextual entailment, contradiction detection, numeric exactness and units, aggregation denominators, misleading axis/scale/labels, chart-to-data correspondence, qualified inference and user task success. Current span and software tests provide no substitute for these measurements.

## 14. Cheapest decisive experiments

**Proposals only: none were run.** Engineering times are estimates; reviewer recruitment/annotation may dominate elapsed time. A bounded continuation should use at most roughly ten engineering days before a new disposition decision, with reviewer availability established first. The estimates overlap rather than adding an automatic multiweek program. Any confirmatory run needs its own identity, preregistration and experiment-scoped authorization; exposed data and consumed H6/H7 evaluations stay consumed.

| Experiment | Assumption and smallest practical test | Resources/effort estimate | Success, failure and decision consequence |
| --- | --- | --- | --- |
| **A. User-need falsification first** | Five prospective builders review current synthetic examples and their normal alternative in blinded/counterbalanced tasks; collect corrections, time, useful omissions and preference | Existing outputs; about one engineering day plus 2–4 participant hours; no model spend necessary | ≥3/5 prefer and ≥20% median review-time benefit without worse critical errors. If no repeated relevant task or benefit, stop product investment; preserve research |
| **B. Independent semantic comparison** | First check 20 independently annotated development cases; if evaluable, test a separate 100-case fresh screening cohort, with no smoke-case reuse. Compare all four baselines, raw→checked→note, with code/prompts/scoring fixed before screening | About 3–5 engineering days plus independently estimated annotation/adjudication time; local cached models first. Optional hosted ceiling: propose ≤$10 incremental usage only within verified experiment authorization and remaining budget | Thresholds in §13. Invalid instrument → no model verdict. Failure to improve coverage/review burden at matched errors → adopt simpler baseline/archive product. Critical errors → repair only on separate development cases; any later confirmation uses a fresh cohort |
| **C. Reproducible use and distribution** | Clean environment, documented Python/dependency set, one CLI example, explicit suites, saved replay, license and model/data manifest review | 1–2 engineering days; CPU/reference path first, no new training | A new user reproduces outputs using instructions without undocumented local state. Failure directs maintenance effort; persistent expensive setup undermines a library/release strategy |
| **D. Narrow alternative comparison** | Only if A fails but participants identify document comparison as a real need: ten public/synthetic document pairs, existing assistant/extraction tools, manually scored changes and review effort | 1–2 days, no new product code; use free/local options first, proposed ≤$5 optional hosted ceiling subject to budget rules | ≥20% time benefit with no critical claim/number errors and ≥3 users wanting repeat use. If existing tools already satisfy the need, use them; do not transform nano-lm merely to reproduce them |
| **E. Model economics, conditional** | Only after A/B succeed: compare existing pretrained compact and hosted proposals at the same quality/coverage before any fine-tuning | 1–2 days; record all-in cost, latency and reviewer time; no initial self-hosted frontier model | Require a credible ≥30% total-cost reduction or a specific quality gain before pursuing custom training. If serving/annotation costs dominate, retain hosted/pretrained proposals |

Run A and annotation feasibility before semantic engineering. A 20-case smoke is developmental and cannot be reused as untouched confirmation. The 100-case cohort is a small screening decision, not sufficient evidence for medical safety or broad deployment.

The safe experiments actually completed were local software checks, a reference CLI/fixture run, and byte-for-structure equality of all 20 saved pilot drafts. They resolve execution/replay uncertainty; they do not answer user demand, generalization or comparative economics. [T01–T11](evidence-register.md#observed-execution)

## 15. Main risks, uncertainties and unresolved questions

1. **Semantic correlation:** generation constraints, binder and checker share assumptions. Strong internal agreement can conceal shared errors. Independent annotations and a simple validation baseline are decisive.
2. **Selection and exposure:** 20 dialogues are ten paired worlds, developed on repeatedly. The five remaining checked-field errors are all medication; new language may be substantially harder. No confidence claim about broad clinical safety follows.
3. **Usefulness versus abstention:** 78 checked assertions become 56 final claims. Whole-note quarantine is conservative but can discard valid information. Matched-coverage comparisons prevent gaming by refusal.
4. **No validated demand:** existing products may already meet the intended need. A research contribution can remain worthwhile without a product; continuation must have an explicit research question or user outcome.
5. **Maintenance/distribution:** runtime drift, incomplete default/CI test selection, stale narrative and missing license text complicate adoption. A passing preexisting environment is insufficient release evidence.
6. **Security and data:** no PHI/private material is permitted; production authorization, retention, durable storage and deployment have not been established. No comprehensive security assessment was performed.
7. **Economics:** no current provider balance or future workload was verified. Respect the existing $180 autonomous envelope, $200 hard campaign cap and $20 owner reserve; those limits are not remaining balances. Free Kaggle when suitable, RunPod when exact requirements justify it, hosted teacher when cheaper. Hardware availability alone is not a reason to train or self-host.
8. **Integration scope:** feature-branch work is not live-master integration; other branches and the external checkpoint archive were not audited. No deletion or blanket replacement decision about them is supported.

## 16. Recommended next actions in priority order

1. Review this disposition and choose whether the bounded continuation is worth its opportunity cost. This investigation ends with the reports; it launches no experiment or implementation.
2. In the next separately selected phase, recruit independent reviewers and test the actual user task before expanding code. If this cannot be done, keep a research snapshot or archive active product work.
3. Verify the new phase's canonical remote base and preserve the feature-branch evidence during any normal integration. Resolve reproducible setup, license text, explicit suite coverage and stale status references without altering protected evidence.
4. Address the demonstrated medication/conflict and overly broad quarantine failures using separate development data. Keep solver/source boundaries, raw outputs, error denominators and failure visibility intact.
5. Run a new prospectively specified independent comparison only under the applicable experiment rules; decide continue/adopt-baseline/archive from the results.
6. If both user benefit and independent quality pass, select one bounded P1 extension. Consider P2/P3 only through their existing prerequisites. Do not treat this report as an alternative roadmap.

## 17. Evidence and reasoning behind the recommendation

The strongest argument **for** continuation is an already executable, reproducible boundary between proposals, evidence, records and conservative outputs. The extensive passing software tests and exact replay make a targeted repair cheap relative to a rewrite. The current active plan already names separate-data semantic-state/conflict work, so this recommendation follows the actual milestone rather than inventing a new program. [E02–E05, T01–T11](evidence-register.md)

The strongest argument **against** continuation is that the favorable result is a small exposed hybrid development set, usefulness is reduced by abstention, semantic errors persist, and established tools offer much of the general workflow. No independent demand or cost advantage has been measured. H6 is a real negative result; H7 cannot establish comparative model quality because its gold instrument failed. Neither should be disguised as success, and neither by itself proves the whole program impossible. [E09–E15; X01–X12](evidence-register.md)

A full rewrite is unsupported because no measured architectural barrier prevents trying a better proposal or context model through the existing seam. A broad transformation is unsupported because it adds the most unverified dependencies while competing with available tools. Immediate archival is defensible under severe resource constraints, but otherwise a short user/independent-data test has a favorable information-to-effort ratio. That is an inference, not a return-on-investment claim.

**Primary recommendation: continue by repairing and stabilizing bounded P1, contingent on a short independent benefit test; do not begin broad transformation or substantial model investment.** This is moderate-confidence support for a limited continuation, not evidence that the eventual product will succeed.

**Fallback: preserve the empirical record and archive active product development if independent review, comparative quality or user benefit fails.** Change toward incremental evolution only when those tests demonstrate useful advantage; change toward a narrow alternative only when its user need and baseline gap are independently shown. More hardware, additional features, or previous effort alone should not change the decision.
