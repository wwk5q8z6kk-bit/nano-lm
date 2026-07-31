# Discussion-to-Implementation Matrix
*Generated 2026-07-31T12:40:09.520615+00:00 · HEAD `0e01d73` · 57 rows*
## Status counts
| Status | n |
|--------|---|
| DISCUSSED_ONLY | 25 |
| EXPERIMENT_CLOSED_SUPPORTED | 8 |
| PARTIALLY_IMPLEMENTED | 7 |
| IMPLEMENTED_AND_VERIFIED | 5 |
| PREREGISTERED_NOT_IMPLEMENTED | 3 |
| EXPERIMENT_CLOSED_REFUTED | 2 |
| BLOCKED | 2 |
| SUPERSEDED | 1 |
| EXPERIMENT_CLOSED_UNRESOLVED | 1 |
| ABANDONED_WITH_REASON | 1 |
| DOCUMENTED_BUT_UNANCHORED | 1 |
| IMPLEMENTED_NOT_VERIFIED | 1 |

## Matrix

| ID | Proposal | Status | Code | Tests | Artifacts | Evidence (short) | Gaps | Disposition |
|----|----------|--------|------|-------|-----------|------------------|------|-------------|
| EMP-T-LADDER | Stage T / T-v2 Pythia ladder multi-instance gaps | EXPERIMENT_CLOSED_SUPPORTED | trajectory/kaggle_arm1.py, kaggle_arm1_v2.py, rescore_anchors.py | No dedicated pytest; determinism checks documented | results_arm1_v2_pythia-*.json, results_anchors_v2_*.json | JSON gaps nano 18.3±1.3 scale 18.7±1.5 pythia160m 3.5±0.7; FINDINGS; paper-alpha-v1 | Prereg RESULT sections stale/missing; raw JSONL not in git; 1B is interval | Keep as α core; sync prereg RESULT sections |
| EMP-FIELDWISE | Fieldwise / open vs closed localization | EXPERIMENT_CLOSED_SUPPORTED | trajectory/kaggle_pythia_fieldwise.py, fieldwise_anchors.py | None | results_fieldwise_*.json | dur/sev gap 0; open fields nonzero in anchors JSON | No pytest pin | Keep; optional CI pin of closed-field zeros |
| EMP-OWNSTACK | Own-stack 160M full-FT / LoRA / Chinchilla factorial | EXPERIMENT_CLOSED_SUPPORTED | kaggle_ownstack_160m.py, kaggle_ownstack_160m_lora.py | None | results_ownstack_v2_160m_*.json | fullft 16.9±1.7; lora 7.1; chinchilla 7.0 | Prereg header still 'Not executed' | Update prereg RESULT; treat as Supported behavioral |
| EMP-CORNER | Factorial corner 3.2B+LoRA seed replication | EXPERIMENT_CLOSED_SUPPORTED | kaggle_ownstack_160m_lora.py (corner config) | None | results_corner_3p2b_lora_seed{0,1}.json | both seeds diluted_gap_mean 4.24±0.91 | Mechanism unidentified | Behavioral claim only |
| EMP-DIVERSITY | Slot-diversity sweep D5/D20/D80/D20-pos | EXPERIMENT_CLOSED_SUPPORTED | kaggle_sweep_10m.py, gen_slot_diversity_eval.py | None | results_sweep_10m.json, sweep_eval/ | H-slot SUPPORTED; +66.7 pts; position innocent | Prereg Status still 'Nothing has been run' | Sync prereg RESULT |
| EMP-C1-COVERAGE | C-1 simple token-coverage account | SUPERSEDED | gen_token_coverage_pools.py (pools) | None | token_coverage_bands.json (design); no GPU band results | Dry-run falsified coverage-as-driver; amended to C-1b | Never GPU-executed as C-1 bands | Remove from active plan |
| EMP-C1B | C-1b lexical interference | EXPERIMENT_CLOSED_REFUTED | run_interference_10m.py, analyze_interference.py | None dedicated | results_interference_10m.json, outputs_if_seed*.jsonl (local, gitignored) | delta negative; REFUTED → C-3 promoted | JSONL gitignored; no formal replication | Closed |
| EMP-C3 | C-3 transition × boundary × length binding probe | EXPERIMENT_CLOSED_UNRESOLVED | run_c3_10m.py, recompute_c3.py | trajectory/test_recompute_c3.py PASS (7) | results_c3_10m.json, results_c3_recompute.json, outputs_c3_seed*.jsonl, replications/c3/ | Recompute 2026-07-31: T+1.7 REFUTED B-8.3 REFUTED L+25 UNRESOLVED; replication REPRODUCES | H-length unresolved; morphology exploratory; unstable-exclusion bug fixed 823e1ca | Keep closed; morphology not causal claim |
| EMP-MORPH | Morphology / re-inflection residual causal follow-up | DISCUSSED_ONLY |  |  | C-3 error census prose | Docs say descriptive; no causal prereg executed | No prereg, no runner | SPECULATIVE_DEFER unless new question |
| EMP-POINTER-P1 | Pointer/copy head P1 | ABANDONED_WITH_REASON | scribe/pointer/train.py, gate.py | Manipulation check in gate | result_pointer.json, result_baseline.json | M=0.18 < 0.2 → VOID |  | Historical; do not cite as mechanism evidence |
| EMP-POINTER-P2 | Pointer/copy head P2 supervised | EXPERIMENT_CLOSED_REFUTED | train2.py, gate2.py | Manip PASSED | result_pointer2.json | manip M=0.97; val held 10%→10% zero delta | Does not refute all copy mechanisms | Cite only as this-impl failure |
| EMP-E1 | E1 non-LM utility kill-gate | EXPERIMENT_CLOSED_SUPPORTED | trajectory/e1/*.py | No dedicated pytest for U scorer | results_e1_utility.json (+ 48 method/item JSONs); sensitivity JSON | decision.verdict=KILL; U(M1)=0.999 vs official M0=0.925; sensitivity stable; pytest suite does not cover E1 | Untracked vs paper-alpha-v1; official M0 verify-off not in utility rows; ρ mislabeled in DECISION_P1; 'human' construct  | Treat KILL as settled for this U/task; sync docs; add U scorer unit tests |
| EMP-E3-AUTO | E3 normalize construct (auto) | EXPERIMENT_CLOSED_SUPPORTED | trajectory/e3/run_e3_normalize.py | None | results_e3_normalize_construct.json | 0/486 rescues; gap_shrink 0.0 |  | Keep |
| EMP-E3-HUMAN | E3 human/soft faithfulness arm | PARTIALLY_IMPLEMENTED | Labels in results_e3_human.json (no dual-rater tooling) | None | e3_human_rating_pack.json, results_e3_human.json | Pack labeled; rater=agent-rubric-pass-1; IAA null; faithful-rate 0.00 | Not clinician dual-annotate; FINDINGS/RESEARCH_PROGRAM still BLOCKED; Paper α still 'pending' | Document as rubric audit; optional real human IAA |
| EMP-E2 | E2 LoRA universe discrimination U1–U4 | BLOCKED | trajectory/e2/run_u3_earlystop.py | None | None results_e2_*; runpod_partial e2_* metadata only | No RESULT JSON; EMPIRICAL_FOUNDATION says pod terminated; runpodctl pod list empty | Status drift; partial setup logs | Converge docs to GATED/STOP/VOID; no unpaid compute found |
| EMP-E4 | E4 R★ classical vs generative kill gate | PREREGISTERED_NOT_IMPLEMENTED | No R★ builder | None | None | §4.7 builder not implemented; no classical probe JSON | Entire data world missing | Owner authorize Stage 4 or Idle |
| EMP-STAGEM-PATCH | Stage M activation-patching Q(M) | PREREGISTERED_NOT_IMPLEMENTED |  |  |  | Prereg only |  | DEFER; gated by program |
| EMP-STAGEM-IND | Stage M induction curriculum | PARTIALLY_IMPLEMENTED | stage_m/stage_m_kernel.py, _probe.py | None | None results | Kernel exists; amended pre-measurement; no RESULT JSON | Never measured | DEFER |
| EMP-COMP-VERIFY | Compositional verification Stage V | PREREGISTERED_NOT_IMPLEMENTED |  |  |  | Prereg says not authorized |  | DEFER / fabric gated |
| FAB-SCHEMAS | Typed claim / evidence / decision packets | IMPLEMENTED_AND_VERIFIED | fabric/schemas.py | schemas self-test + test_fabric.py uses Claim |  | pytest fabric 8/8 PASS; frozen dataclasses + _cid |  | Keep as regression harness |
| FAB-V1V2 | Presence/absence grounding verifiers v1/v2 | IMPLEMENTED_AND_VERIFIED | verify_value*, verify_absent* | 8 tests cover grounding/absence/decide | results_slice_v1.json, ledger_*.jsonl | pytest PASS; measured presented-error→0 under v2 | v2 is rules-perfect extractor (documented) | Keep; do not expand to product |
| FAB-ABSTAIN | Abstention / present / qualify policy | IMPLEMENTED_AND_VERIFIED | decide() | test_decide | results_slice_v1.json abstained counts | pytest + measured stats | No calibrated risk controller | Minimal policy only |
| FAB-LEDGER | Append-only content-addressed evidence ledger | PARTIALLY_IMPLEMENTED | slice.py ledger.write with open(...,'w') | No ledger tests | ledger_*.jsonl local/gitignored? | IDs content-addressed REAL; file opened with 'w' truncates each run — not append-only API | No duplicate detection, stale rejection, transactions, schema version | Docs must not claim append-only DB |
| FAB-INTENT-CTRL | Intent → Control kernel path | DOCUMENTED_BUT_UNANCHORED | Absent — jump to generator |  |  | Comment claims Intent→Control; code has no kernel | Entire control plane missing | Remove from 'implemented' language |
| FAB-SEMANTIC | Semantic verifier | DISCUSSED_ONLY |  |  |  | README: v2 not semantic |  | DEFER |
| FAB-COMPOSITIONAL | Compositional / adversarial verifier suite | PARTIALLY_IMPLEMENTED |  | Partial: wrong-speaker/wrong-slot in test_fabric for v1/v2 |  | Some adversarial cases in unit tests; no full Stage V | No paraphrase/negation/distractor eval generators | HIGH_VALUE for harness hardening; not product unlock |
| FAB-MEMORY | Typed memory write of validated state | DISCUSSED_ONLY | Absent |  |  | No memory module |  | REMOVE from current claims |
| FAB-SLICE-E2E-TEST | End-to-end run_slice pytest with checkpoints | IMPLEMENTED_NOT_VERIFIED | run_slice exists | No e2e test in CI | results_slice_v1.json | Measurement exists historically; CI only unit tests | CI does not re-run slice | Optional offline pin; costly |
| NS-KERNEL | Control kernel | DISCUSSED_ONLY |  |  |  | Specified in NANOSCRIBE_VNEXT; no code | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-STATE | Factorized state S=I×K×R×M×P×V×E | DISCUSSED_ONLY |  |  |  | MASTER_PLAN formula only | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-DAG | Task DAG | DISCUSSED_ONLY |  |  |  | vNext Next queue | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-ROUTER | Capability routing | DISCUSSED_ONLY |  |  |  | Specified; no router | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-CTX | Context compiler | DISCUSSED_ONLY |  |  |  | Specified; no code | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-TOOLS | External/symbolic tools | DISCUSSED_ONLY |  |  |  | No adapters | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-PERMS | Permissions / write authorization | DISCUSSED_ONLY |  |  |  | Invariant 6 prose only | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-MEM | Memory classes (episodic/semantic/procedural/graph) | DISCUSSED_ONLY |  |  |  | Specified only | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-DIST | Distributed workers / replay | DISCUSSED_ONLY |  |  |  | D1 stage future | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-UI | Audit / human review UI | DISCUSSED_ONLY |  |  |  | No frontend | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-OBS | Observability / dashboard | PARTIALLY_IMPLEMENTED |  |  |  | JSONL+results counters only | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| NS-DEPLOY | Release/deploy product path | BLOCKED |  |  |  | E1 KILL; no deploy pipeline | Not required for current research program until E4 SURVIVE | SPECULATIVE_DEFER / STOP |
| OSS-LANGEXTRACT | LangExtract | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-LLGUIDANCE | llguidance | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-GUIDANCE | Guidance | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-OUTLINES | Outlines | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-INSTRUCTOR | Instructor | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-RAGCHECKER | RAGChecker | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-OPENEVALS | OpenEvals | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-SIGSTORE | Sigstore/model-transparency | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-SSM COMPARIS | SSM comparisons | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| OSS-POINTER-GENE | pointer-generator baseline M6 | DISCUSSED_ONLY |  |  |  | rg across repo: zero matches for LangExtract/llguidance/Guidance/Outlines/Instructor/RAGChecker/OpenEvals/Sigstore | Not in requirements*.txt | DEFER; BENCHMARK_ONLY if E4 needs stronger classical/structured-decoding refs |
| DISC-CI | CI regression suite | IMPLEMENTED_AND_VERIFIED | .github/workflows/ci.yml | Local pytest PASS 2026-07-31 | _pytest_*.txt | 15+8+7 tests green locally; workflow present (untracked?? check) | .github may be untracked relative to origin tag | Commit CI if owner wants archival |
| DISC-REPRO | Reproducibility packaging (hashes, env, tags) | PARTIALLY_IMPLEMENTED | Various fingerprint fields in JSONs | C3 recompute | tags paper-alpha-v1, stage-t-v2-results | Tags exist; many post-α artifacts untracked; JSONL gitignored | E1/E3/P1 stack not in git | MUST_FIX_BEFORE_CLAIM for post-α locks |
| DISC-PREREG-GATES | Preregistration-before-run discipline | PARTIALLY_IMPLEMENTED |  |  | 15 PREREG files | Strong for E1/C3/C-1b; several preregs lack RESULT updates after execution | Stale Status headers | Sync RESULT sections |
| DISC-PAPER-ALPHA | Paper α camera-ready release | IMPLEMENTED_AND_VERIFIED | make_figures.py |  | paper1.pdf, tag 0e01d73 | Tag exists; manuscript has E1 KILL §0 | Limitation text still 'pending human study' after Stage 1 rubric audit; arXiv pending | Optional limitation wording patch (owner) |
| DISC-PAPER-BETA | Paper 2/β verification systems manuscript | DISCUSSED_ONLY |  |  | paper2_draft.md reclassified as α extension | No dedicated β manuscript |  | DEFER |
| EMP-M4-CONSTRAINED | E1 M4 constrained/finite-state copy baseline | EXPERIMENT_CLOSED_SUPPORTED | trajectory/e1/methods.py | None | results_e1_nonlm_M4_*.json | U≈0.819 in utility JSON | Not Outlines/Guidance integration | Keep as in-house constrained baseline |
| NS-STRUCTURED-DEC | llguidance/Outlines/Guidance structured decoding product path | DISCUSSED_ONLY |  |  |  | No dependency or adapter |  | DEFER |

## Full fields

See `DISCUSSION_TO_IMPLEMENTATION_MATRIX.json` for source/spec/intent/superseded_by.
