# nano-lm — a language model pretrained and instruction-tuned from scratch on a MacBook

## Public status (2026-07-31)

**E1 (one sentence):** Under the frozen E1 utility, classical/rules methods beat official LoRA-160M on this closed scribe task (**KILL**; M1 U≈0.999 vs official M0 U≈0.925, δ=0.05).

| | |
|---|---|
| **Freeze tag** | `post-alpha-evidence-freeze-2026-07-31` |
| **Program state** | Science: `IDLE_AFTER_E4_KILL` (E4 **KILL** on R★). Product: `IDLE_AFTER_DOGFOOD` + runtime CLI live; next needs typed `AUTHORIZE_WEDGE_V1_*`. No E2/fabric/NanoScribe expansion; R★ revision budget 1 |
| **In-tree** | Paper α measurement spine (`paper-alpha-v1`) **plus** post-α E1/E3 evidence bundle archived at the freeze tag |
| **Not claimed by tag** | Retroactive proof of pre-run preregistration chronology; full CUDA bit-identical fine-tuning; dual-clinician human evaluation |
| **E3** | Agent-applied rubric audit (`agent-rubric-pass-1`) — **not** human/clinician evaluation; dual-clinician IAA **open** |
| **ρ in E1 \(U\)** | **Review load** (flagged fields / \(n\)), **not** hallucination |
| **Fabric** | Scoped verification / regression harness — **≠** NanoScribe architecture |

**Outsider entry:** [`papers/PUBLIC_ONE_PAGER.md`](papers/PUBLIC_ONE_PAGER.md) · **Speech acts:** [`papers/OWNER_SPEECH_ACTS.md`](papers/OWNER_SPEECH_ACTS.md) (`continue` = M0 only) · **Hybrid closeout:** [`audit/discussion-to-implementation/COUNCIL_HYBRID_CLOSEOUT.md`](audit/discussion-to-implementation/COUNCIL_HYBRID_CLOSEOUT.md) · **Evidence map:** [`papers/EVIDENCE_MANIFEST.json`](papers/EVIDENCE_MANIFEST.json) · [`audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`](audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md) · [`papers/EMPIRICAL_FOUNDATION.md`](papers/EMPIRICAL_FOUNDATION.md) · soft-claim withdrawals: [`audit/discussion-to-implementation/WITHDRAWAL_SPEC.md`](audit/discussion-to-implementation/WITHDRAWAL_SPEC.md). Distinguish **Paper α** (`paper-alpha-v1`) from the **post-α freeze bundle** (E1 KILL + E3 construct artifacts).

A 3.15M-parameter decoder-only transformer built end-to-end: data pipeline, tokenizer,
pretraining, and supervised fine-tuning (SFT) — all trained locally on Apple Silicon (MPS),
no cloud GPUs. Total pretraining wall-clock: **20 minutes**.

The point is not the model's size — it's exercising the *entire* modern LLM build stack
at a scale where every decision is legible and auditable.

## Purpose

nano-lm is a **testbed for evaluation-gated model development**: every capability the
model gains must pass a pre-registered behavioral gate (held-out prompts, base-model
control, honest failure reporting) before the next stage begins. At this scale a full
train-evaluate-gate-iterate cycle takes minutes, so evaluation *design* — what to
measure, how to prevent contamination, when a pass is real — can be prototyped and
stress-tested far faster than at production scale. The end goal is a miniature
**ambient-scribe** task: converting short dialogue transcripts into structured summaries,
gated on *faithfulness* (no content in the summary that isn't grounded in the dialogue) —
the same evaluation problem that production clinical documentation AI faces.

## Progression

| Stage | Status | Gate result |
|---|---|---|
| Pretrain (FineWeb, Chinchilla-scaled) | ✅ done | val loss 3.96, zero spikes |
| SFT v1 (SmolTalk, ChatML) | ✅ done | ❌ honest FAIL (72% stop, 33% refusal) |
| SFT v2 (diversified refusal slice, +format) | ✅ done | ✅ PASS (98% stop, 92% held-out refusal) |
| Preference pairs (best-of-n + rubric judge) | ✅ done | 163 pairs, margin ≥ 0.6 |
| DPO (β=0.1) | ✅ done | ✅ PASS (win-rate 80.6% vs SFT, 95% CI [75.6, 85.6]) |
| RLVR/GRPO (verifiable-reward slice) | ✅ done | ✅ PASS (pass@1 12.5% → 85.9%, Δ CI [+61.7, +85.2]) |
| Scribe v1 (faithfulness gate, pre-registered) | ✅ done | ❌ honest FAIL (recall 74%, halluc 14%) — diagnosis in audit |
| Scribe v2 (pre-specified diversity sweep) | ✅ done | ❌ FAIL by 1.5 pts (recall 81% ✅, halluc 11.5% vs ≤10%) — **stage closed per protocol** |
| Stage G: grounding-verifier guardrail | ✅ done | ❌ FAIL by 0.8 pts — but **0.0% residual hallucination** (23/23 caught, 14% review load); miss = unverifiable absence claims |
| Stage A: absence-verifier axis | ✅ done | ✅ **PASS — presented precision 100%** (0.0% residual halluc AND omissions, 33/33 errors caught, 19% review load) |
| Stage C: copy-curriculum hypothesis test | ✅ done | ❌ FAIL — **hypothesis decided**: held-out gap unchanged (22 pts) → capacity, not curriculum; omissions (10→0) converted into fabrications (11.5→17.5%) |
| Stage S: ~10M scale test (Kaggle T4) | ✅ done | ✅ **GATE PASS — first model to clear the bars** (parse 100%, recall 88%, halluc 7.5%) — but held-out GAP 23 pts, unchanged → capacity hypothesis **weakened** |
| Over-refusal gate axis (XSTest-style) | 🔜 planned | known gap, documented in audit |

The scribe track's arc is the finding. Three stages closed at honest FAIL — each protocol
allowed one measurement (plus, for v1→v2, one pre-specified sweep), and "one more try"
against seen results would be bar-chasing. Each FAIL located the next lever: v1's
position-anchored extraction → template diversity; v2's out-of-distribution hallucination
→ verification architecture; Stage G's unverifiable absence claims → the lexicon axis.
Stage A then passed its bars on this synthetic distribution: with a drafting model
whose intrinsic hallucination rate is 11.5%, the two-axis verification layer yields
**100% presented precision at 19% review load** under the measured verifier relation —
scoped to this task, not open-world hallucination elimination. Full trail:
`scribe/AUDIT.md`.

Stage S (10M params, Kaggle T4) sharpened the conclusion: the larger model became the
first to PASS the model-side bars (halluc 7.5%) — yet its out-of-distribution gap didn't
move (23 pts vs 22 at 3M). A model can pass a well-designed average-case gate while
keeping its tail failure mode, which is exactly why the verification guardrail is not
retired by scale. Full trail: `scale/AUDIT.md`.

## Research track: boundary conditions for held-out copying (`trajectory/`, `papers/`)

Stage S ended on a puzzle: a model can pass every model-side gate and keep a ~23-point
held-out gap. The research track turned that into a measured, pre-registered program.
**Baseline (2026-07-30, Scientific Research Council):** no architecture thesis; no
mechanism claim beyond evidence; fabric/v2 expansion gated behind E1–E3. Surviving
contribution: under low-diversity extraction regimes, small transformers can converge
to closed-set prediction strategies that fail held-out symbolic emission; diversity,
adaptation regime, and deterministic verification change the reliability profile.

Numbers in the **Paper α / `paper-alpha-v1` measurement spine** (anchors, Stage T/T-v2,
own-stack, diversity, C-3, pointer, fabric) trace to content-addressed per-run JSONs
under `trajectory/`. **Post-α E1/E3** utility/construct artifacts are archived under
tag `post-alpha-evidence-freeze-2026-07-31` and mapped in `papers/EVIDENCE_MANIFEST.json`
plus `audit/discussion-to-implementation/CANONICAL_STATUS_TABLE.md`. “Immutable” means
tagged archival state — not that every working-tree lockfile edit is frozen. Program
lockfile: `papers/EMPIRICAL_FOUNDATION.md` · map: `papers/RESEARCH_PROGRAM.md` ·
packaging: `trajectory/REPRODUCIBILITY.md`.

**Paper 1 — measurement** (`papers/latex/paper1.pdf` + empirical companion historically
in `papers/paper2_draft.md`): held-out value copying on a 5-instance instrument —
**18.3±1.3** (3.15M) / **18.7±1.5** (10M); failure localizes to open-vocabulary fields
(closed fields = 0). Within-stack 160M full-FT reads **16.9±1.7** (stack-dominant vs
Pythia 3.5±0.7). LoRA (**7.1±1.2**) and Chinchilla data (**7.0±1.0**) are substitutes;
factorial corner 3.2B+LoRA **4.2±0.9** (seed |Δ|=0.00). Slot diversity causal
(**+66.7 pts**, H-slot SUPPORTED). C-1b interference **REFUTED**; C-3 T/B **REFUTED**,
L UNRESOLVED; morphology residual **descriptive only**. Pointer head: this
implementation does **not** close the OOD gap (not a claim against all copy
mechanisms).

**Paper 2 — deterministic verification** (Stage G/A + `fabric/`; β manuscript TBD):
propose→verify→abstain under a decidable verifier relation on this distribution.
Presented precision **100%** at ~19% review load (Stage A) and fabric presented-error
**18.4%→0.0% / 11.5%→0.0%** under a rules-strong verifier — **scoped to this task and
\(R\)**, not open-world hallucination elimination. Fabric is a regression harness
(**≠** NanoScribe). After E1 KILL: product/architecture expansion **STOP**; E3
agent-rubric audit done (**EXACT_SURVIVES**; dual-clinician IAA open — not human
eval); E2 **GATED/STOP** (no results JSON). R★/E4 protocol docs may exist; **no**
E4 **KILL** on locked R★; operating posture **`IDLE_AFTER_DOGFOOD`** (science: `IDLE_AFTER_E4_KILL`; see `papers/AMBITION.md`).

Reproduce artifacts / env / CI: `trajectory/REPRODUCIBILITY.md`. Compute venues: local
Apple Silicon (MPS), Kaggle T4, RunPod; largest single run ≈ $37.

## Results

| Stage | Metric | Value |
|---|---|---|
| Pretrain | train loss | 8.35 (= ln V, init sanity) → 3.70 |
| Pretrain | val loss (held-out stream) | 3.96 |
| Pretrain | throughput | ~25k tok/s on MPS, zero loss spikes |
| SFT | masked loss (assistant tokens only) | 3.97 → ~1.5 (7.6 min) |
| SFT | throughput | ~53k tok/s |
| Behavioral gate (v2) | clean ChatML stop, sampled decoding | base 2% → **98%** |
| Behavioral gate (v2) | refusal on held-out unsafe phrasings | base 8% → **92%** |

## Architecture (`pretrain/train.py`)

- Decoder-only transformer, **3.15M params**, d=192, seq len 512
- RMSNorm (pre-norm), **SwiGLU** MLP (ff = 8/3·d), **RoPE** (base 1e4)
- **Grouped-query attention** 6q:2kv, tied input/output embeddings
- No dropout; init 0.02 depth-scaled; z-loss 1e-4

## Data & tokenizer

- Corpus: `HuggingFaceFW/fineweb` (sample-10BT), 12,000 docs streamed
- Filtering: Gopher/C4 heuristics; dedup via exact SHA-1 + MinHash (5-gram, 112 hashes, 14×8 bands)
- Tokenizer: byte-level BPE (HF `tokenizers`) + digit pre-tokenization, **V=4096**
  (sized by the 2·V·d embedding-budget rule); measured fertility 1.84 tok/word
- Tokenize-once → uint16 shards: **10.96M unique tokens**

## Compute budgeting

Chinchilla-style arithmetic drove the model size, not vibes:
10.96M unique tokens × ≤4-epoch cap → D≈33M tokens → **N≈3M params** (D≈20N).
Trained 4,000 steps / 32.8M tokens (~3.1 epochs). Optimizer: AdamW(0.9, 0.95),
peak LR 3e-3, linear warmup → cosine decay to 10% floor, grad clip 1.0, wd 0.1
on ≥2D params only.

## SFT (`sft/`)

- Chat format: ChatML (`<|im_start|>` / `<|im_end|>`); the pretrain tokenizer lacked
  role tokens, so embeddings were **resized 4096 → 4098** with new rows initialized
  from the `<|endoftext|>` embedding
- **Loss masking**: cross-entropy computed only on assistant completion tokens
- Data: `HuggingFaceTB/smoltalk`, capacity-adapted mixture (heavy code/LaTeX slices
  dropped at nano scale; refusal + format slices kept), 23,685 examples, 2 epochs
- `gate_sft.py`: post-SFT behavioral gate with **pre-registered pass/fail bars**, held-out
  refusal prompts (no train/test phrase overlap), and a base-model control — PASS requires
  the SFT model to clear the bars AND the base to fail them
- **v1 failed its bars honestly** (72% stop < 80%, 33% refusal < 66%); diagnosis traced the
  refusal miss to a low-diversity refusal slice. One pre-specified improvement sweep
  (refusal diversity 7→160 templates, +format slice, 3 epochs) → **v2 passed: 98% / 92%**.
  Full trail in `sft/AUDIT.md`, including a known limitation (mild over-refusal on benign
  prompts — the gate lacks an XSTest-style over-refusal axis)

## Methodology: guided-build audits

Each stage was executed strictly from my own research-vault recipes, with an audit log
(`pretrain/AUDIT.md`, `sft/AUDIT.md`) recording every decision's source — and every
**STALL**: a point where the documentation was insufficient and outside knowledge was
required. Each stall becomes a documentation fix. The build validates the knowledge
base as much as the knowledge base drives the build.

## Reproduce

Pinned deps: `requirements.txt` (CPU CI subset) and `requirements-ml.txt` (training/eval).
Environment notes + artifact SHA/tag instructions: `trajectory/REPRODUCIBILITY.md`.
License: MIT (`LICENSE`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # fabric + recompute CI
pytest fabric/test_fabric.py trajectory/test_recompute_c3.py -q

# Full ML stack (GPU/MPS training and scoring — not required for CI):
pip install -r requirements-ml.txt
cd pretrain && python train.py           # ~20 min on Apple Silicon
python generate.py
cd ../sft && python build_sft_data.py && python sft.py
```

Checkpoints and tokenized shards are excluded from the repo (see `.gitignore`);
trained checkpoints are release assets. Result JSONs under `trajectory/` are the
content-addressed scientific record at freeze tags — verify against
`trajectory/REPRODUCIBILITY.md` (tagged archival state ≠ “every lockfile edit is frozen”).
