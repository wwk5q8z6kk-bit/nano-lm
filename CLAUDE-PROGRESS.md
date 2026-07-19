# nano-lm — session progress (updated 2026-07-19)

## Where the program stands (one paragraph)
Paper 1 is complete and camera-ready (13pp PDF, `papers/latex/paper1.pdf`): the held-out
value-copying gap, measured on one multi-instance instrument across a 5-rung ladder,
field-localized via the open/closed control, sharpened by the clean value-level metric
(~80–87 pts at the anchors), failure-mode-characterized (omission vs memorized
substitution), and — as of tonight — **deconfounded**: the pre-registered own-stack 160M
control ran (7h Kaggle T4) and fired **STACK-dominant** (diluted 16.9±1.7; own-stack flat
across 50× scale while Pythia-160M reads 3.5). The Pythia fieldwise runs completed too:
the entire Pythia residual is the 5-training-value allergy slot (clean cc/med 0.0, alg
83.6→100), grounding the slot-diversity hypothesis. All results immutable in
`trajectory/results_*.json`; everything pushed to origin/master.

## Key numbers (all mean ± across-instance SD, m0–m4)
| rung | diluted | clean | note |
|---|---|---|---|
| nano 3.15M | 18.3±1.3 | 87.3±2.7 | anchors re-scored, byte-exact cross-checks |
| scale 10M | 18.7±1.5 | 79.5±2.1 | |
| **own-stack 160M** | **16.9±1.7** | **66.6±5.0** | **STACK-dominant; single run; 16× under-Chinchilla** |
| pythia-160m | 3.5±0.7 | 14.7±2.1 | comparability exact (0.0); residual = alg only |
| pythia-410m | 4.2±0.9 | 17.7±3.2 | alg clean 100.0 |
| pythia-1b | [0,5] | (3rd draw: 4.5±2.6) | comparability FAILED as §5.2 predicts |

inst0 (public instance) is the hard draw at **6/6** rungs. Slot diversity ~190/18/5
(cc/med/alg) tracks solved/solved/total-failure at every scale — the per-slot
copy-vs-classify hypothesis (labeled, falsifiable) is in P1 §6.1 and RESEARCH_PROGRAM.

## Kaggle operational knowledge (hard-won tonight)
- CLI auth: OAuth as `hassaneljesr` (no kaggle.json needed). Push pattern: dirs in
  scratchpad `kk/`, metadata cloned from `tpl/` (the tv2 kernel: docker_image pin +
  machine_shape NvidiaTeslaT4 — KEEP the pin for env comparability).
- Kernel slugs ending in bare `-1b` get "Notebook not found" — append `-run`.
- 160M own-stack OOMs at batch 32×512 on T4 → micro 8 × accum 4 (committed; PREREG
  pre-authorizes noting this deviation). Measured 10-11k tok/s → 200M tokens ≈ 5h;
  full run (pretrain+FT+scoring) 7.0h — fits the 12h cap single-session.
- ALWAYS run the ~$0 throughput probe first (`PHASE=pretrain` + tiny TARGET_TOKENS);
  it caught the OOM in 53s. `kaggle kernels status` can flicker COMPLETE mid-run —
  verify against the log timeline before believing an early exit.
- Quota used this week ≈ 16h of ~30.

## Next steps, in order
1. **LoRA arm of the 2×2** (O4 completion): own-stack 160M + LoRA r=16. BEFORE spending
   quota: validate peft wrapping of the custom GPT module locally (target_modules
   ["q","k","v","o","g","u","dn"]; verify trainable-param count + a CPU forward). The
   pretrain ckpt exists in the fullft kernel's output (pull `ownstack160m_pretrain.pt`
   or mount via kernel_sources) — do NOT re-pretrain.
2. **Paper 2 drafting** (`papers/paper2_draft.md`): headline = stack-dominant flat 50×
   own-stack curve + the slot-diversity structure; decomposition experiments (LoRA arm,
   Chinchilla-control TARGET_TOKENS≈3.2B, tokenizer swap) as its experimental program.
3. **P1 submission**: replace `[Author]` placeholder in `papers/latex/paper1.tex`
   (owner identity needed), arXiv upload, then workshop/*ACL-Findings per the council
   decision. P1 §7 carries a clearly-marked post-freeze addendum re: stack-dominant.
4. Optional precision: duplicate 160M finetune (training-variance probe at the new rung);
   1B multi-seed remains rejected (bistable — interval is the honest representation).

## Governance trail
DWA council (submit-as-is + riders) → AAEA audit (validity fixes; dilution) → clean
metric → Reviewer-#2 pass (fixed a real self-contradiction) → fieldwise Kaggle arc →
deconfounder. Audit log: `papers/writing_audit.md`. Program: `papers/RESEARCH_PROGRAM.md`.
Plan: `~/.claude/plans/calm-frolicking-newell.md`.
