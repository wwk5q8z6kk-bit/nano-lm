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
- CLI auth: OAuth as `hassaneljesr` (no kaggle.json needed). RunPod ALSO available
  locally: /opt/homebrew/bin/runpodctl + ~/.runpod/config.toml (authenticated; peer's
  Chinchilla pod deleted, nothing billing) — check tooling before claiming a venue is
  unavailable (this claim was wrong twice: kaggle.json, runpod). Push pattern: dirs in
  scratchpad `kk/`, metadata cloned from `tpl/` (the tv2 kernel: docker_image pin +
  machine_shape NvidiaTeslaT4 — KEEP the pin for env comparability).
- Kernel slugs ending in bare `-1b` get "Notebook not found" — append `-run`.
- 160M own-stack OOMs at batch 32×512 on T4 → micro 8 × accum 4 (committed; PREREG
  pre-authorizes noting this deviation). Measured 10-11k tok/s → 200M tokens ≈ 5h;
  full run (pretrain+FT+scoring) 7.0h — fits the 12h cap single-session.
- ANY kernel that imports peft MUST `pip uninstall -y -q torchao` (Kaggle preinstalls
  0.10.0; peft's LoRA dispatcher hard-fails probing it). Documented for the Pythia
  kernels — bit again on the own-stack LoRA arm (v2, ~10 min lost). Generalize: it is
  a peft trap, not a Pythia trap.
- Batched scorer (trajectory/batched_scorer.py) validated BYTE-IDENTICAL vs native on
  BOTH anchors' inst0 (nano pre-commit, scale post-commit, same session; 0/40 string
  mismatches each) — CUDA fast path for future scoring; native stays the reference;
  re-verify once per new model size.
- ALWAYS run the ~$0 throughput probe first (`PHASE=pretrain` + tiny TARGET_TOKENS);
  it caught the OOM in 53s. `kaggle kernels status` can flicker COMPLETE mid-run —
  verify against the log timeline before believing an early exit.
- Quota used this week ≈ 16h of ~30.

## Next steps, in order (updated after the LoRA-arm launch)
1. ☑ **LoRA arm DONE** (v3; v1 mount-glob, v2 torchao — each ~10min fail-fast): diluted
   7.1±1.2, clean 29.6±3.7 → METHOD carries ~73% of the stack effect; both methods
   memorize (≈0 train loss) but only full-FT destroys the copy pathway; LoRA@160M solves
   cc (clean 0.0), alg 100 in own-stack configs (NOT universal: py-1b draw3 = 24.6; type-level n=1 caveat). Folded into paper2_draft + P1 addendum.
   CHINCHILLA CONTROL DONE (peer session, RunPod H100, ~$37): 3.2B tokens + full FT =
   7.0±1.0 / 29.4±4.0 — IDENTICAL to 200M+LoRA → data & method are SUBSTITUTES
   (interaction account replaces 73/27; papers + program updated). Base ckpt preserved
   at checkpoints/chinchilla-160m/ (gitignored, 2.5GB).
   NEXT best run: the missing factorial corner 3.2B+LoRA (~30 min on the preserved base);
   then LoRA at the anchors (~1.5h).
   DESIGN NOTE (base-matching): nano scribe.pt was finetuned from dpo.pt (chat lineage),
   scale from scale10m_pretrain.pt (raw pretrain) — the LoRA cells must use the SAME base
   per anchor (nano-LoRA from dpo.pt, scale-LoRA from scale10m_pretrain.pt; all v0.1
   release assets, downloadable in-kernel). Write a short PREREG paragraph before running.
1b. (was) **LoRA arm ⏳ RUNNING** (`nano-lm-ownstack-160m-lora` v2): peft wrap VALIDATED locally
   (98 modules, 4.028M trainables; scratchpad venv `venv-peft`, peft 0.19.1); kernel
   reuses the fullft pretrain ckpt via kernel_sources — NOTE: mounts land under
   `/kaggle/input/notebooks/...`, use a recursive glob (v1 failed fast on this, ~10 min).
   On completion: fold the cell into paper2_draft.md §3.2, archive
   `results_ownstack_v2_160m_lora.json`, interpret the 2×2.
2. **Paper 2 drafting**: skeleton EXISTS (`papers/paper2_draft.md`, commit 26c0128) with
   the LoRA cell marked ⏳; after the cell lands, tighten abstract + §4 and add related
   work by reference to P1. Remaining decompositions designed in §5 (Chinchilla control,
   tokenizer swap, slot-diversity intervention, duplicate finetune).
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
