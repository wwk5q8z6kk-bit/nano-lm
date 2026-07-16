# Guided-build execution audit — Recipe ③ Post-Training on the nano base (2026-07-16)
Rule: every step follows ONLY vault instructions; every outside-the-vault reach = STALL logged.
Base model: `~/AI-builds/nano-pretrain-2026-07-15/ckpt.pt` (3.15M params, V=4096, S=512, loss 3.70/val 3.95).
Recipe under test: Build Recipe ③ — Post-Training and Alignment (+ pages it routes to).

## Stage 0 · Orientation
- Pipeline order per Recipe ③: SFT → (RM optional, skipped per "DPO lets you skip the RM") → DPO → RLVR/GRPO → safety. Eval gate between stages.
- Nano scope decision: run Stages 1, 3c, 4 fully; fold Stage 5 (safety) into the SFT refusal slice + a format-level refusal/over-refusal gate. Stage 2 skipped per the recipe's own guidance ("Reach for DPO before PPO" Build Note).

## Stage 1 · SFT — findings so far
- **Chat template**: Recipe ③ says "rendered through the chat template" but the *concrete* ChatML syntax + special-token rules live in `post-training-recipe (RV)` (§Chat template & special tokens) and `agents.md` — NOT in Recipe ③ or the main `post-training-recipe` page it links. Found in-vault by search. CLEAN-ish (observation: main page lost the RV import's chat-template/loss-masking/packing paragraphs).
- **STALL #1 (tokenizer special tokens)**: the nano tokenizer has only `<|endoftext|>` — no chat-role markers, no `<reserved_N>` headroom — even though `tokenizer-training.md` §4 + Pitfalls prescribe both. Root cause: Recipe ② §2's tokenizer table (which drove the pretrain walk) omits special tokens entirely; the instruction never surfaced on the executed path. Fix path (embedding resize + init-from-existing-token) found in-vault (`LLM.md` §Resize the Model's Token Embeddings). Executing: add `<|im_start|>`, `<|im_end|>` → V=4098, init new rows from the `<|endoftext|>` row.
- **STALL-lite #2 (dataset IDs)**: `sft-and-instruction-datasets.md` names SmolTalk/Tülu-3/Dolly but lists NO exact HF IDs — the same gap iteration-12 fixed for pretraining-corpora (`HuggingFaceFW/fineweb`). Used outside knowledge: `HuggingFaceTB/smoltalk`.
- **STALL-lite #3 (SFT LR at small scale)**: Recipe ③ gives "1e-5 to 2e-5 full FT; 1e-6 for very large" — the numbers implicitly assume ≥7B. No scaling rule for small models (pretrain table scales LR by width; SFT table doesn't). Chose 3e-4 by mirroring the ≈×0.07 pretrain→SFT ratio the 7-8B numbers imply (3e-4 pretrain → 2e-5 SFT) applied to nano's 3e-3 pretrain peak.
- Scale adaptation (not a stall): code/math/tool slices dropped at nano capacity; mixture renormalized toward general/multi-turn/format/refusal + a synthetic verifiable "repeat" slice to seed Stage 4 (synthetic fill per `synthetic-data` routing in Recipe ③).
