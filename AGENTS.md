# AGENTS.md

## Cursor Cloud specific instructions

`nano-lm` is a pure-Python ML research repo (Python 3.12). There are **no services, no
package manager, and no database** — everything is standalone scripts run with
`python <script>.py`. Standard run commands live in `README.md` (`pretrain/`), and each
subdir's `AUDIT.md` documents its stage. Notes below are the non-obvious parts.

### Environment
- The cloud VM is **CPU-only** (no CUDA GPU, no Apple MPS). Scripts written for MPS
  (`pretrain/train.py`, `sft/model_nano.py`) fall back to `cpu` automatically via their
  `torch.backends.mps.is_available()` check, so they run unchanged — just slower.
- Dependencies (`torch tokenizers datasets numpy matplotlib`) are installed by the
  update script. `trajectory/` and `scale/` also import `transformers`/`peft`; those arms
  target Kaggle/RunPod GPUs and are not runnable end-to-end locally.

### Running scripts — cwd matters
- Scripts load `tokenizer.json`, `shard_000.npy`, and `ckpt.pt` by **relative path**, so
  each must be run **from inside its own directory** (e.g. `cd pretrain && python train.py`,
  not `python pretrain/train.py`). `pretrain/generate.py` execs the top of `train.py`, so it
  also expects `shard_000.npy` in the cwd.

### Data / checkpoints are not in the repo
- `*.pt` checkpoints and `*.npy` tokenized shards are **gitignored** and shipped as GitHub
  release assets — they are not present after a fresh clone. There is **no data-builder
  script under `pretrain/`**. To run `pretrain/train.py` from scratch you must first create
  `pretrain/shard_000.npy` (a `uint16` token array): stream a few hundred docs from
  `HuggingFaceFW/fineweb` (`sample-10BT`) via `datasets`, encode with `pretrain/tokenizer.json`
  (vocab 4096), and `np.save`. HF Hub access is anonymous (no token needed).
- `pretrain/train.py` has `STEPS=4000` hardcoded (~40 min on CPU) and only writes `ckpt.pt`
  at `step % 1000 == 0` or the final step — reach step 1000 to get a checkpoint for
  `generate.py`.

### Fast checks (no model, no network, no GPU)
- `python3 fabric/schemas.py`, `python3 fabric/test_fabric.py`, and
  `python3 trajectory/test_recompute_c3.py` are fixture-based regression tests — the quickest
  way to validate the environment.

### Known harmless quirk
- `tokenizers` can emit a `PyGILState_Release` fatal error **at interpreter shutdown** after
  a script has already finished its work and written its output; the output file is intact.
