# Result — cross-model control: a refuted hypothesis and a confounded observation

**2026-08-06.** EXPLORATORY. Selects nothing, gates nothing, trains nothing.
Local MLX, $0. Artifacts: `crossmodel_direct.json`, `crossmodel_twostage.json`.

## 1. Why this was run

`DECISION_MEMO_20260806.md` proposes H7-V — widen Nano's training vocabulary —
because every span-carrying state recovers to 95–100% on familiar wording. That
inference is sound *for Nano*, but it has a premise worth testing: that the
failure comes from Nano's narrow diet. If a model with enormous vocabulary
exposure fails the same way on the same documents, vocabulary breadth alone
would not fix it.

Control: run the identical denial arms over the identical development documents
through `mlx-community/Llama-3.2-3B-Instruct-4bit` (~230× Nano's parameters),
zero-shot, locally.

## 2. A hypothesis of mine, tested and refuted

A single-document probe found the 3B model quoting the span correctly and
paraphrasing its meaning correctly — *"has tried no medication"* — then emitting
`STATED` in the same sentence. That is Nano's exact signature: of 177 gold-absent
fields Nano mislabelled `supported`, **176 had the span exactly right**
(`FINDING_DENIAL_RECOGNITION.md`). It suggested comprehension was intact and
only the map to a discrete label was broken — and therefore that asking for
meaning first and the label last would recover accuracy.

**It does not.** Over 17 arms × 15 documents × 2 fields:

| | direct | two-stage |
|---|---|---|
| overall mean, 17 arms | **21.2%** | **23.5%** |
| DEV arm | 0.0% | 0.0% |
| per-arm deltas | — | +66.7, +40.0, −76.7, −83.3, … |

A 2.3-point overall difference with per-arm swings of ±80 in both directions is
noise, not an effect. **The two-stage decomposition hypothesis is rejected.** It
was formed on n=1 and did not survive n=30 per arm — which is the whole reason
single observations are not measurements, a lesson this cycle has now learned
twice (see `RESULT_SURFACE_HARNESS_RUN1.md` §5).

## 3. What the probe does show, and what confounds it

The 3B model's surface sensitivity on these arms is **73.3–90.0 points**, against
Nano's 59.8. Its held-out mean is 12.2–22.2% against Nano's 60.0%. It scores
**0.0%** on the two development denial phrasings under both prompt modes.

Read naively that says surface sensitivity is not a small-model artifact and is
*worse* in the larger general model — which would undercut H7-V's premise.
**That reading is not available**, for three reasons:

1. **Zero-shot versus task-trained.** Nano was trained on this exact task and
   output format. The 3B model was shown a prompt. This is not a fair head-to-head
   and must never be quoted as "Nano beats Llama-3.2-3B."
2. **The prompt is unoptimised and demonstrably load-bearing.** The same model,
   same document, answered `DENY` on an isolated line and `STATED` on the full
   transcript. Prompt phrasing moved the answer before any arm did.
3. **No few-shot control.** Two or three worked examples are the standard remedy
   for exactly this failure and were not tried.

So the honest status: **the control is inconclusive.** It neither confirms nor
refutes H7-V's premise.

## 4. What a clean version would require

Fine-tune the 3B model on Nano's training partition — same data, same task, same
output format — and run the same arms. That holds task-training constant and
varies only vocabulary exposure and scale, which is the actual question. It is
the only version of this experiment whose result could move H7-V.

Cost: local, MLX LoRA, hours not dollars. Worth doing **before** H7-V rather
than after, because a negative result there would redirect the whole experiment.

## 5. Consequence for the plan

`DECISION_MEMO_20260806.md` stands unamended. H7-V remains the recommended next
experiment: the evidence for it is Nano's own factorial (95–100% recovery on
familiar vocabulary across every span-carrying state), and nothing here
contradicts that. What this adds is a **preflight**: run §4's fine-tuned control
first. If a vocabulary-rich model still shows 70+ points of surface sensitivity
after task training, H7-V's premise is wrong and the experiment should not be run
as written.

## 6. Honest limits

- One model, one quantisation, one prompt family, 15 documents per arm.
- `DENIED` is scored only against gold-`absent` fields; the probe does not test
  whether the model over-produces `DENIED` on gold-`supported` fields, so its
  precision is unmeasured. A model answering `DENIED` always would score 100%
  here — the same degeneracy this project has caught twice, and it is **not**
  guarded against in this probe. That alone bars any positive claim about the
  3B model's competence from these numbers.
