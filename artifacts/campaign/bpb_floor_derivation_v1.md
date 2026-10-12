# BPB floor re-derivation for the BPE instrument — PREREG D3.3 §7.3

**Method.** D6's lesson was that a threshold must be calibrated against the
*actual failure population it claims to detect*, not against a remembered
number. So rather than assuming BPB transfers across tokenizers, this measures
both populations under both instruments, and then tries to reconstruct the
historical failure directly.

All runs: `d_model=192, n_layers=4, n_heads=4, max_seq=512, batch=8, 40 steps`,
CPU, real `p1_distill_train_v1` corpus. Driver:
`scripts/derive_bpb_floor.py`. Raw: `bpb_floor_derivation_v1.json`.

---

## 1. Does BPB transfer across tokenizers? — **Yes, close enough**

| tokenizer | condition | init BPB | final BPB | lm nats/token | tokens/byte |
|---|---|---|---|---|---|
| hash | honest | 171.59 | **6.614** | 4.584 | 1.000 |
| hash | leaking (D1.1) | 174.23 | 7.989 | 5.538 | 1.000 |
| bpe | honest | 91.38 | **5.007** | 7.026 | 0.494 |
| bpe | leaking (D1.1) | 97.86 | 6.604 | 9.267 | 0.494 |

The per-byte construction works as intended: BPE uses 0.494 tokens per byte and
correspondingly higher nats per token (7.03 vs 4.58), and the two effects
largely cancel — honest final BPB is 6.61 (hash) vs 5.01 (bpe), the same order
of magnitude and well within the floor's margin.

> **§7.3 answer: the floor does not need a different numeric value for BPE.**
> `BPB_EARLY_FLOOR = 0.5` sits ~10× below the honest BPE reading, the same
> relationship it has to the honest hash reading.

## 2. The result that matters more — **the floor's calibration anchor cannot be reproduced**

The floor was anchored to a historical `loss ≈ 0.002` at 40 steps. Attempting to
reconstruct that signature from the recorded defect descriptions:

| D3.1 (64-char cap) | D1.1 (bidirectional) | loss @40 | BPB |
|---|---|---|---|
| no | no | 4.584 | 6.61 |
| no | **yes** | 4.925 | 7.11 |
| **yes** | no | 4.938 | 21.63 |
| **yes** | **yes** | 5.291 | 23.18 |

And with D2.2 (supervising prompt positions) added:

| D1.1 | D2.2 | loss @40 | BPB |
|---|---|---|---|
| no | no | 4.718 | 6.81 |
| **yes** | no | 3.989 | 5.76 |
| no | **yes** | 11.303 | 401.58 |
| **yes** | **yes** | 9.608 | 341.36 |

**None of the seven defect reconstructions produces anything near 0.002.** The
closest any comes is 3.99 — three orders of magnitude away.

D1.1 *is* real and moves loss in the expected direction when isolated with the
loss otherwise correct (3.989 vs 4.718, a 15% improvement from reading the
future). But that is a modest effect, not a collapse.

## 3. What this means for the gate

**The BPB floor does not detect any live single defect.** It was calibrated
against a signature this repository can no longer produce, which is exactly the
D6 failure mode recurring one level up: a threshold anchored to an anecdote
rather than to a measured failure population.

That does **not** make it useless, and it should not be removed — but its claim
must be narrowed to what it can support:

> **BPB is a catastrophic-collapse backstop, not a leak detector.** It fires
> when an objective becomes trivially solvable to the point of near-zero loss.
> It does **not** detect D1.1, D2.1, D2.2 or D3.1 at the magnitudes those
> defects actually produce. Each of those is caught exactly, and separately, by
> `assert_no_attention_leakage` + the causal-reference differential test
> (D1.1), `assert_target_present_in_loss` (D2.1/D2.2), and
> `assert_prompt_not_silently_capped` + `assert_context_budget` (D3.1/D3.3).

The exact assertions are the detectors. BPB is the net underneath them.

## 4. Scope limits — stated rather than implied

- One config (`d_model=192, 4 layers`), 40 steps. The historical figure came
  from the 30M arm (`d_model=256, 8 layers`) and possibly a different step
  count; a larger or longer run may behave differently, and D8 means init loss
  is itself width-coupled.
- The reconstructions are *my* reading of the defect descriptions. Failure to
  reproduce 0.002 may mean the descriptions are incomplete rather than that the
  original measurement was wrong. **It is not evidence that the historical
  defects were harmless** — three independent assertions confirm they were real.
- Nothing here re-opens or reinterprets any of the four frozen provenance
  classes.

## 5. Verdict on §7.3

**VERIFIED, with the claim narrowed.** The floor value needs no change for the
BPE instrument (§1). The blocking item is satisfied not by re-deriving a number
but by establishing what the number can and cannot support (§3) — and the
honest answer is that BPB was never the thing catching these defects.
