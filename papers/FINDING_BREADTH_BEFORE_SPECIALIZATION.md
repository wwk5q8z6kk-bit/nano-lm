# Convergent finding — breadth first, specialize later

**Recorded 2026-08-25.** Layer 2/3 synthesis. **Not a new measurement** — it
connects this program's frozen evidence to external controlled studies that
independently reach the same conclusion.

## The claim

> A model pretrained broadly and specialized later outperforms a domain-native
> model trained from scratch on the domain alone — except under a token-rich
> budget, where from-scratch becomes competitive.

## This program already measured it

The Stage-T ladder is a controlled comparison at **matched parameter count**
between an own-stack model trained from scratch on in-domain synthetic scribe
data and Pythia, a generally-pretrained model. Diluted held-out gap
(lower is better), from `C_ADAPT_DATA_CELLS` and `C_OWNSTACK_200M_FULLFT_GATE`:

| cell | pretraining | diluted gap |
|---|---|---|
| own-stack 159M, full-FT | 200M tokens, in-domain only | **16.9 ± 1.7** |
| own-stack 159M, full-FT | **3.2B tokens** (Chinchilla) | **7.0 ± 1.0** |
| own-stack 159M, LoRA | 200M tokens | 7.1 ± 1.2 |
| own-stack 159M, LoRA | 3.2B tokens | 4.2 ± 0.9 |
| **Pythia-160M** | **general pretraining** | **3.5 ± 0.7** |

At the *same parameter count*, the generally-pretrained model reads **3.5**
against the domain-native model's **16.9** — a 4.8× gap that parameter count
does not explain. Feeding the own-stack model 16× more data (3.2B tokens) moves
it to 7.0; adding LoRA on top reaches 4.2, approaching Pythia's 3.5.

This is the program's own interaction account, already frozen: the large gap
requires *both* an under-trained base *and* full-parameter adaptation, and
removing either recovers ~10 points. Data and method are **substitutes**.

## External studies reach the same conclusion under matched budgets

- **Peña & Herbold, "Pre-Training on Software Engineering Texts"**
  (arXiv 2607.06613, 2026-07). Compares continual pre-training against
  pre-training from scratch on a domain corpus, **controlled under
  constant-token and compute-matched budgets**, evaluating both domain
  adaptation and retained general-language understanding. Verbatim: *"across
  families and sizes, reusing an existing LM dominates training a domain-native
  one from scratch … PTS pays a large and usually decisive penalty on both axes
  and becomes competitive only for small LMs under a token-rich budget."*

  The exception clause is the same one this program measured: from-scratch
  closes the gap only when given far more tokens (our 200M → 3.2B cell).

- **"Small LLMs: Pruning vs. Training from Scratch"** (arXiv 2606.14150).
  Token-matched: a pruned large parent beats random initialization; the
  advantage narrows as the token budget grows. Same direction — inherited
  breadth substitutes for tokens.

- **"Is Biomedical Specialization Still Worth It?"** (arXiv 2604.06903) casts
  doubt on domain-adaptive pretraining generally, finding it viable mainly in
  small-scale resource-constrained settings, and reports that **model merging
  is needed to mitigate the generalization trade-off**.

- **"Trade-offs in Medical LLM Adaptation"** (arXiv 2606.19266). Medical QA:
  CPT+SFT gains over SFT alone are small and often not statistically
  significant, making plain SFT on a general base a strong default.

Two independent lines — one internal at 159M on a copying task, one external
across families and sizes on SE text — agree, including on the exception.

## What this implies for the program

The nano line trains **3.15M–160M models from scratch on synthetic in-domain
data**. The convergent evidence says that is the configuration most likely to
underperform, and the causalfix wave is consistent with it: 30M from scratch
sits below the capability floor for `p1_screening_eval_v1`.

**What it does not say.** It does not say the from-scratch line was wasted. Its
purpose was mechanistic — isolating the held-out copying gap under full control
— and `C_GAP_EXISTS`, `C_DIVERSITY` and the pointer-head results came from
exactly that control. A generally-pretrained model would have confounded them.
Mechanistic isolation and capability are different objectives, and the ladder
was built for the first.

**The live question is which objective the next wave serves.** If capability,
the evidence says start from a pretrained base. If mechanism, from-scratch
control remains correct and the capability floor is not a criticism of it.

## Caveats, stated plainly

- Our internal comparison is one task (synthetic scribe copying), one instrument
  family, few runs per cell, and mixed venue. It is suggestive, not decisive.
- `C_OWNSTACK_200M_FULLFT_GATE` labels the difference **STACK-dominant** — a
  bundle of architecture, pretraining breadth, tokenizer and method. This note
  argues breadth is a major component of that bundle; it does **not** isolate it.
  Tokenizer remains unseparated, and D3.3 shows tokenizer effects are live.
- The external studies are on other domains and larger scales.

## Status

**No new claim is added to `EVIDENCE_LEDGER.md`.** This is synthesis over
existing claims plus outside work. Promoting "breadth causes the gap" to a
ledger claim would require an experiment that varies pretraining breadth alone
at fixed tokenizer, architecture and method — which no cell currently does.
