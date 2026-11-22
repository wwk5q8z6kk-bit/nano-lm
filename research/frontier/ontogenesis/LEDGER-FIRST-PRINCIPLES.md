# First-principles and memory-line ledger (continuation)

Continues the additive sections that the session owning `research/memory-saturation-factorial` wrote at the end of [LEDGER.md](LEDGER.md). They moved to this file on 2026-11-22, as agreed with nano-lm-ad (owner of `research/ontogenesis-phase1`), so that two sessions appending to one file stop conflicting on PR #195. Same rules: append only, every row tied to a commit, nothing above changed.

---

## 2026-11-22 — Joint three-way router (nano-lm-d5's study, using this line's memory)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): draft protocol `84587b9`, revisions after review `8110acc` and `ee2f847`, registration `77e2131` (13:22), results `b25644c` (14:07), record `THREEWAY-20261122.md`. Seeds 7301–7305. It uses this line's two-table memory, confidence gate and private 2,000-step write; the protocol review came from this session and from nano-lm-ad. Figures below are copied from that record and were not rerun here.

| Arm | Old entities revised after a rule change | Exceptions kept | Noise resisted | Stable rule |
|---|---|---|---|---|
| No routing | 15% | 33% | 92% | 83% |
| Private routing only, τ = 0.1 | 11% | 52% | 50% | 80% |
| Change detection only | 90% | 22% | 93% | 77% |
| Three-way, change detector first | 86% | 89% | 91% | 73% |
| Three-way, private routing first | 43% | 91% | 91% | 75% |
| Three-way, 200-step revision | 43% | 99% | 92% | 79% |
| Oracle (true kinds) | 98% | 99.9% | 90% | 70% |

- Z1–Z5 and Z8 held; Z6 and Z7 failed.
- One label-free router handles all three kinds of surprise in a world where rules change, where each component alone fails at one of them.
- Two points from this line, now tested in a second world:
  - The detector order matters: routing privately first holds the old rule (43% against 86%).
  - A shared rewrite needs a larger relative budget: 200 steps revises 43%.
- Cost: the stable rule falls from 83% to 73%. That session's post-hoc reading is that the cost follows the total volume written, not which facts, since the oracle loses as much (70%). That matches this line's R3. It is post hoc and untested there.
- Limits: typed addresses give the types for free, so a shared revision reaches the whole type at once; synthetic world; value tables. Not evidence about a language model.
