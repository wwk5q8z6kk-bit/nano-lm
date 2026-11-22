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

---

## 2026-11-22 — Where the three-way router's stable-rule cost comes from (nano-lm-d5's study)

Run and recorded by nano-lm-d5 on `research/g04-rule-change` (PR [#204](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/204)): registration `a50fc4e` (14:32), results `eb8073f` (16:03), record `THREEWAY-COST-20261122.md`. Seeds 7401–7405. The decomposition uses a masking diagnostic suggested by this session: inference on unseen keys with the private read hidden. Figures copied from that record, not rerun here.

| Arm | Stable rule | Shared table only | Old entities revised |
|---|---|---|---|
| No routing | 81.7% | 82.2% | 18% |
| Oracle | 66.5% | 69.4% | 98% |
| Oracle, private table ×16 | 69.2% | 69.4% | 98% |
| Oracle, world without rule changes | 76.9% | 81.5% | — |
| Three-way | 71.8% | 74.3% | 71% |
| Three-way, private table ×16 | 73.6% | 73.8% | 73% |

- C1–C4 held (5/5); C5 (a larger private table lifts the router's stable rule by at least the registered margin) failed narrowly, by 0.019 on 3/5 seeds.
- Two sources of cost:
  - a small one, leakage between facts in the private table, which a 16× table removes;
  - a larger one, about 0.13, in the shared table, from rewriting rules for a whole type.
- The second is the type-scale version of this line's write-pass result (W4): large writes into shared typed slots damage later inference.
- Seed-set variance is large. The router revised 86% of old entities on seeds 7301–7305 and 71% on 7401–7405. Pooled over all ten seeds: 78% (range 58–91%).

---

## 2026-11-22 — Addresses built from the data instead of given types

Branch `research/memory-saturation-factorial` (PR [#199](https://github.com/wwk5q8z6kk-bit/nano-lm/pull/199)): protocol `25acd5d` committed before any confirmatory run, results `f22c020`. Seeds 4101–4105; single shared path; the typed and random arms reused from the same seeds.

| Entity vectors | Acquisition | Inference, new entities | Corrections |
|---|---|---|---|
| typed (hidden type given) | 0.836 | 0.737 | 0.034 |
| derived from facts shown at creation | 0.893 | 0.302 | 0.034 |
| random | 0.622 | 0.106 | 0.196 |

- G1 and G2 held (5/5). Codes built from shown facts recover 0.27–0.34 of the typed–random inference gap (mean 0.31), and acquire and retain better than given types.
- **Consequence for every typed-memory result in this line, including the joint three-way studies:** the given types overstate inference, by about a factor of three here, for a learner that must discover them.
- Development seed only: a greedy online clustering of entities into types collapsed (3–6 clusters, purity 0.15–0.24). With frozen addresses an early wrong grouping cannot be revised. Revisable, learned addresses are the next problem (KEYSTORE in `LEDGER.md`; its earlier straight-through pilot failed).
- Limits: one simple code, not tuned; 3 of 8 relations shown per entity; synthetic world. Not evidence about a language model.
