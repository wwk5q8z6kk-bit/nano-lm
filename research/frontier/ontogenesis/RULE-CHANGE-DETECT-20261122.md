# G04 second study: a label-free rule-change detector (2026-11-22)

> **Correction (2026-11-23).** Code review found bugs. With corrected code (`CORRECTIONS-G04-20261123.md`) the detector does better: old entities revised 92.8% (not 86.9%), false alarms 23 (not 48). Verdicts are unchanged. Figures below are the original run.


`rule_change_detect.py`. Threshold chosen on development seed 1 and declared tuned; predictions Y1–Y5 and scorer committed in `4159de4` before any registered run. Registered seeds 7201–7203 are new. Raw files: `results/rule_change_detect/`; scored output: `results/rule_change_detect/verdict.json`. [Observed]

**Question.** The first study (`RULE-CHANGE-20261122.md`) showed that repetition separates noise from exceptions without labels, but leaves rule changes unrevised: a 45-point gap to a ceiling that knows the hidden types. Can a label-free signal detect the changes and write them into the shared rule?

**Detector.** Entities are grouped by the model's own answers on the other relations, a stand-in for the hidden type. A change is flagged when at least two new entities in a group show the same value on a relation where the group's older members are answered differently. The decoder is then retrained with the older members' addresses frozen. The targets are the new value on that relation and the model's current answers elsewhere.

**Answer.** Yes, for most changes. Old entities answer the changed rule 87% of the time with the detector, against 58% without it and 93% for the ceiling. Its many false alarms cost little here. It still misses 2–3 of the 12 changes per seed.

| Arm | Old entities revised | New entities after a change, 1–5 / 6–10 phases | Stable rule | Exceptions kept | Noise resisted |
|---|---|---|---|---|---|
| Seen twice (no detector) | 58% | 74% / 77% | 96% | 100% | 98% |
| **Seen twice + detector** | **87%** | **87% / 90%** | 95% | 100% | 97% |
| Ceiling (hidden types) | 93% | 94% / 93% | 99% | 100% | 99% |

Detector per seed: 65, 88 and 41 flags; 44, 70 and 29 of them false alarms; 3, 2 and 2 of 12 changes missed; mean delay 1.0–2.6 phases from a change to its first flag.

## Against the registered predictions

- Y1 (old entities revised ≥ 0.80, and ≥ 0.25 above seen twice): **held**, 0.87 against 0.58.
- Y2 (new entities after a change ≥ 0.85): **held**, 0.87. Per seed 0.84, 0.93 and 0.84, so it holds on the mean only.
- Y3 (false alarms harmless: stable rule within 0.02 of seen twice, exceptions ≥ 0.97, noise ≥ 0.93): **held**. Stable rule 0.945 against 0.959; exceptions 1.00; noise 0.97.
- Y4 (within 0.10 of the ceiling on old entities): **held**, 0.06 below.
- Y5 (misses at most 1 of 12 changes per seed): **failed**, 2–3 missed.

The registered reading for Y1, Y2 and Y4 holding applies: in this world, agreement among new entities that the model groups together is enough to detect most rule changes and write them into the shared rule.

## What this shows

- **A label-free change signal exists here.** New entities that the model places together, and that agree with each other against their group's history, mark a changed rule. The signal needs no types and no labels: only the model's own answers and the observations.
- **Writing the change into the shared rule is what makes it count.** The revision is the same decoder-only write as the ceiling's, with the model's own answers as targets.
- **False alarms are frequent but cheap in this world.** Between half and four fifths of the flags are false, yet the stable rule drops only 1.3 points. A false alarm retrains a group toward the value its new members showed, and the model's own answers on the other relations hold the group in place. A world where false alarms hit relations that matter, or where groups are less separable, could make them costly. Not tested.
- **Misses are the remaining gap.** Two or three of the twelve changes per seed are never flagged. Likely causes, not checked: a type whose new entities rarely show the changed relation within a few phases, or a group whose signature is split by earlier noise.

## Limits

Bilinear model, one synthetic world, three seeds, one threshold (chosen on the development seed). Groups are separable here because types are few and rules are random. In a world with overlapping types or graded rules, signatures would be less clean. Noise and exceptions are separable by design (seen once against seen twice). Not a language-model result.

## Next

1. Explain the misses per change (exposure count, signature splits). That tells whether a lower threshold on rarely shown relations would catch them at acceptable false-alarm cost.
2. A harder world: overlapping types, or rule changes that apply to part of a type.
3. Combine with the memory-line session's private-slot memory: exceptions to private slots, changes to the shared rule. That is the three-outcome routing that INVESTIGATION.md Q8 now names.
