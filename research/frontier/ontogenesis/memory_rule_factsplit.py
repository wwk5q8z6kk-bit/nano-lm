"""Fact-level retention at the end of a run, split by kind. Measurement only; used by memory_rule_recluster.py
and memory_rule_transport.py so that what dropping or replaying costs individual facts can be seen.

  regular     shown facts that follow their type's rule at creation and were never corrected
  exception   shown facts that broke their type's rule at creation and were never corrected
  corrected   shown facts corrected at least once, scored against their latest value
"""
import numpy as np


def fact_split(acc, truth0, rule, typ, shown, corrected):
    se, sr = np.nonzero(shown)
    regular = truth0[se, sr] == rule[typ[se], sr]
    corr = corrected[se, sr]
    out = {}
    for name, m in (("regular", regular & ~corr), ("exception", ~regular & ~corr), ("corrected", corr)):
        out[name] = round(acc(se[m], sr[m]), 4) if m.any() else None
        out[name + "_n"] = int(m.sum())
    return out
