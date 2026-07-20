# Read-only balance diagnostic over the ALREADY-FROZEN c3_pools.json (commit 943b446).
# Does NOT re-select or modify any candidate. gen_c3_pools.py's own hard gate (token-
# count diff <=0.5, bigram-range nonoverlap, min cell size, family-cap) already PASSED
# and is the binding gate per PREREG_C3_binding_probe.md -- this script cannot fail the
# pool. It reports one covariate the hard gate does not check: per-junction-token
# corpus frequency (junction_left_right_unigram_freq), via standardized mean difference
# (SMD), across the T-avail vs T-sep contrast at each matched (B,L) stratum. Rationale:
# T-sep only requires each junction token to individually occur >=20 times in the D80
# output stream -- it does not bound how much MORE frequent T-avail's tokens might be,
# so a large SMD here would flag "T-avail candidates are also just more frequent
# words in general," a confound the frozen gate doesn't test for. This is diagnostic
# only: per the project's own math-toolkit discipline (smallest honest tool first),
# a report is added, not a solver -- re-selection only if this reveals gross imbalance.
#
# Metric: SMD = (mean_A - mean_B) / pooled_sd, pooled_sd = sqrt((sd_A^2+sd_B^2)/2).
# Convention (Cohen): |SMD| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large.
import json, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
pools = json.load(open(os.path.join(HERE, "c3_pools.json")))
assert pools["valid"] and pools["hard_gate_pass"], "frozen pool must already pass its own gate"
manifest = pools["manifest"]


def smd(a, b):
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((va + vb) / 2)
    return (ma - mb) / pooled if pooled > 0 else 0.0


def freq_values(values):
    # mean of the two junction-token unigram frequencies per candidate (symmetric
    # proxy for "how common are this value's junction tokens in general")
    out = []
    for v in values:
        lf, rf = manifest[v]["junction_left_right_unigram_freq"]
        out.append((lf + rf) / 2)
    return out


rows = []
for B in ("B-sub", "B-space"):
    for Lb in ("short", "long"):
        avail = [v for v, m in manifest.items()
                 if m["T"] == "T-avail" and m["B"] == B and m["L"] == Lb]
        sep = [v for v, m in manifest.items()
               if m["T"] == "T-sep" and m["B"] == B and m["L"] == Lb]
        if not avail or not sep:
            rows.append((B, Lb, None, len(avail), len(sep), "empty cell (already reported by hard gate)"))
            continue
        fa, fs = freq_values(avail), freq_values(sep)
        d = smd(fa, fs)
        flag = "OK" if d is None or abs(d) < 0.8 else "LARGE SMD -- investigate"
        rows.append((B, Lb, d, len(avail), len(sep), flag))

print("Junction-token frequency SMD, T-avail vs T-sep (matched B,L) -- diagnostic only:")
print(f"{'B':<8}{'L':<8}{'SMD':<10}{'n_avail':<9}{'n_sep':<7}flag")
worst = 0.0
for B, Lb, d, na, ns, flag in rows:
    dstr = f"{d:+.3f}" if d is not None else "n/a"
    print(f"{B:<8}{Lb:<8}{dstr:<10}{na:<9}{ns:<7}{flag}")
    if d is not None:
        worst = max(worst, abs(d))

print(f"\nmax |SMD| across T-contrasts: {worst:.3f} "
      f"({'no gross imbalance -- pool stands as frozen' if worst < 0.8 else 'LARGE -- review before launch'})")

json.dump({
    "diagnostic": "junction_token_frequency_smd",
    "purpose": "detect frequency confound not covered by the frozen hard gate",
    "rows": [{"B": B, "L": Lb, "smd": d, "n_avail": na, "n_sep": ns, "flag": flag}
             for B, Lb, d, na, ns, flag in rows],
    "max_abs_smd": worst,
    "blocking": False,
    "note": "This report cannot invalidate c3_pools.json; it supplements the frozen "
            "hard gate with one additional covariate check per advisor guidance.",
}, open(os.path.join(HERE, "c3_balance_report.json"), "w"), indent=1)
print("-> c3_balance_report.json")
