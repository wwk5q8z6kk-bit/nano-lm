"""Independent, bounded source/result audit. No model imports, training, or upstream writes.

Run: python v6_review/correction_theory/independent_audit.py
Uses only stdlib + NumPy to recompute scalar summaries, seeded world metadata,
and small synthetic counterexamples. Stored percentages have four-decimal precision.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1] / "evidence/upstream_v6/correction"
SRC = BASE / "sources"
OUT = BASE / "independent_audit_results.json"


def read(group, path):
    return json.loads((SRC / group / path).read_text())


def rows(group, folder, arms):
    out = {a: read(group, f"{folder}/{a}.json") for a in arms}
    seeds = None
    for a, rr in out.items():
        ss = [r["seed"] for r in rr]
        assert len(set(ss)) == len(ss), (folder, a, "duplicate seed")
        if seeds is None:
            seeds = ss
        assert ss == seeds, (folder, a, "unpaired seed order")
    return out


def avg(data, arm, key):
    return sum(r[key] for r in data[arm]) / len(data[arm])


def table(data, keys):
    return {a: {k: round(avg(data, a, k), 6) for k in keys} for a in data}


def gate(held, data, per_seed):
    seeds = [r["seed"] for r in next(iter(data.values()))]
    values = {str(seed): bool(per_seed(i)) for i, seed in enumerate(seeds)}
    return {"held_on_mean": bool(held), "seeds_meeting_bound": sum(values.values()), "per_seed": values}


def check_registered(gates, upstream, flag="held_on_mean"):
    for name, row in gates.items():
        assert row["held_on_mean"] == upstream["predictions"][name][flag], (name, "mean disagreement")
        if "seeds_meeting_bound" in upstream["predictions"][name]:
            assert row["seeds_meeting_bound"] == upstream["predictions"][name]["seeds_meeting_bound"]


def contrast(data, a, b, key):
    dd = [(ra[key] - rb[key]) * 100 for ra, rb in zip(data[a], data[b])]
    return {"a": a, "b": b, "metric": key, "mean_pp": round(sum(dd)/len(dd), 6),
            "range_pp": [round(min(dd), 6), round(max(dd), 6)],
            "paired_seed_deltas_pp": [round(x, 6) for x in dd]}


def g04_world(seed):
    """Independent reconstruction of pinned threeway.build_world random draws only."""
    phases, new, zcount, rcount, values, shown_count = 40, 64, 16, 8, 32, 3
    n = phases * new
    rng = np.random.default_rng(seed)
    rule = rng.integers(values, size=(zcount, rcount))
    typ = rng.integers(zcount, size=n)
    shown = np.zeros((n, rcount), bool)
    for e in range(n):
        shown[e, rng.choice(rcount, shown_count, replace=False)] = True
    exc = np.full((n, rcount), -1)
    for e in range(n):
        for r in np.flatnonzero(shown[e]):
            if rng.random() < 0.05:
                exc[e, r] = (rule[typ[e], r] + 1 + rng.integers(values-1)) % values
    changes = []
    for phase in (10, 20, 30):
        pairs = set()
        while len(pairs) < 4:
            pairs.add((int(rng.integers(zcount)), int(rng.integers(rcount))))
        for z, r in sorted(pairs):
            changes.append((phase, z, r, int((rule[z, r] + 1 + rng.integers(values-1)) % values)))
    rule_at = np.repeat(rule[None], phases, 0).copy()
    for phase, z, r, value in changes:
        rule_at[phase:, z, r] = value
    born = np.repeat(np.arange(phases), new)
    perfect = {}
    for a, b, label in ((1, 6, "revise_next5"), (6, 11, "revise_next10")):
        denominator, matched = 0, 0
        for p, z, r, value in changes:
            nxt = min([p2 for p2,z2,r2,_ in changes if (z2,r2)==(z,r) and p2>p] or [phases])
            mask = ((typ==z) & (born>=p+a) & (born<min(p+b,phases,nxt)) & ~shown[:,r] & (exc[:,r]<0))
            count = int(mask.sum())
            denominator += count
            matched += count * int(rule_at[-1,z,r] == value)
        perfect[label] = {"correct": matched, "denominator": denominator,
                          "accuracy": matched/denominator,
                          "false_errors_against_historical_target": denominator-matched}
    count_pairs = collections.Counter((z,r) for _,z,r,_ in changes)
    same = sum(int(rule_at[p-1,z,r]==v) for p,z,r,v in changes)
    return {"seed": seed, "changes": [list(x) for x in changes],
            "repeat_events": sum(v-1 for v in count_pairs.values()),
            "same_value_assignment_events": same,
            "perfect_final_truth_predictor_in_pinned_scorer": perfect}


def g06_world(seed):
    """Reconstruct assignments and metric denominators; no correction solver/model."""
    zcount, rcount, values, shown_count, phases, new, fix = 16, 8, 32, 3, 40, 64, 16
    rng = np.random.default_rng(seed)
    rule = rng.integers(values, size=(zcount, rcount))
    typ = rng.integers(zcount, size=phases*new)
    truth = rule[typ].copy()
    shown = np.zeros_like(truth, dtype=bool)
    for e in range(phases*new):
        shown[e, rng.choice(rcount, shown_count, replace=False)] = True
    corrected = np.zeros_like(shown)
    when = np.full_like(truth, -1)
    entity_prior = np.zeros(phases*new, bool)
    same, repeated, total, recorrect_entity, dropped_phases = 0, 0, 0, 0, []
    for ph in range(1, phases):
        oe, rr = np.nonzero(shown[:ph*new])
        pick = rng.choice(len(oe), fix, replace=False)
        fe, fr = oe[pick], rr[pick]
        ys = rng.integers(values, size=fix)
        same += int((ys == truth[fe,fr]).sum())
        repeated += int(corrected[fe,fr].sum())
        recorrect_entity += int(entity_prior[fe].sum())
        total += fix
        truth[fe,fr] = ys
        corrected[fe,fr] = True
        when[fe,fr] = ph
        prior_rows = np.flatnonzero(entity_prior[:ph*new])
        old_bug_rows = np.flatnonzero(corrected[:ph*new].any(1) & (when[:ph*new].max(1)<ph))
        dropped_phases.append(int(len(set(prior_rows)-set(old_bug_rows))))
        entity_prior[fe] = True
    e,r = np.nonzero(shown)
    age = (phases-1) - np.repeat(np.arange(phases),new)[e]
    cold = corrected[e,r] & ((phases-1)-when[e,r]>=20)
    regular = ~corrected[e,r]
    hit = corrected.any(1)[e]
    return {"seed": seed, "entities": phases*new, "shown_facts": int(shown.sum()),
            "assignment_events": total, "same_value_assignments": same,
            "actual_value_change_events": total-same,
            "repeat_fact_assignment_events": repeated,
            "unique_assigned_fact_keys": int(corrected.sum()),
            "corrected_old_denominator": int(cold.sum()),
            "rule_old_corrected_denominator": int((regular & (age>=20) & hit).sum()),
            "rule_old_clean_denominator": int((regular & (age>=20) & ~hit).sum()),
            "infer_late_denominator": 10*new*(rcount-shown_count),
            "reassigned_prior_corrected_entity_events": recorrect_entity,
            "entity_phase_protection_omissions_fixed_by_owm_ent_patch": sum(dropped_phases)}


def main():
    manifest = json.loads((BASE / "source_manifest.json").read_text())
    for e in manifest:
        content = (SRC/e["group"]/e["name"]).read_bytes()
        assert hashlib.sha256(content).hexdigest()==e["sha256"]
        assert hashlib.sha1(b"blob "+str(len(content)).encode()+b"\0"+content).hexdigest()==e["git_blob_sha"]
    results = {"classification": "INDEPENDENT_ARITHMETIC_AND_SYNTHETIC_DIAGNOSTICS_NOT_MODEL_TRAINING",
               "verified_source_files": len(manifest), "source_bytes": sum(e["bytes"] for e in manifest)}
    metrics = ["infer_stable","infer_stable_shared_only","revise_old","revise_next5","exception_kept","noise_resisted","flags","false_alarms","missed"]
    nf = "results_v2/threeway_narrow"
    nn = rows("g04", nf, ["none","oracle","oracle_s02","oracle_s035","threeway","threeway_s02","threeway_fixed"])
    nm = lambda a,k: avg(nn,a,k)
    npv = lambda a,k,i: nn[a][i][k]
    so,ro,st = "infer_stable_shared_only","revise_old","infer_stable"
    ng = {
      "N1":gate(nm("oracle_s02",so)-nm("oracle",so)>=.06,nn,lambda i: npv("oracle_s02",so,i)-npv("oracle",so,i)>=.06),
      "N2":gate(nm("oracle_s02",ro)>=.88,nn,lambda i: npv("oracle_s02",ro,i)>=.88),
      "N3":gate(nm("oracle_s035",ro)<nm("oracle_s02",ro) and nm("oracle_s035",so)>=nm("oracle_s02",so),nn,lambda i:npv("oracle_s035",ro,i)<npv("oracle_s02",ro,i) and npv("oracle_s035",so,i)>=npv("oracle_s02",so,i)),
      "N4":gate(nm("threeway_s02",so)-nm("threeway",so)>=.04 and nm("threeway",ro)-nm("threeway_s02",ro)<=.10,nn,lambda i:npv("threeway_s02",so,i)-npv("threeway",so,i)>=.04 and npv("threeway",ro,i)-npv("threeway_s02",ro,i)<=.10),
      "N5":gate(nm("none",st)-nm("threeway_fixed",st)<=.06 and nm("threeway_fixed",ro)>=.60,nn,lambda i:npv("none",st,i)-npv("threeway_fixed",st,i)<=.06 and npv("threeway_fixed",ro,i)>=.60)
    }
    check_registered(ng,read("g04",nf+"/verdict.json"))
    results["g04_narrow"]={"gates":ng,"means":table(nn,metrics)}
    cf="results_v2/threeway_cooldown"
    cc=rows("g04",cf,["none","threeway","threeway_cd","threeway_s02","threeway_s02_cd"])
    cm=lambda a,k:avg(cc,a,k)
    cv=lambda a,k,i:cc[a][i][k]
    cg={
      "K1":gate(cm("threeway_cd","flags")<=.6*cm("threeway","flags"),cc,lambda i:cv("threeway_cd","flags",i)<=.6*cv("threeway","flags",i)),
      "K2":gate(cm("threeway_cd","missed")-cm("threeway","missed")<=.5 and abs(cm("threeway_cd",ro)-cm("threeway",ro))<=.05,cc,lambda i:cv("threeway_cd","missed",i)-cv("threeway","missed",i)<=1 and abs(cv("threeway_cd",ro,i)-cv("threeway",ro,i))<=.05),
      "K3":gate(cm("threeway_s02_cd",st)-cm("threeway",st)>=.03,cc,lambda i:cv("threeway_s02_cd",st,i)-cv("threeway",st,i)>=.03),
      "K4":gate(cm("threeway_s02_cd","exception_kept")-cm("threeway","exception_kept")>=.03,cc,lambda i:cv("threeway_s02_cd","exception_kept",i)-cv("threeway","exception_kept",i)>=.03),
      "K5":gate(cm("none",st)-cm("threeway_s02_cd",st)<=.08,cc,lambda i:cv("none",st,i)-cv("threeway_s02_cd",st,i)<=.08)
    }
    check_registered(cg,read("g04",cf+"/verdict.json"))
    results["g04_cooldown"]={"gates":cg,"means":table(cc,metrics),"contrasts":[contrast(cc,"threeway_s02_cd","none",st),contrast(cc,"threeway_s02_cd","threeway",ro),contrast(cc,"threeway_s02_cd","threeway_s02",st),contrast(cc,"threeway_s02_cd","threeway_s02",ro)]}
    ef="results_v2/threeway_em_cooldown"
    ee=rows("g04",ef,["em_nodet","em_narrow","em_narrow_cd"])
    em=lambda a,k:avg(ee,a,k)
    ev=lambda a,k,i:ee[a][i][k]
    eg={
      "Q1":gate(abs(em("em_narrow_cd",st)-em("em_nodet",st))<=.02,ee,lambda i:abs(ev("em_narrow_cd",st,i)-ev("em_nodet",st,i))<=.02),
      "Q2":gate(em("em_narrow_cd",ro)-em("em_nodet",ro)>=.25,ee,lambda i:ev("em_narrow_cd",ro,i)-ev("em_nodet",ro,i)>=.25),
      "Q3":gate(em("em_narrow_cd",st)-em("em_narrow",st)>=.02,ee,lambda i:ev("em_narrow_cd",st,i)-ev("em_narrow",st,i)>=.02),
      "Q4":gate(em("em_narrow_cd","exception_kept")>=.93 and abs(em("em_narrow_cd","noise_resisted")-em("em_nodet","noise_resisted"))<=.02,ee,lambda i:ev("em_narrow_cd","exception_kept",i)>=.93 and abs(ev("em_narrow_cd","noise_resisted",i)-ev("em_nodet","noise_resisted",i))<=.02)
    }
    check_registered(eg,read("g04",ef+"/verdict.json"))
    results["g04_em_cooldown"]={"gates":eg,"means":table(ee,metrics+["purity","changed_types_split"]),"contrasts":[contrast(ee,"em_narrow_cd","em_nodet",st),contrast(ee,"em_narrow_cd","em_nodet",ro),contrast(ee,"em_narrow_cd","em_narrow",st),contrast(ee,"em_narrow_cd","em_narrow",ro)]}
    for data in (cc,ee):
        for arm,rr in data.items():
            for row in rr:
                flags=row["flag_log"]
                assert len(flags)==row["flags"]
                assert sum(not f["true_change"] for f in flags)==row["false_alarms"]
                true_keys={(f["change_phase"],f["z"],f["r"]) for f in flags if f["true_change"]}
                assert 12-len(true_keys)==row["missed"]
                if arm.endswith("_cd"):
                    keys=[(f["pkey"],f["r"],f["v"]) for f in flags]
                    assert len(set(keys))==len(keys)
    results["flag_log_integrity"]="all cooldown-study counts and lifetime-key uniqueness agree with rows"
    gd="results/nullspace_durability"
    dd=rows("g06",gd,["none","owm_1","owm_1e-2","owm_1e-4","corr_replay"])
    dm=lambda a,k:round(avg(dd,a,k),4)
    reference=read("g06","results/correction_interference/constrained.json")
    dg={
      "N1":all([r[k] for r in dd["none"]]==[r[k] for r in reference] for k in ("corrected_old","corrected_now")),
      "N2":dm("owm_1e-4","corrected_old")>=.80 and dm("owm_1e-4","rule_old_corrected")>=.95,
      "N3":abs(dm("owm_1e-4","acquire_now")-dm("none","acquire_now"))<=.05,
      "N4":abs(dm("owm_1e-4","infer_late")-dm("none","infer_late"))<=.05,
      "N5":dm("corr_replay","corrected_old")>=.90 and abs(dm("corr_replay","acquire_now")-dm("none","acquire_now"))<=.02,
      "N6":all(r["room_end"] is not None and r["room_end"]<=.05 for r in dd["owm_1e-4"])
    }
    assert all(bool(v)==read("g06",gd+"/verdict.json")["predictions"][k]["held"] for k,v in dg.items())
    gm=["acquire_now","corrected_now","corrected_old","rule_old_corrected","rule_old_clean","infer_late","in_store"]
    results["g06_durability"]={"gates":dg,"means":table(dd,gm)}
    lf="results/nullspace_local"
    ll=rows("g06",lf,["none_m24","owm_all_m24","owm_ent_m24","owm_rel_m24","owm_all-a10_m24","owm_all-a100_m24","none_m96","owm_all_m96","owm_ent_m96","owm_rel_m96"])
    original=table(ll,gm)
    for arm in ("owm_ent_m24","owm_ent_m96"):
        ll[arm]=read("g06",lf+"/fixed_owm_ent/"+arm+".json")
    lm=lambda a,k:round(avg(ll,a,k),4)
    lg={
      "L1":all([r[k] for r in ll["none_m24"]]==[r[k] for r in dd["none"]] for k in ("corrected_old","infer_late")) and all(abs(lm("owm_all_m24",k)-dm("owm_1",k))<=.01 for k in ("corrected_old","infer_late")),
      "L2":lm("owm_ent_m24","infer_late")<=.80,
      "L3":lm("owm_rel_m24","corrected_old")>=.90 and lm("owm_rel_m24","infer_late")>=.85,
      "L4":lm("owm_rel_m96","corrected_old")>=.90 and abs(lm("owm_rel_m96","infer_late")-lm("none_m96","infer_late"))<=.03,
      "L5":lm("owm_all_m96","infer_late")<=.85,
      "L6":lm("owm_all_m24","corrected_old")>=lm("owm_all-a10_m24","corrected_old")>=lm("owm_all-a100_m24","corrected_old") and lm("owm_all_m24","infer_late")<=lm("owm_all-a10_m24","infer_late")<=lm("owm_all-a100_m24","infer_late")
    }
    assert all(bool(v)==read("g06",lf+"/verdict.json")["predictions"][k]["held"] for k,v in lg.items())
    results["g06_local"]={"gates_with_corrected_entity_arms":lg,"original_means":original,"corrected_means":table(ll,gm),"superseded_arms":["owm_ent_m24","owm_ent_m96"]}
    seeds=[*range(7201,7204),*range(7301,7306),*range(7401,7406),*range(7501,7506),*range(7601,7606),*range(7701,7706),*range(7801,7806)]
    world_results={str(s):g04_world(s) for s in seeds}
    for data in (nn,cc,ee):
        for rr in data.values():
            for row in rr:
                expected={(x[0],x[1],x[2]) for x in world_results[str(row["seed"])]["changes"]}
                actual={(x["phase"],x["z"],x["r"]) for x in row["detector_stage"]}
                assert expected==actual,(row["seed"],"world reconstruction mismatch")
    results["g04_world_only_reconstruction"]=world_results
    results["g06_world_only_reconstruction"]=[g06_world(s) for s in (8001,8002,8003)]
    # Tiny deterministic analytic examples, not evidence from trained runs.
    alpha=1.; A=np.eye(2); P=alpha*np.linalg.inv(A@A.T+alpha*np.eye(2))
    u=np.array([1.,0.]); dW=np.array([[1.,0.]])
    beta1,beta2,lr,epsilon=.9,.999,.003,1e-8
    momentum,second=1-beta1,1-beta2
    momentum*=beta1;second*=beta2  # second step has exactly zero gradient
    step2=-lr*(momentum/(1-beta1**2))/(math.sqrt(second/(1-beta2**2))+epsilon)
    fired=set(); requests=[("g",0,"B"),("g",0,"C"),("g",0,"B")]
    accepted=[]
    for key in requests:
        accepted.append(key not in fired);fired.add(key)
    results["synthetic_diagnostics"]={
      "finite_alpha": {"A":A.tolist(),"P":P.tolist(),"P_squared":(P@P).tolist(),"protected_logit_change":(dW@P@u).tolist(),"trace_over_m":float(np.trace(P)/2),"exact_free_nullspace_dimension":0},
      "adam": {"gradient_sequence":[1.,0.],"learning_rate":lr,"beta1":beta1,"beta2":beta2,"epsilon":epsilon,"zero_gradient_second_step_displacement":step2,"first_moment_attenuation_after_200_zero_gradient_steps":beta1**200,"observed_effect_in_upstream_trained_runs":"NOT_MEASURED"},
      "lifetime_cooldown_reversion": {"requests":requests,"accepted":accepted,"limitation":"B after C is a distinct rule epoch but suppressed by reused group/relation/value key"},
      "nonlinear_first_order": {"f":"theta squared","theta":0.,"delta":.1,"jacobian_at_theta":0.,"exact_output_change":.01},
      "historical_target_minimal": {"timeline":"rule A at phase0, B at10, C at20; entity born11; final query39","historical_target":"B","correct_final_target":"C","perfect_final_prediction":"C","pinned_historical_scorer_correct":False}
    }
    print(json.dumps(results,indent=2,allow_nan=False))



if __name__=="__main__":
    main()
