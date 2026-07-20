# NanoScribe fabric — Phase 1 minimal vertical slice (Master Plan action #3).
# Intent -> Control -> nano-lm generator -> Claims -> Verifier -> Risk decision ->
# typed Ledger, on the existing scribe task, measured on the frozen inst0 instrument.
#
# Success gate (EMPIRICAL_FOUNDATION / vNext V1): presented-claim error rate <= the raw
# generator's error rate (the verifier must catch the known copying failures), AND 100%
# span provenance for every PRESENTed claim, AND absence never inferred from silence.
#
# Verifier design (closed synthetic world — provenance of the lexicons):
#   grounding.v1 — a VALUE claim is VERIFIED iff the exact value string appears with
#     word boundaries in a PATIENT line of the source dialogue (role-aware: doctor
#     lines never ground a patient-state claim; e.g. a wrong "severe" prediction must
#     not be grounded by the doctor's "from mild to severe" question).
#   absence.v1 — an ABSENT claim is VERIFIED iff the patient's reply to the slot's
#     question is a known denial template. Lexicon = union of the training generator's
#     P_MED_NO/P_ALG_NO (scribe/build_scribe_data_v2.py) and the eval generator's
#     variants observed across all frozen instruments ("Nothing at all.",
#     "None whatsoever."). No denial found -> UNVERIFIABLE (never VERIFIED): the
#     hard rule that lack of evidence is not evidence of absence.
import importlib.util, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from schemas import (Claim, ClaimKind, Decision, DecisionAction, EvidenceSpan,
                     VerificationResult, VerificationState, to_json)

spec = importlib.util.spec_from_file_location("ra", os.path.join(HERE, "..", "trajectory", "rescore_anchors.py"))
ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)

PRED = {"cc": "reports_cc", "dur": "has_duration", "sev": "has_severity",
        "med": "takes_med", "alg": "allergic_to"}
SLOT_Q = {"med": "Did you try any medicine?", "alg": "Do you have any known allergies?"}
DENIALS = {  # provenance: build_scribe_data_v2.py P_MED_NO/P_ALG_NO + eval-observed variants
    "med": {"No, nothing yet.", "I haven't taken anything.", "Not taking anything for it.",
            "No medication so far.", "Nothing, I wanted to check with you first.",
            "Nothing at all."},
    "alg": {"No allergies.", "Not that I know of.", "None that I'm aware of.",
            "No, no allergies.", "Nothing on record.", "None whatsoever."},
}


def patient_lines(content):
    """(line_text_without_prefix, abs_start_offset_of_text) for each Patient: line."""
    out, off = [], 0
    for ln in content.splitlines(keepends=True):
        if ln.startswith("Patient: "):
            out.append((ln[9:].rstrip("\n"), off + 9))
        off += len(ln)
    return out


def verify_value(claim, content, source_id):
    pat = re.compile(r"\b" + re.escape(claim.obj) + r"\b", re.IGNORECASE)
    for text, start in patient_lines(content):
        m = pat.search(text)
        if m:
            sp = EvidenceSpan(source_id, "patient", start + m.start(), start + m.end(),
                              content[start + m.start():start + m.end()])
            return VerificationResult(claim.claim_id, VerificationState.VERIFIED, (sp,),
                                      "grounding.v1", "verbatim in patient turn")
    return VerificationResult(claim.claim_id, VerificationState.UNSUPPORTED, (),
                              "grounding.v1", "value absent from all patient turns")


def verify_absent(claim, content, source_id):
    q = SLOT_Q.get(claim.slot)
    if q is None:  # cc/dur/sev have no denial grammar in this world
        return VerificationResult.verify_absence(claim, (), "absence.v1")
    lines = content.splitlines(keepends=True)
    off = 0
    for i, ln in enumerate(lines):
        if ln.startswith("Doctor: ") and q in ln:
            roff = off + len(ln)
            for nxt in lines[i + 1:]:
                if nxt.startswith("Patient: "):
                    reply = nxt[9:].rstrip("\n")
                    if reply in DENIALS[claim.slot]:
                        sp = EvidenceSpan(source_id, "patient", roff + 9,
                                          roff + 9 + len(reply), reply)
                        return VerificationResult.verify_absence(claim, (sp,), "absence.v1")
                    break
                roff += len(nxt)
            break
        off += len(ln)
    return VerificationResult.verify_absence(claim, (), "absence.v1")


# --- grounding.v2 / absence.v2: template-anchored answer extraction ---------------
# World-grammar-aware verifier for the closed synthetic world: the claim must equal
# the CAPTURED GROUP of the slot reply's template match (train+held template union
# from scribe/build_scribe_data.py, exec'd as a prefix — importing would run the
# generator). Catches all three failure classes literal grounding cannot:
# cross-slot capture ("moderate" as cc), template-word capture ("troubling"),
# partial copy ("throat" ⊂ "throat lozenges") — and makes CONTRADICTED first-class,
# with the actually-bound value as counter-evidence. Openly a rules-perfect reference
# extractor for this world; the slice measures the FABRIC, not verifier novelty.
_src = open(os.path.join(HERE, "..", "scribe", "build_scribe_data.py")).read()
_ns = {}
exec(compile(_src.split("def sample_tuple")[0], "bsd[templates]", "exec"), _ns)

def _tpl_re(t):
    t = t.replace("{n} {unit}", "{v}").replace("{cc}", "{v}").replace("{sev}", "{v}") \
         .replace("{med}", "{v}").replace("{alg}", "{v}")
    return re.compile(re.escape(t).replace(re.escape("{v}"), "(.+?)") + r"$")

QANCH = {"cc": _ns["D_OPEN_TRAIN"] + _ns["D_OPEN_HELD"],
         "dur": _ns["D_DUR_TRAIN"] + _ns["D_DUR_HELD"],
         "sev": _ns["D_SEV_TRAIN"] + _ns["D_SEV_HELD"],
         "med": _ns["D_MED_TRAIN"] + _ns["D_MED_HELD"],
         "alg": _ns["D_ALG_TRAIN"] + _ns["D_ALG_HELD"]}
TPL = {k: [_tpl_re(t) for t in sorted(v, key=len, reverse=True)] for k, v in {
    "cc": _ns["P_CC_TRAIN"] + _ns["P_CC_HELD"],
    "dur": _ns["P_DUR_TRAIN"] + _ns["P_DUR_HELD"],
    "sev": _ns["P_SEV_TRAIN"] + _ns["P_SEV_HELD"],
    "med": _ns["P_MED_YES_TRAIN"] + _ns["P_MED_YES_HELD"],
    "alg": _ns["P_ALG_YES_TRAIN"] + _ns["P_ALG_YES_HELD"]}.items()}
DENY_TPL = {"med": set(_ns["P_MED_NO_TRAIN"] + _ns["P_MED_NO_HELD"]),
            "alg": set(_ns["P_ALG_NO_TRAIN"] + _ns["P_ALG_NO_HELD"])}

def _norm(s):
    s = s.strip().lower()
    for art in ("a ", "an ", "the "):
        if s.startswith(art): return s[len(art):]
    return s

def _slot_reply(slot, content):
    """(reply_text, abs_offset) of the patient reply to the slot's question."""
    lines = content.splitlines(keepends=True)
    off = 0
    hit = False
    for ln in lines:
        if hit and ln.startswith("Patient: "):
            return ln[9:].rstrip("\n"), off + 9
        if ln.startswith("Doctor: ") and any(q in ln for q in QANCH[slot]):
            hit = True
        off += len(ln)
    return None, None

def _extract(slot, content, source_id):
    """-> ('value', span) | ('denial', span) | (None, None). span carries the binding."""
    reply, roff = _slot_reply(slot, content)
    if reply is None: return None, None
    if reply in DENY_TPL.get(slot, ()):
        return "denial", EvidenceSpan(source_id, "patient", roff, roff + len(reply), reply)
    for rx in TPL[slot]:
        m = rx.match(reply)
        if m:
            return "value", EvidenceSpan(source_id, "patient", roff + m.start(1),
                                         roff + m.end(1), m.group(1))
    return None, None

def verify_value_v2(claim, content, source_id):
    kind, sp = _extract(claim.slot, content, source_id)
    if kind is None:
        return VerificationResult(claim.claim_id, VerificationState.UNVERIFIABLE, (),
                                  "grounding.v2", "no template match for slot reply")
    if kind == "denial":
        return VerificationResult(claim.claim_id, VerificationState.CONTRADICTED, (sp,),
                                  "grounding.v2", "slot explicitly denied in source")
    if _norm(sp.text) == _norm(claim.obj):
        return VerificationResult(claim.claim_id, VerificationState.VERIFIED, (sp,),
                                  "grounding.v2", "equals template-bound answer")
    return VerificationResult(claim.claim_id, VerificationState.CONTRADICTED, (sp,),
                              "grounding.v2", f"bound value differs: {sp.text!r}")

def verify_absent_v2(claim, content, source_id):
    kind, sp = _extract(claim.slot, content, source_id)
    if kind == "denial":
        return VerificationResult.verify_absence(claim, (sp,), "absence.v2")
    if kind == "value":  # omission: source binds a value the model claims absent
        return VerificationResult(claim.claim_id, VerificationState.CONTRADICTED, (sp,),
                                  "absence.v2", f"source binds {sp.text!r}; not absent")
    return VerificationResult.verify_absence(claim, (), "absence.v2")

VERIFIERS = {"v1": (verify_value, verify_absent), "v2": (verify_value_v2, verify_absent_v2)}


def decide(result):
    if result.state == VerificationState.VERIFIED:
        return DecisionAction.PRESENT, "grounded in source"
    if result.state == VerificationState.CONTRADICTED:
        return DecisionAction.ABSTAIN, "contradicted by source binding"
    if result.state == VerificationState.UNSUPPORTED:
        return DecisionAction.ABSTAIN, "no grounding span; likely copy failure"
    return DecisionAction.QUALIFY, "cannot verify; surfaced as unconfirmed"


INSTRUMENTS = {"inst0": lambda: ra.inst0, **{f"m{k}": (lambda k=k: ra.fresh[k]) for k in range(5)}}

def run_slice(tag, verifier="v1", instrument="inst0", model_cache={}):
    if tag not in model_cache:
        model_cache[tag] = ra.load(tag)[0]
    m = model_cache[tag]
    items_src = INSTRUMENTS[instrument]()
    vv, va = VERIFIERS[verifier]
    ledger_path = os.path.join(HERE, f"ledger_{tag}_{instrument}_{verifier}.jsonl")
    stats = {"parsed": 0, "review": 0, "raw_pred": 0, "raw_err": 0,
             "presented": 0, "presented_err": 0, "abstained": 0, "qualified": 0,
             "caught_err": 0, "lost_correct": 0, "present_no_span": 0,
             "contradicted": 0, "absent_verified": 0, "absent_unverifiable": 0}
    t0 = time.time()
    with open(ledger_path, "w") as ledger:
        for idx, it in enumerate(items_src):
            content = it["convo"][0]["content"]
            src = f"{instrument}/{idx}"
            pids = ra.prompt_ids(content)
            out = ra.generate(m, pids)
            text = ra.tok.decode(out[len(pids):]).strip()
            mm = ra.RE.match(text)
            if not mm:
                stats["review"] += 1
                ledger.write(json.dumps({"source": src, "action": "review",
                                         "reason": "unparseable output", "raw": text}) + "\n")
                continue
            stats["parsed"] += 1
            pred = dict(zip(ra.FIELDS, [g.strip() for g in mm.groups()]))
            for f in ra.FIELDS:
                p, t = pred[f], it["tuple"][f]
                kind = ClaimKind.ABSENT if p == "none" else ClaimKind.VALUE
                claim = Claim("patient", PRED[f], p, kind, f, generator=f"nano-lm/{tag}")
                res = (va if kind == ClaimKind.ABSENT else vv)(claim, content, src)
                action, why = decide(res)
                dec = Decision(claim.claim_id, res.result_id, action, why, "risk.v1")
                dec.check_presentation_gate(res, claim)   # raises if provenance gate broken
                correct = (p == t)
                stats["raw_pred"] += 1
                stats["raw_err"] += (not correct)
                if action == DecisionAction.PRESENT:
                    stats["presented"] += 1
                    stats["presented_err"] += (not correct)
                    stats["present_no_span"] += (len(res.evidence) == 0)
                else:
                    stats["abstained" if action == DecisionAction.ABSTAIN else "qualified"] += 1
                    stats["caught_err"] += (not correct)
                    stats["lost_correct"] += correct
                stats["contradicted"] += (res.state == VerificationState.CONTRADICTED)
                if kind == ClaimKind.ABSENT and res.state != VerificationState.CONTRADICTED:
                    k = "absent_verified" if res.state == VerificationState.VERIFIED else "absent_unverifiable"
                    stats[k] += 1
                ledger.write(json.dumps({
                    "source": src, "held": it["held_values"],
                    "claim": json.loads(to_json(claim)), "result": json.loads(to_json(res)),
                    "decision": json.loads(to_json(dec)),
                    "eval_only": {"truth": t, "correct": correct}}) + "\n")
    dt = time.time() - t0
    raw_rate = stats["raw_err"] / max(1, stats["raw_pred"])
    pres_rate = stats["presented_err"] / max(1, stats["presented"])
    gate = (pres_rate <= raw_rate) and stats["present_no_span"] == 0
    rep = {"model": tag, "verifier": verifier, "instrument": instrument,
           "raw_error_rate": round(raw_rate, 4), "presented_error_rate": round(pres_rate, 4),
           "provenance_complete": stats["present_no_span"] == 0,
           "gate_pass": gate, "runtime_s": round(dt, 1), **stats}
    print(f"[{tag}/{verifier}/{instrument}] raw_err {stats['raw_err']}/{stats['raw_pred']} ({raw_rate:.1%})  "
          f"presented_err {stats['presented_err']}/{stats['presented']} ({pres_rate:.1%})  "
          f"caught {stats['caught_err']}  lost_correct {stats['lost_correct']}  "
          f"contradicted {stats['contradicted']}  "
          f"absence V/U {stats['absent_verified']}/{stats['absent_unverifiable']}  "
          f"provenance {'100%' if stats['present_no_span'] == 0 else 'INCOMPLETE'}  "
          f"GATE {'PASS' if gate else 'FAIL'}  ({dt:.0f}s)")
    return rep


if __name__ == "__main__":
    tags = sys.argv[1:] or ["nano", "scale"]
    insts = os.environ.get("FABRIC_INSTRUMENTS", "inst0").split(",")
    reps = [run_slice(t, v, ins) for t in tags for ins in insts for v in ("v1", "v2")]
    json.dump({"slice": "V1", "gates": "presented<=raw error, 100% span provenance, "
               "absence never inferred from silence", "reports": reps},
              open(os.path.join(HERE, "results_slice_v1.json"), "w"), indent=1)
    print("-> fabric/results_slice_v1.json")
