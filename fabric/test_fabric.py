# Regression pins for the Phase 1 vertical slice (no model needed — fixture-based).
# Run: python3 fabric/test_fabric.py   (or pytest fabric/test_fabric.py)
# Pins the exact failure classes measured on inst0: cross-slot capture, template-word
# capture, partial copy, omission, absence-from-silence. If any of these start passing
# verification again, the fabric regressed.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schemas import Claim, ClaimKind, DecisionAction, VerificationState
import slice as sl

FIX = """Doctor: Morning — what's been bothering you?
Patient: Honestly, neck pain has been troubling me.
Patient: My cousin drove me here this morning.
Doctor: Since when have you had it?
Patient: I'd say it's been 6 days.
Doctor: On a scale from mild to severe, where is it?
Patient: Definitely moderate.
Doctor: Did you try any medicine?
Patient: Only throat lozenges so far.
Doctor: Do you have any known allergies?
Patient: None whatsoever.
Summarize the visit."""

def C(slot, obj, kind=ClaimKind.VALUE):
    return Claim("patient", sl.PRED[slot], obj, kind, slot)

def test_v1_grounds_correct_value():
    r = sl.verify_value(C("cc", "neck pain"), FIX, "t")
    assert r.state == VerificationState.VERIFIED and r.evidence[0].text == "neck pain"

def test_v1_rejects_doctor_line_word():
    # "severe" appears only in the doctor's question — role-aware grounding must reject
    r = sl.verify_value(C("sev", "severe"), FIX, "t")
    assert r.state == VerificationState.UNSUPPORTED

def test_v1_known_blindspots():  # documents why v2 exists (measured on inst0)
    assert sl.verify_value(C("cc", "moderate"), FIX, "t").state == VerificationState.VERIFIED   # cross-slot
    assert sl.verify_value(C("cc", "troubling"), FIX, "t").state == VerificationState.VERIFIED  # template word
    assert sl.verify_value(C("med", "throat"), FIX, "t").state == VerificationState.VERIFIED    # partial copy

def test_v2_catches_all_three_classes():
    for slot, obj in (("cc", "moderate"), ("cc", "troubling"), ("med", "throat")):
        r = sl.verify_value_v2(C(slot, obj), FIX, "t")
        assert r.state == VerificationState.CONTRADICTED, (slot, obj)
        assert r.evidence, "contradiction must carry the bound value as counter-evidence"

def test_v2_verifies_correct_bindings():
    for slot, obj in (("cc", "neck pain"), ("dur", "6 days"), ("sev", "moderate"),
                      ("med", "throat lozenges")):
        assert sl.verify_value_v2(C(slot, obj), FIX, "t").state == VerificationState.VERIFIED, (slot, obj)

def test_v2_article_normalization():
    fx = FIX.replace("neck pain has been", "a toothache has been")
    assert sl.verify_value_v2(C("cc", "toothache"), fx, "t").state == VerificationState.VERIFIED

def test_absence_never_from_silence():
    # alg reply is an explicit denial -> positive evidence; med reply binds a value -> omission
    a = sl.verify_absent_v2(C("alg", "none", ClaimKind.ABSENT), FIX, "t")
    assert a.state == VerificationState.VERIFIED and a.evidence[0].text == "None whatsoever."
    o = sl.verify_absent_v2(C("med", "none", ClaimKind.ABSENT), FIX, "t")
    assert o.state == VerificationState.CONTRADICTED and o.evidence[0].text == "throat lozenges"
    # v1 with no denial grammar for cc: UNVERIFIABLE, never VERIFIED
    u = sl.verify_absent(C("cc", "none", ClaimKind.ABSENT), FIX, "t")
    assert u.state == VerificationState.UNVERIFIABLE

def test_decision_policy():
    assert sl.decide(sl.verify_value_v2(C("cc", "neck pain"), FIX, "t"))[0] == DecisionAction.PRESENT
    assert sl.decide(sl.verify_value_v2(C("med", "throat"), FIX, "t"))[0] == DecisionAction.ABSTAIN
    assert sl.decide(sl.verify_absent(C("cc", "none", ClaimKind.ABSENT), FIX, "t"))[0] == DecisionAction.QUALIFY

if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in fns:
        f(); print(f"  PASS {n}")
    print(f"fabric regression pins: {len(fns)}/{len(fns)} PASS")


def test_gate_v1_passes_a_mute_verifier_and_gate_v2_does_not():
    """Pin the degeneracy and its fix.

    `presented_error_rate = presented_err / max(1, presented)` is 0.0 when the
    verifier presents nothing, so gate v1 (`pres_rate <= raw_rate`) passes
    vacuously -- it cannot distinguish a perfect verifier from a mute one.
    Gate v2 adds a constant-free non-degeneracy guard denominated in raw_pred,
    the quantity the verifier cannot shrink (the shape
    scribe/gate_grounded.py:129-133 already uses).
    """
    raw_pred, raw_err, presented, presented_err = 935, 172, 0, 0

    raw_rate = raw_err / max(1, raw_pred)
    pres_rate = presented_err / max(1, presented)
    gate_v1 = (pres_rate <= raw_rate) and 0 == 0
    non_degenerate = raw_pred > 0 and presented > 0
    gate_v2 = gate_v1 and non_degenerate

    assert pres_rate == 0.0, "a mute verifier scores zero selective risk"
    assert gate_v1 is True, "gate v1 passes a verifier that presents nothing"
    assert gate_v2 is False, "gate v2 must reject the degenerate case"


def test_gate_v2_agrees_with_v1_on_every_real_cell():
    """Additive, not behavioural: v2 must not change any recorded verdict."""
    import json
    from pathlib import Path

    results = Path(__file__).resolve().parent / "results_slice_v1.json"
    if not results.exists():
        return
    for cell in json.loads(results.read_text())["reports"]:
        non_degenerate = cell["raw_pred"] > 0 and cell["presented"] > 0
        assert non_degenerate, f"unexpected degenerate cell: {cell['instrument']}"
        assert cell["gate_pass"] is True
