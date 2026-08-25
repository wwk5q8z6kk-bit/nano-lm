"""Unit pins for causal attribution.

The claim being protected is narrow and load-bearing: a documented rationale and
a temporal association must not render identically. Most of these tests exist to
make the over-claim *unconstructible* rather than merely discouraged, so the
guard survives a future call site that would rather not think about it.
"""

from __future__ import annotations

import pytest

from nano.contracts import DerivationMode, EpistemicStatus
from nano.causal import (
    CausalBasis,
    CausalLink,
    attribute_cause,
    post_hoc_attribute,
)

ACTIONS = (("iv1", "actor_00", "a", "state", "2026-02-18", 2),
           ("iv2", "actor_01", "a", "state", "2026-02-19", 1))
DOCUMENTED = {("a", "state", "2026-02-20", "up"): ("iv1", "span_doc")}


def _attr(fn=attribute_cause, value="up"):
    return fn("a", "state", value, "2026-02-20", actions=ACTIONS,
              documented_causes=DOCUMENTED)


# ---------------------------------------------------------------------------
# The over-claim is unconstructible
# ---------------------------------------------------------------------------

def test_a_documented_rationale_must_cite_the_span_that_documents_it():
    with pytest.raises(ValueError, match="claiming to be a quotation"):
        CausalLink(entity="a", attribute="state", effect_value="up",
                   effect_time="2026-02-20",
                   basis=CausalBasis.DOCUMENTED_RATIONALE,
                   candidate_cause="iv1", evidence_span_ids=())


def test_an_association_must_say_whether_it_looked_for_alternatives():
    """'None found' and 'never looked' must not render identically."""
    with pytest.raises(ValueError, match="never.*looked"):
        CausalLink(entity="a", attribute="state", effect_value="up",
                   effect_time="2026-02-20",
                   basis=CausalBasis.TEMPORAL_ASSOCIATION,
                   candidate_cause="iv1")
    ok = CausalLink(entity="a", attribute="state", effect_value="up",
                    effect_time="2026-02-20",
                    basis=CausalBasis.TEMPORAL_ASSOCIATION,
                    candidate_cause="iv1", searched_for_alternatives=True)
    assert ok.basis is CausalBasis.TEMPORAL_ASSOCIATION


def test_an_absence_cannot_name_a_candidate_cause():
    with pytest.raises(ValueError, match="smuggled in as an absence"):
        CausalLink(entity="a", attribute="state", effect_value="up",
                   effect_time="2026-02-20", basis=CausalBasis.NONE,
                   candidate_cause="iv1")


def test_a_basis_that_names_no_cause_is_refused():
    with pytest.raises(ValueError, match="names no cause"):
        CausalLink(entity="a", attribute="state", effect_value="up",
                   effect_time="2026-02-20",
                   basis=CausalBasis.TEMPORAL_ASSOCIATION,
                   searched_for_alternatives=True)


# ---------------------------------------------------------------------------
# Only one basis may be stated as a cause
# ---------------------------------------------------------------------------

def test_only_a_documented_rationale_may_be_stated_as_a_cause():
    documented = _attr()
    assert documented.basis is CausalBasis.DOCUMENTED_RATIONALE
    assert documented.may_be_stated_as_cause

    association = _attr(value="down")     # no rationale for this value
    assert association.basis is CausalBasis.TEMPORAL_ASSOCIATION
    assert not association.may_be_stated_as_cause


def test_derivation_mode_follows_the_basis():
    """A documented rationale is OBSERVED because the *statement* was observed;
    an association is INFERRED because nothing observed the link."""
    assert _attr().derivation is DerivationMode.OBSERVED
    assert _attr().epistemic_status is EpistemicStatus.DIRECT_DOCUMENTATION
    assoc = _attr(value="down")
    assert assoc.derivation is DerivationMode.INFERRED
    assert assoc.epistemic_status is EpistemicStatus.UNCERTAIN


def test_association_wording_never_asserts_causation():
    """Wording is the deliverable here, so it is pinned rather than trusted."""
    text = _attr(value="down").describe()
    assert "association only" in text
    assert "not established as the cause" in text
    assert "caused" not in text
    assert "as documented" not in text


def test_documented_wording_says_it_is_documented():
    assert "as documented" in _attr().describe()


# ---------------------------------------------------------------------------
# The value is part of the rationale's identity
# ---------------------------------------------------------------------------

def test_a_rationale_does_not_transfer_to_a_different_value():
    """Regression pin. Keyed on (entity, attribute, time) alone, a rationale
    attached to whichever value was asked about at that moment — including one
    it does not document. That is a fabricated citation wearing a
    documented-rationale badge."""
    other = attribute_cause("a", "state", "sideways", "2026-02-20",
                            actions=ACTIONS, documented_causes=DOCUMENTED)
    assert other.basis is not CausalBasis.DOCUMENTED_RATIONALE
    assert "span_doc" not in other.evidence_span_ids


def test_competing_explanations_are_carried_not_dropped():
    assoc = _attr(value="down")
    assert assoc.candidate_cause == "iv2"        # nearest by lag
    assert "iv1" in assoc.competing_explanations
    assert assoc.searched_for_alternatives


def test_no_nearby_action_yields_nothing_rather_than_a_guess():
    link = attribute_cause("zz", "state", "up", "2026-02-20", actions=ACTIONS,
                           documented_causes={})
    assert link.basis is CausalBasis.NONE
    assert link.candidate_cause == ""
    assert link.searched_for_alternatives


def test_actions_outside_the_window_are_not_candidates():
    link = attribute_cause("a", "state", "down", "2026-02-20",
                           actions=ACTIONS, documented_causes={},
                           window_days=0)
    assert link.basis is CausalBasis.NONE


# ---------------------------------------------------------------------------
# The control is a real control, and its defect is pinned
# ---------------------------------------------------------------------------

def test_the_post_hoc_control_states_an_association_as_a_cause():
    """This is the control's defining defect, asserted so the arm cannot
    silently become correct and quietly stop being a control."""
    link = _attr(fn=post_hoc_attribute, value="down")
    assert link.basis is CausalBasis.DOCUMENTED_RATIONALE
    assert link.may_be_stated_as_cause
    assert "as documented" in link.describe()


def test_the_control_can_only_do_it_by_citing_something_that_documents_nothing():
    """The tell: satisfying the documented-rationale invariant requires citing
    the action record, which records that someone acted — not why the world
    moved."""
    link = _attr(fn=post_hoc_attribute, value="down")
    assert link.evidence_span_ids == ("action_record:iv2",)
    assert "span_doc" not in link.evidence_span_ids


def test_the_control_agrees_with_the_basis_aware_arm_on_real_documentation():
    """It only differs on associations. If it differed everywhere it would be a
    strawman rather than an ablation of one distinction."""
    aware, control = _attr(), _attr(fn=post_hoc_attribute)
    assert aware.basis is control.basis
    assert aware.candidate_cause == control.candidate_cause
    assert aware.evidence_span_ids == control.evidence_span_ids


def test_both_arms_abstain_when_there_is_nothing_nearby():
    for fn in (attribute_cause, post_hoc_attribute):
        link = fn("zz", "state", "up", "2026-02-20", actions=ACTIONS,
                  documented_causes={})
        assert link.basis is CausalBasis.NONE


# ---------------------------------------------------------------------------
# Structural guard
# ---------------------------------------------------------------------------

def test_causal_attribution_cannot_reach_the_world():
    import pathlib
    source = pathlib.Path(__file__).with_name("causal.py").read_text()
    assert "nano.slw" not in source
    assert "truth_at" not in source
    assert "caused_by" not in source, (
        "the attributor must not be able to read the world's own record of "
        "what caused what")
