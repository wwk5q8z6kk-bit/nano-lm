"""Unit pins for interval relations and recurrence.

`relate` is the piece everything above it trusts, so it is pinned directly
rather than only through the benchmark: all thirteen relations are exercised on
exact intervals, and the inverse law is checked on every pair. A wrong entry in
the constraint table would be invisible end-to-end — it would just look like the
world being hard — and these tests are what stop that.
"""

from __future__ import annotations

import pytest

from nano.temporal import (
    CONCURRENT,
    INVERSE,
    AllenRelation,
    Interval,
    RecurrenceCount,
    count_recurrences,
    document_order_relate,
    is_sound,
    relate,
    sharpness,
)


def _iv(label, start, end=None, horizon="2026-01-31"):
    """Days of one month, so the arithmetic stays readable."""
    fmt = lambda d: f"2026-01-{d:02d}"          # noqa: E731
    return Interval(label=label, start=fmt(start),
                    end=fmt(end) if end is not None else "",
                    open_ended=end is None, horizon=horizon)


#: One exact example per relation, so the table is checked against a reference
#: rather than against itself.
CASES = {
    AllenRelation.BEFORE:        ((1, 2), (5, 6)),
    AllenRelation.MEETS:         ((1, 5), (5, 6)),
    AllenRelation.OVERLAPS:      ((1, 5), (3, 7)),
    AllenRelation.STARTS:        ((1, 3), (1, 7)),
    AllenRelation.DURING:        ((3, 4), (1, 7)),
    AllenRelation.FINISHES:      ((3, 7), (1, 7)),
    AllenRelation.EQUALS:        ((1, 7), (1, 7)),
    AllenRelation.FINISHED_BY:   ((1, 7), (3, 7)),
    AllenRelation.CONTAINS:      ((1, 7), (3, 4)),
    AllenRelation.STARTED_BY:    ((1, 7), (1, 3)),
    AllenRelation.OVERLAPPED_BY: ((3, 7), (1, 5)),
    AllenRelation.MET_BY:        ((5, 6), (1, 5)),
    AllenRelation.AFTER:         ((5, 6), (1, 2)),
}


# ---------------------------------------------------------------------------
# The relation table is correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expected,spans", sorted(
    CASES.items(), key=lambda kv: kv[0].value))
def test_exact_intervals_determine_exactly_one_relation(expected, spans):
    a, b = _iv("a", *spans[0]), _iv("b", *spans[1])
    assert relate(a, b) == frozenset({expected})


def test_every_relation_is_reachable():
    """If a relation were unreachable the table would have a dead entry, and a
    real occurrence of it would be silently reported as something else."""
    reached = {next(iter(relate(_iv("a", *x), _iv("b", *y))))
               for x, y in CASES.values()}
    assert reached == set(AllenRelation)


@pytest.mark.parametrize("spans", list(CASES.values()))
def test_the_inverse_law_holds(spans):
    """relate(b, a) must be the inverses of relate(a, b). A wrong constraint
    almost always breaks this, which is why it is checked on every case."""
    a, b = _iv("a", *spans[0]), _iv("b", *spans[1])
    forward, backward = relate(a, b), relate(b, a)
    assert backward == frozenset(INVERSE[r] for r in forward)


def test_a_relation_set_is_never_empty():
    """Two intervals always stand in some relation. An empty set would read as
    'none of them holds', which is false of any pair."""
    for x, y in CASES.values():
        assert relate(_iv("a", *x), _iv("b", *y))


def test_concurrency_and_sequence_partition_the_relations():
    assert CONCURRENT < set(AllenRelation)
    sequential = set(AllenRelation) - CONCURRENT
    assert sequential == {AllenRelation.BEFORE, AllenRelation.AFTER,
                          AllenRelation.MEETS, AllenRelation.MET_BY}


# ---------------------------------------------------------------------------
# Imprecision widens the answer instead of picking one
# ---------------------------------------------------------------------------

def test_a_month_against_a_day_inside_it_does_not_determine_the_relation():
    """The whole point. Naming one relation here would be the same defect
    `strictly_after` exists to stop, one dimension up."""
    coarse = Interval(label="coarse", start="2026-01", end="2026-02")
    fine = Interval(label="fine", start="2026-01-15", end="2026-01-20")
    found = relate(coarse, fine)
    assert len(found) > 1
    assert sharpness(found) < 1.0


def test_disjoint_months_still_determine_the_relation():
    """Imprecision is not an excuse to give up: non-overlapping ranges still
    decide, and an over-cautious answer would be as useless as a wrong one."""
    early = Interval(label="early", start="2026-01", end="2026-01")
    late = Interval(label="late", start="2026-03", end="2026-03")
    assert relate(early, late) == frozenset({AllenRelation.BEFORE})


def test_sharpness_spans_the_full_range():
    assert sharpness(frozenset({AllenRelation.BEFORE})) == 1.0
    assert sharpness(frozenset(AllenRelation)) == 0.0


def test_an_all_relations_answer_is_sound_and_useless():
    """Manipulation check on the metric pair. Soundness alone would rate this
    perfect, which is why sharpness is never reported without it."""
    everything = frozenset(AllenRelation)
    assert all(is_sound(everything, r) for r in AllenRelation)
    assert sharpness(everything) == 0.0


# ---------------------------------------------------------------------------
# An interval has to say what it does not know
# ---------------------------------------------------------------------------

def test_an_interval_with_no_end_must_declare_itself_open():
    """A missing end and an unfinished span are different facts."""
    with pytest.raises(ValueError, match="must say it is open"):
        Interval(label="x", start="2026-01-01")
    ok = Interval(label="x", start="2026-01-01", open_ended=True,
                  horizon="2026-01-31")
    assert ok.open_ended


def test_an_interval_needs_a_start():
    with pytest.raises(ValueError, match="needs a start"):
        Interval(label="x", start="", end="2026-01-05")


def test_precision_is_visible_on_the_interval():
    assert _iv("a", 1, 2).precise
    assert not Interval(label="m", start="2026-01", end="2026-02").precise


# ---------------------------------------------------------------------------
# The control's defining property
# ---------------------------------------------------------------------------

def test_document_order_always_commits_to_one_relation():
    """Which is what makes it look decisive, and what makes it wrong. Pinned so
    the control cannot quietly acquire nuance and stop being a control."""
    a, b = _iv("a", 1, 5), _iv("b", 3, 7)
    for da, db in (("2026-02-01", "2026-02-05"), ("2026-02-05", "2026-02-01"),
                   ("2026-02-01", "2026-02-01")):
        found = document_order_relate(a, b, a_documented=da, b_documented=db)
        assert len(found) == 1
        assert sharpness(found) == 1.0


def test_document_order_ignores_when_things_actually_happened():
    """Two intervals that plainly overlap are reported as sequential because of
    when they were written down. This is RSN-TEMPORAL's named failure mode."""
    a, b = _iv("a", 1, 5), _iv("b", 3, 7)
    assert relate(a, b) == frozenset({AllenRelation.OVERLAPS})
    control = document_order_relate(a, b, a_documented="2026-02-01",
                                    b_documented="2026-02-05")
    assert control == frozenset({AllenRelation.BEFORE})
    assert not is_sound(control, AllenRelation.OVERLAPS)


# ---------------------------------------------------------------------------
# Recurrence
# ---------------------------------------------------------------------------

def test_a_repeated_report_is_one_episode_not_two():
    """The distinction between a repeated report and a genuine return is the
    whole content of recurrence."""
    assert count_recurrences("x", ["up", "up", "up"], "up").low == 1


def test_a_return_is_counted():
    assert count_recurrences("x", ["up", "down", "up"], "up").low == 2
    assert count_recurrences("x", ["up", "down", "up", "down", "up"],
                             "up").low == 3


def test_a_gap_widens_the_count_upward():
    """An unreported change could hide an episode."""
    got = count_recurrences("x", ["up", "down", "up"], "up", gaps=2)
    assert (got.low, got.high) == (2, 4)
    assert not got.certain


def test_a_contested_report_widens_the_count_downward():
    """A spurious boundary could have invented an episode. Extending only
    upward leaves every invented episode uncountable."""
    got = count_recurrences("x", ["up", "down", "up"], "up", contested=1)
    assert (got.low, got.high) == (1, 3)


def test_a_clean_record_gives_a_certain_count():
    got = count_recurrences("x", ["up", "down", "up"], "up")
    assert got.certain and got.low == 2
    assert got.describe().endswith(": 2")


def test_an_uncertain_count_says_why_it_is_uncertain():
    text = count_recurrences("x", ["up"], "up", gaps=3).describe()
    assert "between 1 and 4" in text
    assert "unreported change" in text


def test_an_uncertain_count_with_nothing_to_explain_it_is_refused():
    """A range with no corruption behind it is a guess dressed as a bound."""
    with pytest.raises(ValueError, match="a guess, not a range"):
        RecurrenceCount(label="x", low=1, high=3, gaps_in_window=0)


def test_an_inverted_range_is_refused():
    with pytest.raises(ValueError, match="low .* > high"):
        RecurrenceCount(label="x", low=3, high=1, gaps_in_window=2)


def test_the_count_never_goes_below_zero():
    got = count_recurrences("x", ["down"], "up", contested=5)
    assert got.low == 0
