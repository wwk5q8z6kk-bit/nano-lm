"""Pure-stdlib tests for X07 / MM01 source-preserving dual memory (`nano.mechanisms`)."""

from __future__ import annotations

import pytest

from nano.mechanisms import DualMemoryController, EvidenceRecord


def test_dual_memory_preserves_rare_verbatim_facts_where_fast_sketch_collides():
    ctrl = DualMemoryController(fast_slots=4, slot_bytes=16)
    items = [
        ("drug_a", "Xyzalorfen-420mg-XR"),
        ("drug_b", "Clopidogrel-75mg"),
        ("drug_c", "Atorvastatin-40mg"),
        ("drug_d", "Metformin-500mg-ER"),
        ("drug_e", "Levothyroxine-50mcg"),
        ("drug_f", "Albuterol-HFA-90mcg"),
    ]
    for i, (key, val) in enumerate(items):
        ctrl.commit(
            subject_id="subj-1",
            key=key,
            verbatim_value=val,
            source_id="src-1",
            start=i * 30,
            end=i * 30 + len(val),
            valid_time=100 + i,
            observed_time=100 + i,
        )

    # Fast sketch had to collide/truncate (4 slots, 16 bytes each)
    fast_exact_hits = sum(
        ctrl.query_fast_only(subject_id="subj-1", key=k).values == (v,)
        for k, v in items
    )
    assert fast_exact_hits < len(items)

    # Dual memory retains 100% exact verbatim values and zero corruption
    for k, v in items:
        assert ctrl.verify_no_corruption(subject_id="subj-1", key=k, expected_verbatim=v) is True
    acct = ctrl.resource_accounting()
    assert acct["fast_value_capacity_bytes"] == 64
    assert acct["fast_state_bytes"] > 64
    assert acct["evidence_store_bytes"] > 0
    assert acct["index_overhead_bytes"] > 0
    assert acct["total_dual_bytes"] > acct["fast_state_bytes"]
    assert acct["empirical_corruption_rate"] == 0.0
    assert acct["fast_overwrites"] > 0


def test_supersession_conflict_and_verifiable_deletion():
    ctrl = DualMemoryController(fast_slots=8, slot_bytes=32)
    r1 = ctrl.commit(
        subject_id="subj-1", key="medication", verbatim_value="ibuprofen 200mg",
        source_id="t1", start=0, end=15, valid_time=1, observed_time=1,
    )
    # Superseding record updates value cleanly
    r2 = ctrl.commit(
        subject_id="subj-1", key="medication", verbatim_value="naproxen 500mg",
        source_id="t2", start=0, end=14, valid_time=2, observed_time=2,
        supersedes_id=r1.record_id,
    )
    q1 = ctrl.query_dual(subject_id="subj-1", key="medication")
    assert q1.status == "supported"
    assert q1.values == ("naproxen 500mg",)

    # Un-superseded conflicting denial produces explicit conflict
    ctrl.commit(
        subject_id="subj-1", key="medication", verbatim_value="no medications",
        source_id="t3", start=0, end=14, valid_time=3, observed_time=3, polarity="denied",
    )
    q_conf = ctrl.query_dual(subject_id="subj-1", key="medication")
    assert q_conf.status == "conflicting"
    assert set(q_conf.values) == {"naproxen 500mg", "no medications"}

    # Hard deletion clears both stores and issues receipt
    receipt = ctrl.delete(subject_id="subj-1", key="medication", observed_time=4)
    assert r2.record_id in receipt.deleted_record_ids
    assert receipt.bytes_freed > 0
    q_del = ctrl.query_dual(subject_id="subj-1", key="medication")
    assert q_del.status == "deleted"
    assert q_del.values == ()
    assert ctrl.query_fast_only(subject_id="subj-1", key="medication").status == "missing"


def test_cross_subject_supersession_rejected():
    ctrl = DualMemoryController()
    r1 = ctrl.commit(
        subject_id="subj-a", key="allergy", verbatim_value="penicillin",
        source_id="s1", start=0, end=10, valid_time=1, observed_time=1,
    )
    with pytest.raises(ValueError, match="cross-subject"):
        ctrl.commit(
            subject_id="subj-b", key="allergy", verbatim_value="latex",
            source_id="s2", start=0, end=5, valid_time=2, observed_time=2,
            supersedes_id=r1.record_id,
        )
    # Invalid record metadata fails fast
    with pytest.raises(ValueError):
        EvidenceRecord("id", "", "k", "v", "s", 0, 1, 1, 1)


def test_uncertainty_and_invalid_supersession_preserve_prior_evidence():
    ctrl = DualMemoryController()
    fields = dict(subject_id="s", key="k", verbatim_value="v", start=0, end=1,
                  valid_time=1, observed_time=1, polarity="uncertain")
    prior = ctrl.commit(source_id="src1", **fields)
    with pytest.raises(ValueError):
        ctrl.commit(source_id="src2", supersedes_id=prior.record_id, **dict(fields, start=-1))
    assert ctrl.query_dual(subject_id="s", key="k").record_ids == (prior.record_id,)
    ctrl.commit(source_id="src2", **fields)
    assert ctrl.query_dual(subject_id="s", key="k").status == "uncertain"
    assert not ctrl.verify_no_corruption(subject_id="s", key="k", expected_verbatim="v")


def test_fast_identity_and_utf8_capacity_survive_slot_collisions():
    ctrl = DualMemoryController(fast_slots=1, slot_bytes=5)
    fast = ctrl.fast
    fast.write("a:b", "c", "🙂🙂", "uncertain")
    assert fast.query("a", "b:c").status == "missing"
    fast.delete_key("a", "b:c")
    assert fast.query("a:b", "c").values == ("🙂",)
    assert fast.fast_state_bytes == 5 + len(b"a:bcuncertain")
    fast.write("a", "b:c", "x")
    assert fast.overwrites == 1
    assert fast.query("a:b", "c").status == "missing"
