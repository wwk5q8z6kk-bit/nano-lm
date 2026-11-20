"""Side-by-side v1.0 / v1.1 comparison with mechanical delta explanations.

The owner's acceptance gate for v1.1: *if the migration cannot explain every
material v1.0 → v1.1 delta, stop; no recalibration to make numbers look nicer.*
This module is the gate's instrument. For one transcript it emits both logs
(v1.0 from the frozen module, v1.1 from `benchmarks.v1_1`), computes both Class
A profiles, and for every key with a different value produces an explanation
that is **checked, not narrated**: each category below has a predicate that
must hold for the delta to count as explained. Anything that fails its
predicate is reported as `unexplained`, and the gate fails.

Categories and their predicates

  decomposition   action_count: v11 - v10 == Σ (effects_per_call - 1)
  hidden-mutation files_changed / premature_edit_rate / reproduction_before_repair
                  / verification_attempted: every v1.1 mutate target absent from
                  v1.0's mutate set comes from a call v1.0 labelled `verify`
                  (precedence hid it); the flip-causing sub-action is named
  hidden-read     files_inspected: every new read target comes from a call v1.0
                  labelled verify / mutate / execute
  hidden-read-before-edit
                  premature_edit_rate moved with no new file: a read that v1.0
                  hid inside a mutate call (`cat x && sed -i x`) now precedes
                  the edit in order; the un-blinded files are listed; no file
                  may become newly blind
  over-attributed-mutate
                  a v1.0 mutate target absent from v1.1's mutate set must be a
                  v1.1 READ target of the same call, whose v1.0 trace carried
                  write evidence — v1.0 pooled heredoc and inline evidence per
                  call and mis-attributed read-only literals (found by the
                  62-transcript rerun; a v1.0 defect, documented, not patched)
  denominator     invalid_tool_call_rate / destructive_action_rate: equal to v1.0
                  by construction (per-call, D4) — a delta here is unexplained
  repeat-rule     repeated_action_rate: v1.1 numerator == v1.0 rule applied to
                  the sequence of first siblings; denominator == call_count
  pre-state       reversion_count: v11 - v10 == reversions the v1.1 rule finds
                  minus reversions the v1.0 rule finds on the SAME v1.1 log
  scope-pairs     scope_violation_rate: every new mutation-target pair comes from
                  a split sub-action (its call was not `mutate` in v1.0)
  identical       equal values
  new-key         v1.1-only keys; not a delta

Instrument/migration evidence only. Nothing here is benchmark evidence.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.process_metrics import class_a_profile as profile_v10
from benchmarks.process_metrics import repeated_action_rate as repeated_v10
from benchmarks.process_metrics import reversion_count as reversion_v10
from benchmarks.transcript_emitter import emit as emit_v10
from benchmarks.transcript_sanitizer import iter_records
from benchmarks.v1_1.emitter import emit as emit_v11
from benchmarks.v1_1.metrics import _calls, class_a_profile as profile_v11

V10_KEYS = ("action_count", "invalid_tool_call_rate", "repeated_action_rate", "premature_edit_rate",
            "reproduction_before_repair", "verification_attempted", "reversion_count", "files_inspected",
            "files_changed", "destructive_action_rate", "scope_violation_rate", "rollback_success", "cost_totals")


@dataclass
class Delta:
    key: str
    v10: Any
    v11: Any
    category: str
    explanation: str
    explained: bool


@dataclass
class Comparison:
    name: str
    p10: dict[str, Any]
    p11: dict[str, Any]
    deltas: list[Delta] = field(default_factory=list)
    invariants: dict[str, tuple[bool, str]] = field(default_factory=dict)
    transition: dict[str, dict[str, int]] = field(default_factory=dict)
    new_mutate_targets: dict[str, str] = field(default_factory=dict)   # target -> v1.0 class of origin call
    new_read_targets: dict[str, str] = field(default_factory=dict)
    over_attributed: dict[str, str] = field(default_factory=dict)      # v1.0 mutate target that is really a read; -> call_id
    lost_mutate_targets: list[str] = field(default_factory=list)       # genuine losses (must be empty)
    unblinded: list[str] = field(default_factory=list)                 # files no longer premature: a hidden read precedes the edit
    newly_blind: list[str] = field(default_factory=list)               # must be empty
    stats10: dict[str, Any] = field(default_factory=dict)
    stats11: dict[str, Any] = field(default_factory=dict)

    @property
    def unexplained(self) -> list[Delta]:
        return [d for d in self.deltas if not d.explained]

    @property
    def changed(self) -> list[Delta]:
        return [d for d in self.deltas if d.category != "identical"]


def compare(
    name: str,
    records: Iterable[dict[str, Any]],
    *,
    verify_patterns: Sequence[re.Pattern[str]] = (),
    read_only_patterns: Sequence[re.Pattern[str]] = (),
    declared_scope: Sequence[str] | None = None,
    source_scope: Sequence[str] | None = None,
    destructive_tools: frozenset[str] = frozenset(),
) -> Comparison:
    records = list(records)
    r10 = emit_v10(records, verify_patterns=verify_patterns, read_only_patterns=read_only_patterns)
    r11 = emit_v11(records, verify_patterns=verify_patterns, read_only_patterns=read_only_patterns,
                   declared_scope=declared_scope, source_scope=source_scope)
    log10 = dict(r10.log)
    if declared_scope is not None:
        log10["declared_scope"] = list(declared_scope)   # v1.0 profile on the same declaration
    p10 = profile_v10(log10, destructive_tools=destructive_tools)
    p11 = profile_v11(r11.log, destructive_tools=destructive_tools)
    c = Comparison(name=name, p10=p10, p11=p11, stats10=r10.stats, stats11=r11.stats)

    a10, a11 = r10.log["actions"], r11.log["actions"]
    calls11 = _calls(a11)
    # v1.0 emits one action per call in the same order as v1.1's calls
    v10_class_of_call = {sibs[0]["call_id"]: a10[i]["action_class"] for i, sibs in enumerate(calls11)} if len(a10) == len(calls11) else {}

    # transition matrix: v1.0 class -> multiset of v1.1 sub-action classes
    trans: dict[str, Counter] = defaultdict(Counter)
    for sibs in calls11:
        origin = v10_class_of_call.get(sibs[0]["call_id"], "?")
        for a in sibs:
            trans[origin][a["action_class"]] += 1
    c.transition = {k: dict(v) for k, v in trans.items()}

    def targets_of(actions: list[dict], cls: str) -> set[str]:
        return {t for a in actions if a["action_class"] == cls for t in a.get("targets", [])}

    mut10, mut11 = targets_of(a10, "mutate"), targets_of(a11, "mutate")
    read10, read11 = targets_of(a10, "read"), targets_of(a11, "read")
    for a in a11:
        origin = v10_class_of_call.get(a["call_id"], "?")
        if a["action_class"] == "mutate":
            for t in a.get("targets", []):
                if t not in mut10:
                    c.new_mutate_targets.setdefault(t, origin)
        if a["action_class"] == "read":
            for t in a.get("targets", []):
                if t not in read10:
                    c.new_read_targets.setdefault(t, origin)

    # v1.0 over-attribution (found by the 62-transcript rerun): v1.0 pooled
    # heredoc and inline-code evidence per CALL, so a read-only literal in a
    # `python -c` became a mutate target whenever a heredoc elsewhere in the
    # same call wrote. Predicate: every v1.0 mutate target absent from v1.1's
    # mutate set is a v1.1 READ target of the SAME call, and that call carried
    # write evidence in v1.0's trace. Anything else is a genuine loss.
    v11_reads_by_call: dict[str, set[str]] = defaultdict(set)
    for a in a11:
        if a["action_class"] == "read":
            v11_reads_by_call[a["call_id"]].update(a.get("targets", []))
    v10_write_evidence_calls = {sibs[0]["call_id"] for i, sibs in enumerate(calls11)
                                if i < len(r10.trace) and any("write-evidence" in n for n in r10.trace[i].notes)}
    v10_mut_call_of_target: dict[str, str] = {}
    for i, a in enumerate(a10):
        if a["action_class"] == "mutate" and i < len(calls11):
            for t in a.get("targets", []):
                v10_mut_call_of_target.setdefault(t, calls11[i][0]["call_id"])
    c.over_attributed = {}
    c.lost_mutate_targets = []
    for t in sorted(mut10 - mut11):
        cid = v10_mut_call_of_target.get(t, "?")
        if t in v11_reads_by_call.get(cid, set()) and cid in v10_write_evidence_calls:
            c.over_attributed[t] = cid
        else:
            c.lost_mutate_targets.append(t)

    # premature_edit_rate can move without any new file: a read v1.0 hid inside
    # a mutate call (`cat x && sed -i x`) now precedes the edit in order.
    def blind_map(actions: list[dict]) -> dict[str, bool]:
        seen: set[str] = set()
        out: dict[str, bool] = {}
        for a in sorted(actions, key=lambda x: x["seq"]):
            if a["action_class"] == "read":
                seen.update(a.get("targets", []))
            elif a["action_class"] == "mutate":
                for t in a.get("targets", []):
                    out.setdefault(t, t not in seen)
        return out
    b10, b11 = blind_map(a10), blind_map(a11)
    c.unblinded = sorted(t for t in b10 if t in b11 and b10[t] and not b11[t])
    c.newly_blind = sorted(t for t in b10 if t in b11 and not b10[t] and b11[t])

    # --- invariants (spec T4) -------------------------------------------------
    inv = c.invariants
    inv["call_count == v1.0 action_count"] = (p11["call_count"] == p10["action_count"], f"{p11['call_count']} vs {p10['action_count']}")
    # Verify presence is per call, not per sub-action: v1.0 labelled a call
    # `verify` iff it had any test-runner segment, so exactly those calls must
    # contain a verify sub-action. The sub-action COUNT may exceed the call
    # count when two test segments are separated by another effect
    # (`pytest a; git add; pytest b` → verify, execute, verify); that surplus is
    # decomposition, not a new verification, and is accounted for explicitly.
    v10_verify_calls = {cid for cid, cls in v10_class_of_call.items() if cls == "verify"}
    v11_verify_calls = {sibs[0]["call_id"] for sibs in calls11 if any(a["action_class"] == "verify" for a in sibs)}
    inv["calls containing a verify == v1.0 verify calls"] = (
        v10_verify_calls == v11_verify_calls,
        f"only-v1.0={len(v10_verify_calls - v11_verify_calls)} only-v1.1={len(v11_verify_calls - v10_verify_calls)}")
    v11_verify = sum(1 for a in a11 if a["action_class"] == "verify")
    interleaved = sum(max(0, sum(1 for a in sibs if a["action_class"] == "verify") - 1) for sibs in calls11)
    inv["verify sub-action surplus == interleaved test segments"] = (
        v11_verify - len(v10_verify_calls) == interleaved,
        f"{v11_verify} sub-actions − {len(v10_verify_calls)} calls = {v11_verify - len(v10_verify_calls)}; interleaved = {interleaved}")
    inv["every v1.0 mutate target is a v1.1 mutate target or a v1.0 over-attribution"] = (
        not c.lost_mutate_targets, f"lost={c.lost_mutate_targets[:5]} over-attributed={dict(list(c.over_attributed.items())[:5])}")
    inv["new mutate targets all originate in v1.0 verify calls"] = (
        all(o == "verify" for o in c.new_mutate_targets.values()),
        f"origins={dict(Counter(c.new_mutate_targets.values()))}")
    inv["v1.1 read targets ⊇ v1.0"] = (read10 <= read11, f"missing={sorted(read10 - read11)[:5]}")
    inv["cost_totals identical"] = (p10["cost_totals"] == p11["cost_totals"], "")
    # hidden verifies: a v1.0 non-verify call cannot contain a verify (verify always won)
    inv["no verify sub-action inside a v1.0 non-verify call"] = (
        all(trans[o].get("verify", 0) == 0 for o in trans if o != "verify"), "")

    # --- per-key deltas --------------------------------------------------------
    def add(key: str, category: str, explanation: str, explained: bool) -> None:
        c.deltas.append(Delta(key, p10.get(key), p11.get(key), category, explanation, explained))

    split_extra = sum((n - 1) * k for n, k in p11["effects_per_call"].items())
    for key in V10_KEYS:
        v10, v11 = p10.get(key), p11.get(key)
        if v10 == v11:
            add(key, "identical", "", True)
            continue
        if key == "action_count":
            add(key, "decomposition", f"Σ(effects_per_call−1) = {split_extra}; histogram {p11['effects_per_call']}", v11 - v10 == split_extra)
        elif key == "files_changed":
            new, over = c.new_mutate_targets, c.over_attributed
            ok = all(o == "verify" for o in new.values()) and not c.lost_mutate_targets and (v11 - v10 == len(new) - len(over))
            cat = "hidden-mutation" if new and not over else ("over-attributed-mutate" if over and not new else "hidden-mutation+over-attribution")
            add(key, cat, f"+{len(new)} newly visible (v1.0 class of call: {dict(list(new.items())[:6])}); −{len(over)} v1.0 over-attributed read literals {dict(list(over.items())[:4])}", ok)
        elif key == "premature_edit_rate":
            new, over = c.new_mutate_targets, c.over_attributed
            moved = bool(new) or bool(over) or bool(c.unblinded) or bool(c.newly_blind)
            ok = moved and all(o == "verify" for o in new.values()) and not c.lost_mutate_targets and not c.newly_blind
            cat = "hidden-read-before-edit" if c.unblinded and not new and not over else "hidden-mutation"
            add(key, cat, f"newly visible files {dict(list(new.items())[:4])}; over-attributed removed {list(over)[:3]}; "
                          f"files no longer blind because a hidden read precedes the edit: {c.unblinded[:6]}; newly blind: {c.newly_blind[:3]}", ok)
        elif key == "files_inspected":
            new = c.new_read_targets
            ok = bool(new) and all(o in ("verify", "mutate", "execute") for o in new.values()) and (v11 - v10 == len(new))
            add(key, "hidden-read", f"newly visible read files (v1.0 class of their call): {dict(list(new.items())[:8])}", ok)
        elif key in ("verification_attempted", "reproduction_before_repair"):
            muts = [a for a in a11 if a["action_class"] == "mutate"]
            pivot = (max(muts, key=lambda a: a["seq"]) if key == "verification_attempted" else min(muts, key=lambda a: a["seq"])) if muts else None
            origin = v10_class_of_call.get(pivot["call_id"], "?") if pivot else "?"
            ok = pivot is not None and origin != "mutate"
            add(key, "hidden-mutation", f"pivot mutation seq={pivot['seq'] if pivot else None} targets={pivot['targets'] if pivot else None} from a call v1.0 labelled `{origin}`", ok)
        elif key == "repeated_action_rate":
            first_siblings = [sibs[0] for sibs in calls11]
            v10_rule_on_calls = repeated_v10(first_siblings)
            add(key, "repeat-rule", f"v1.0 rule over first siblings = {v10_rule_on_calls}; v1.1 = {v11}; denominators {p10['action_count']} vs {p11['call_count']}",
                v10_rule_on_calls is not None and v11 is not None and abs(v10_rule_on_calls - v11) < 1e-12)
        elif key in ("invalid_tool_call_rate", "destructive_action_rate"):
            add(key, "denominator", "per-call denominator must reproduce v1.0 exactly", False)
        elif key == "reversion_count":
            # Two mechanisms, both from C2, can move this key in opposite
            # directions: pre-state seeding ADDS initial-state returns; the
            # no-op exclusion (D10) REMOVES A→A edits that v1.0 counted as a
            # return to a seen state. Check each separately, not their sum.
            v10_rule_on_v11 = reversion_v10(a11)
            non_noop = [a for a in a11 if not (a.get("action_class") == "mutate" and a.get("pre_state_sha256") is not None
                                                and a.get("pre_state_sha256") == a.get("post_state_sha256"))]
            base = reversion_v10(non_noop)
            noop_counted_by_v10 = v10_rule_on_v11 - base
            pre_state_gain = v11 - base
            ok = (v10_rule_on_v11 == v10                      # C1 adds no post-states: v1.0's rule reproduces v1.0 on the v1.1 log
                  and pre_state_gain >= 0
                  and 0 <= noop_counted_by_v10 <= p11["noop_mutations"])
            cat = "pre-state" if pre_state_gain and not noop_counted_by_v10 else ("noop-exclusion" if noop_counted_by_v10 and not pre_state_gain else "pre-state+noop-exclusion")
            add(key, cat, f"v1.0 rule on the v1.1 log = {v10_rule_on_v11}; +{pre_state_gain} initial-state returns via pre-state seeding; "
                          f"−{noop_counted_by_v10} no-op edits v1.0 counted as reversions (noop_mutations={p11['noop_mutations']})", ok)
        elif key == "scope_violation_rate":
            pairs10 = {(a["call_id"] if "call_id" in a else i, t) for i, a in enumerate(a10) if a["action_class"] == "mutate" for t in a.get("targets", [])}
            new_pairs = [(a["call_id"], t) for a in a11 if a["action_class"] == "mutate" for t in a.get("targets", [])
                         if v10_class_of_call.get(a["call_id"]) != "mutate"]
            add(key, "scope-pairs", f"{len(new_pairs)} mutation-target pairs from split sub-actions: {new_pairs[:6]}", bool(new_pairs))
        else:
            add(key, "unexplained", "no explanation category applies", False)
    for key in p11:
        if key not in p10:
            c.deltas.append(Delta(key, None, p11[key], "new-key", "v1.1-only key", True))
    for name_, (ok, detail) in inv.items():
        if not ok:
            c.deltas.append(Delta(f"invariant: {name_}", None, detail, "unexplained", "T4 invariant failed", False))
    return c


def compare_file(path: str | Path, **kwargs: Any) -> Comparison:
    return compare(Path(path).stem, iter_records(path), **kwargs)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return str(v)


def render_markdown(comparisons: Sequence[Comparison], title: str) -> str:
    lines = [f"# {title}\n",
             "> **MIGRATION / INSTRUMENT EVIDENCE, NOT MODEL EVIDENCE.** Side-by-side emission of the frozen v1.0 contract",
             "> and v1.1 on the same transcripts. Every changed value must carry a checked explanation; the gate fails on any",
             "> `unexplained` row. Nothing here is a benchmark result.\n"]
    total_unexplained = sum(len(c.unexplained) for c in comparisons)
    lines.append(f"**Gate:** {'PASS' if total_unexplained == 0 else 'FAIL'} — {total_unexplained} unexplained delta(s) across {len(comparisons)} transcript(s).\n")
    for c in comparisons:
        lines.append(f"## {c.name}\n")
        lines.append(f"calls {c.p11['call_count']} → sub-actions {c.p11['action_count']}; effects per call {c.p11['effects_per_call']}\n")
        lines.append("Invariants (spec T4):\n\n| invariant | holds | detail |\n|---|---|---|")
        for k, (ok, d) in c.invariants.items():
            lines.append(f"| {k} | {'yes' if ok else '**NO**'} | {d} |")
        lines.append("\nTransition (v1.0 call class → v1.1 sub-action classes):\n\n| v1.0 class | v1.1 sub-actions |\n|---|---|")
        for k, v in sorted(c.transition.items()):
            lines.append(f"| {k} | {json.dumps(v, sort_keys=True)} |")
        lines.append("\nChanged keys:\n\n| key | v1.0 | v1.1 | category | explanation | explained |\n|---|---|---|---|---|---|")
        for d in c.changed:
            if d.category == "new-key":
                continue
            lines.append(f"| `{d.key}` | {_fmt(d.v10)} | {_fmt(d.v11)} | {d.category} | {d.explanation} | {'yes' if d.explained else '**NO**'} |")
        idents = [d.key for d in c.deltas if d.category == "identical"]
        lines.append(f"\nIdentical: {', '.join(f'`{k}`' for k in idents)}\n")
        new = [d for d in c.deltas if d.category == "new-key"]
        lines.append("v1.1-only keys: " + ", ".join(f"`{d.key}`={_fmt(d.v11)}" for d in new) + "\n")
    return "\n".join(lines) + "\n"
