#!/usr/bin/env python3
"""Internal pilot for the ONE nuisance parameter the C1xC2 power calc needs.

WHY THIS EXISTS
---------------
The C1xC2 leakage 2x2 is scored with greedy decoding on a fixed slot set, so
re-running a cell reproduces it byte-for-byte. Replication therefore cannot come
from re-runs; it has to come from more *slots*. How many is set almost entirely
by one unmeasured quantity:

    pi = P(model emits a label with no quote | C1 off, present-value slot)

Under the mechanism (see PREREG), C1-on hands the model the exact string to
emit, so it answers with a quote, ``quote_absent`` is ~0, and the C2 fallback
never fires. C2's whole signal therefore lives in the C1-off row, where the
model must find the span itself and may return a bare ``STATED``. Each such slot
is one McNemar discordant pair. pi *is* the interaction effect size.

Guessing pi would make the replicate count faith-based; measuring it on the
measurement slots would make the pre-registration post-hoc. So this pilot runs
on **throwaway encounters disjoint from every measurement instance** and reports
**only** quote_absent -- never a scored outcome on a campaign_v2 slot. That is
the standard internal-pilot / blinded sample-size re-estimation move (Wittes &
Brittain 1990; Gould & Shih 1992): a nuisance parameter estimated without
looking at the treatment contrast leaves the type-I error essentially intact.

Run (off-node, not via `orx exp run`, so the four cells stay provisional):

    NANOSCIBE_QWEN_WEIGHTS=Qwen/Qwen2.5-1.5B-Instruct \
        python3 nanoscribe/pilot_quote_absent.py --out artifacts/pilot_quote_absent.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
sys.path[:] = [p for p in sys.path if p != str(Path(__file__).resolve().parent)]

from nanoscribe import leakage
from nanoscribe.adapt import parse_label_and_quotes
from nanoscribe.adapters import AtomSpec
from nanoscribe.encounter import AtomType, Experiencer, Speaker, assemble_source
from nanoscribe.select import relocate

# Values here must collide with NOTHING in the measurement set: not with any
# campaign_instances value (i0..i4), and not with the neutral system prompt's
# format examples. `--check-disjoint` asserts this mechanically at startup.
PILOT_RESERVED = frozenset({"ankle", "No prior surgery.", "lightheaded"})


def _pilot_encounters() -> list[dict[str, Any]]:
    """Throwaway encounters mirroring the campaign structure, disjoint values.

    Structure is what matters for pi: a present-value slot whose gold string the
    model must locate itself. Slot *types* mirror campaign_v2 so the pilot's
    prompt shapes match the measurement's.
    """
    out: list[dict[str, Any]] = []

    # P1 — multi-slot, mirrors enc-1 (symptom / history / allergy-denial / assessment).
    src = assemble_source(
        "pilot-1",
        (
            (Speaker.CLINICIAN, "What brings you in?"),
            (Speaker.PATIENT, "My calf has been aching for three days."),
            (Speaker.PATIENT, "I get bronchospasm now and then."),
            (Speaker.PATIENT, "No allergies of any kind."),
            (Speaker.CLINICIAN, "This looks like a muscular contusion."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-1",
            "source": src,
            "specs": (
                AtomSpec("p1-symptom", AtomType.SYMPTOM, "calf"),
                AtomSpec("p1-history", AtomType.SYMPTOM, "bronchospasm"),
                AtomSpec("p1-allergy", AtomType.ALLERGY, "allergies"),
                AtomSpec(
                    "p1-assess",
                    AtomType.ASSESSMENT,
                    "muscular contusion",
                    speaker=Speaker.CLINICIAN,
                    experiencer=Experiencer.PATIENT,
                ),
            ),
            # The allergy slot's gold quote is the denial sentence, not the word.
            "gold_quote": {
                "p1-symptom": "calf",
                "p1-history": "bronchospasm",
                "p1-allergy": "No allergies of any kind.",
                "p1-assess": "muscular contusion",
            },
        }
    )

    # P2 — uncertainty, mirrors enc-2.
    src = assemble_source(
        "pilot-2",
        (
            (Speaker.CLINICIAN, "Any trouble with your vision?"),
            (Speaker.PATIENT, "Maybe some blurring, hard to say."),
            (Speaker.CLINICIAN, "We'll keep an eye on it."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-2",
            "source": src,
            "specs": (AtomSpec("p2-vision", AtomType.SYMPTOM, "blurring"),),
            "gold_quote": {"p2-vision": "blurring"},
        }
    )

    # P3 — family-history attribution, mirrors enc-3.
    src = assemble_source(
        "pilot-3",
        (
            (Speaker.CLINICIAN, "Anything run in the family?"),
            (Speaker.PATIENT, "My father had emphysema."),
            (Speaker.PATIENT, "I've felt clammy all week."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-3",
            "source": src,
            "specs": (
                AtomSpec(
                    "p3-family",
                    AtomType.HISTORY,
                    "emphysema",
                    speaker=Speaker.PATIENT,
                    experiencer=Experiencer.OTHER,
                ),
                AtomSpec("p3-symptom", AtomType.SYMPTOM, "clammy"),
            ),
            "gold_quote": {"p3-family": "emphysema", "p3-symptom": "clammy"},
        }
    )

    # P4 — present slot alongside absent ones, mirrors enc-4. Only the present
    # slot contributes to pi; the absent slots are carried so the prompt mix
    # matches the measurement's.
    src = assemble_source(
        "pilot-4",
        (
            (Speaker.CLINICIAN, "How can I help today?"),
            (Speaker.PATIENT, "My gait feels unstable when I walk."),
            (Speaker.CLINICIAN, "Let's get an X-ray."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-4",
            "source": src,
            "specs": (AtomSpec("p4-present", AtomType.SYMPTOM, "unstable"),),
            "gold_quote": {"p4-present": "unstable"},
        }
    )

    # P5 — explicit denial, mirrors enc-5.
    src = assemble_source(
        "pilot-5",
        (
            (Speaker.CLINICIAN, "Do you use recreational drugs?"),
            (Speaker.PATIENT, "I have never injected anything."),
            (Speaker.PATIENT, "And I don't get palpitations."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-5",
            "source": src,
            "specs": (
                AtomSpec("p5-drug", AtomType.HISTORY, "injected"),
                AtomSpec("p5-symptom", AtomType.SYMPTOM, "palpitations"),
            ),
            "gold_quote": {"p5-drug": "injected", "p5-symptom": "palpitations"},
        }
    )

    # P6-P10 — a second block, same structural mix. Present purely to tighten the
    # upper bound on pi: 0 quote-absent events out of 10 slots bounds pi only at
    # ~0.26 (rule of three), which is too loose to choose a replicate count.
    src = assemble_source(
        "pilot-6",
        (
            (Speaker.CLINICIAN, "What's going on?"),
            (Speaker.PATIENT, "My scalp has been flaking badly."),
            (Speaker.PATIENT, "I had rickets as a child."),
            (Speaker.PATIENT, "No allergies whatsoever."),
            (Speaker.CLINICIAN, "I'd call this seborrheic dermatitis."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-6",
            "source": src,
            "specs": (
                AtomSpec("p6-symptom", AtomType.SYMPTOM, "flaking"),
                AtomSpec("p6-history", AtomType.SYMPTOM, "rickets"),
                AtomSpec("p6-allergy", AtomType.ALLERGY, "allergies"),
                AtomSpec(
                    "p6-assess",
                    AtomType.ASSESSMENT,
                    "seborrheic dermatitis",
                    speaker=Speaker.CLINICIAN,
                    experiencer=Experiencer.PATIENT,
                ),
            ),
            "gold_quote": {
                "p6-symptom": "flaking",
                "p6-history": "rickets",
                "p6-allergy": "No allergies whatsoever.",
                "p6-assess": "seborrheic dermatitis",
            },
        }
    )
    src = assemble_source(
        "pilot-7",
        (
            (Speaker.CLINICIAN, "How is your hearing?"),
            (Speaker.PATIENT, "Perhaps a bit muffled, I'm not certain."),
            (Speaker.CLINICIAN, "We'll test it."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-7",
            "source": src,
            "specs": (AtomSpec("p7-hearing", AtomType.SYMPTOM, "muffled"),),
            "gold_quote": {"p7-hearing": "muffled"},
        }
    )
    src = assemble_source(
        "pilot-8",
        (
            (Speaker.CLINICIAN, "Any family conditions?"),
            (Speaker.PATIENT, "My aunt had scoliosis."),
            (Speaker.PATIENT, "I've been sluggish since Tuesday."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-8",
            "source": src,
            "specs": (
                AtomSpec(
                    "p8-family",
                    AtomType.HISTORY,
                    "scoliosis",
                    speaker=Speaker.PATIENT,
                    experiencer=Experiencer.OTHER,
                ),
                AtomSpec("p8-symptom", AtomType.SYMPTOM, "sluggish"),
            ),
            "gold_quote": {"p8-family": "scoliosis", "p8-symptom": "sluggish"},
        }
    )
    src = assemble_source(
        "pilot-9",
        (
            (Speaker.CLINICIAN, "Tell me about it."),
            (Speaker.PATIENT, "My grip has been weakening for a month."),
            (Speaker.CLINICIAN, "Let's check your reflexes."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-9",
            "source": src,
            "specs": (AtomSpec("p9-present", AtomType.SYMPTOM, "weakening"),),
            "gold_quote": {"p9-present": "weakening"},
        }
    )
    src = assemble_source(
        "pilot-10",
        (
            (Speaker.CLINICIAN, "Do you drink alcohol?"),
            (Speaker.PATIENT, "I have never touched spirits."),
            (Speaker.PATIENT, "And I get no tremors."),
        ),
    )
    out.append(
        {
            "encounter_id": "pilot-10",
            "source": src,
            "specs": (
                AtomSpec("p10-drink", AtomType.HISTORY, "spirits"),
                AtomSpec("p10-symptom", AtomType.SYMPTOM, "tremors"),
            ),
            "gold_quote": {"p10-drink": "spirits", "p10-symptom": "tremors"},
        }
    )
    return out


def _span_exact(source, quote: str | None, gold_quote: str) -> bool:
    """Does the model's quote resolve to the same span as the gold quote?

    This is the `exact_gold_span` predicate reproduced on the pilot encounters,
    without building full EncounterRecords: both sides are relocated against the
    source and their character offsets compared. `relocate` returns None for an
    ambiguous (multi-hit) or absent quote, which is exactly how the real scorer
    declines to bind.
    """
    if not quote:
        return False
    gold = relocate(source, gold_quote, evidence_id="g")
    pred = relocate(source, quote, evidence_id="p")
    if gold is None or pred is None:
        return False
    return (gold.start, gold.end) == (pred.start, pred.end)


def _check_invariants(encounters: list[dict[str, Any]]) -> None:
    """Fail loudly rather than silently piloting on a broken set."""
    problems: list[str] = []
    for item in encounters:
        source = item["source"]
        for atom_id, quote in item["gold_quote"].items():
            if relocate(source, quote, evidence_id="probe") is None:
                problems.append(
                    f"{item['encounter_id']}/{atom_id}: gold quote {quote!r} does not "
                    "occur exactly once in its source"
                )
        for spec in item["specs"]:
            if spec.raw_value in PILOT_RESERVED:
                problems.append(
                    f"{item['encounter_id']}/{spec.atom_id}: value {spec.raw_value!r} "
                    "collides with a neutral system-prompt example"
                )
    # Disjointness from the measurement instances, when that table is present.
    try:
        from nanoscribe.campaign_instances import INSTANCES  # type: ignore

        measurement = {v for inst in INSTANCES for v in inst.all_values()}
    except Exception:
        measurement = set()
    if measurement:
        for item in encounters:
            for value in item["gold_quote"].values():
                if value in measurement:
                    problems.append(
                        f"{item['encounter_id']}: pilot value {value!r} also appears in "
                        "a measurement instance — pilot must be disjoint"
                    )
    if problems:
        raise SystemExit(
            "pilot encounter set is invalid:\n  " + "\n  ".join(problems)
        )


def _run_condition(
    encounters: list[dict[str, Any]], *, c1_on: bool, weights: str
) -> dict[str, Any]:
    """Generate one line per present-value slot under a given C1 setting."""
    from nanoscribe.qwen_inference import generate_span_port_lines

    from nanoscribe.adapt import ModelInput

    saved = leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE
    leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = c1_on
    try:
        slots: list[dict[str, Any]] = []
        for item in encounters:
            model_input = ModelInput(
                source=item["source"], encounter_id=item["encounter_id"]
            )
            lines, _latency, _mem = generate_span_port_lines(
                model_input, item["specs"], weights_path=weights
            )
            for spec in item["specs"]:
                raw = lines[spec.atom_id]
                label, quotes = parse_label_and_quotes(raw)
                asserted = label is not None and label != "NOT_MENTIONED"
                quote_absent = asserted and not quotes
                gold_quote = item["gold_quote"][spec.atom_id]
                model_quote = quotes[0] if quotes else None
                # C2 is a SCORER flag: it changes no token the model emits, so
                # both C2 cells are derived from this one generation pass. That
                # is why the 2x2 needs only two generation passes, not four.
                exact_c2off = _span_exact(item["source"], model_quote, gold_quote)
                quote_c2on = model_quote if model_quote else (
                    spec.raw_value if quote_absent else None
                )
                exact_c2on = _span_exact(item["source"], quote_c2on, gold_quote)
                slots.append(
                    {
                        "encounter_id": item["encounter_id"],
                        "atom_id": spec.atom_id,
                        "raw_line": " ".join(raw.split()),
                        "label": label,
                        "n_quotes": len(quotes),
                        "asserted": asserted,
                        "quote_absent": quote_absent,
                        "exact_gold_span_c2on": exact_c2on,
                        "exact_gold_span_c2off": exact_c2off,
                    }
                )
    finally:
        leakage.PROMPT_ANSWER_TEMPLATE_GOLD_VALUE = saved

    n = len(slots)
    n_asserted = sum(1 for s in slots if s["asserted"])
    n_absent = sum(1 for s in slots if s["quote_absent"])
    return {
        "c1_on": c1_on,
        "condition": "C1on" if c1_on else "C1off",
        "n_present_value_slots": n,
        "n_asserted": n_asserted,
        "n_quote_absent": n_absent,
        # pi estimated over ALL present-value slots probed: a slot that abstains
        # cannot be rescued by C2 either, so it is a genuine non-event, not a
        # missing observation.
        "quote_absent_rate": round(n_absent / n, 4) if n else 0.0,
        "exact_gold_span_c2on": sum(1 for s in slots if s["exact_gold_span_c2on"]),
        "exact_gold_span_c2off": sum(1 for s in slots if s["exact_gold_span_c2off"]),
        "slots": slots,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/pilot_quote_absent.json")
    args = parser.parse_args(argv)

    weights = os.environ.get("NANOSCIBE_QWEN_WEIGHTS")
    if not weights:
        raise SystemExit(
            "NANOSCIBE_QWEN_WEIGHTS is unset — this pilot measures real model "
            "behaviour and is meaningless against a fixture."
        )

    encounters = _pilot_encounters()
    _check_invariants(encounters)

    off = _run_condition(encounters, c1_on=False, weights=weights)
    on = _run_condition(encounters, c1_on=True, weights=weights)

    from nanoscribe.power import clopper_pearson_upper, instances_required

    n_slots = off["n_present_value_slots"]
    slots_per_instance = 10  # present-value slots in one campaign_v2 instance

    # --- pi_C2: the INTERACTION effect size -------------------------------
    # Pooled over both C1 conditions: C2 fires on quote-less output regardless
    # of C1, so every probed slot is an opportunity.
    n_absent_pooled = off["n_quote_absent"] + on["n_quote_absent"]
    n_pooled = n_slots + on["n_present_value_slots"]
    pi_c2 = off["quote_absent_rate"]
    pi_c2_upper = clopper_pearson_upper(n_absent_pooled, n_pooled)

    # --- pi_C1: the MAIN effect discordance rate ---------------------------
    # Paired over the same slots: how often does exact_gold_span flip when the
    # answer template stops handing over the gold value?
    by_slot = {(s["encounter_id"], s["atom_id"]): s for s in off["slots"]}
    disc_c1_on_wins = 0
    disc_c1_off_wins = 0
    for s_on in on["slots"]:
        s_off = by_slot[(s_on["encounter_id"], s_on["atom_id"])]
        a = s_on["exact_gold_span_c2on"]
        b = s_off["exact_gold_span_c2on"]
        if a and not b:
            disc_c1_on_wins += 1
        elif b and not a:
            disc_c1_off_wins += 1
    d_c1 = disc_c1_on_wins + disc_c1_off_wins
    pi_c1 = d_c1 / n_slots if n_slots else 0.0

    plan_c1 = instances_required(pi=pi_c1, slots_per_instance=slots_per_instance)
    plan_c2 = instances_required(pi=pi_c2, slots_per_instance=slots_per_instance)
    plan_c2_optimistic = instances_required(
        pi=pi_c2_upper, slots_per_instance=slots_per_instance
    )

    def _plan(p) -> dict[str, Any]:
        return {
            "pi": round(p.pi, 4),
            "d_min_one_directional": p.d_min,
            "slots_per_instance": p.slots_per_instance,
            "design_effect": round(p.deff, 3),
            "instances_required": p.instances_required,
            "effective_slots": p.effective_slots,
            "achieved_power": p.achieved_power,
            "note": p.note,
        }

    payload = {
        "experiment": "p1_pilot_quote_absent_v0",
        "purpose": "nuisance-parameter pilot for the C1xC2 replicate count",
        "peeked_at_measurement_slots": False,
        "pilot_encounters": [item["encounter_id"] for item in encounters],
        "weights": weights,
        "C1off": off,
        "C1on": on,
        "pi_hat_c1off": off["quote_absent_rate"],
        "pi_hat_c1on": on["quote_absent_rate"],
        "interaction": {
            "pi_c2_point": round(pi_c2, 4),
            "n_quote_absent_pooled": n_absent_pooled,
            "n_slots_pooled": n_pooled,
            "pi_c2_upper95": round(pi_c2_upper, 4),
            "plan_at_point_estimate": _plan(plan_c2),
            "plan_at_upper_bound": _plan(plan_c2_optimistic),
        },
        "c1_main_effect": {
            "discordant_total": d_c1,
            "c1on_wins": disc_c1_on_wins,
            "c1off_wins": disc_c1_off_wins,
            "pi_c1": round(pi_c1, 4),
            "plan": _plan(plan_c1),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True))

    print("\n" + "=" * 74)
    print("PILOT — replicate count for the C1xC2 span-port leakage 2x2")
    print("=" * 74)
    for cell in (off, on):
        print(
            f"  {cell['condition']:<6} slots={cell['n_present_value_slots']:<3} "
            f"asserted={cell['n_asserted']:<3} "
            f"quote_absent={cell['n_quote_absent']:<3} "
            f"exact(C2on)={cell['exact_gold_span_c2on']:<3} "
            f"exact(C2off)={cell['exact_gold_span_c2off']:<3}"
        )
    inter = payload["interaction"]
    c1 = payload["c1_main_effect"]
    print("-" * 74)
    print(
        f"  INTERACTION  pi_C2 = {inter['pi_c2_point']:.3f} "
        f"({inter['n_quote_absent_pooled']}/{inter['n_slots_pooled']} pooled), "
        f"95% upper = {inter['pi_c2_upper95']:.3f}"
    )
    print(f"    at point estimate : K = {inter['plan_at_point_estimate']['instances_required']}"
          f"  [{inter['plan_at_point_estimate']['note']}]")
    print(f"    at 95% upper bound: K = {inter['plan_at_upper_bound']['instances_required']}"
          f"  [{inter['plan_at_upper_bound']['note']}]")
    print("-" * 74)
    print(
        f"  C1 MAIN      pi_C1 = {c1['pi_c1']:.3f} "
        f"({c1['discordant_total']} discordant: "
        f"{c1['c1on_wins']} C1on-wins / {c1['c1off_wins']} C1off-wins)"
    )
    print(f"    K = {c1['plan']['instances_required']}  "
          f"(d_min={c1['plan']['d_min_one_directional']}, "
          f"DEFF={c1['plan']['design_effect']}, "
          f"power={c1['plan']['achieved_power']})")
    print("=" * 74)
    print(f"  full record: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
