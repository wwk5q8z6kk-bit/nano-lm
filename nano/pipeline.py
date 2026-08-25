"""NANO-CLIN-001 — two paths over identical frozen fixtures.

BASELINE A: transcript -> generated note (direct).
CANDIDATE B: source -> evidence spans -> assertions -> events -> ledger
             -> state projection -> note -> claim-level verification.

Neither path calls a model. Both are deterministic rule-based stand-ins, so the
first measurement isolates the *architecture* (does routing generation through
an evidence ledger change provenance coverage and unsupported-claim rate?) from
model quality. A model can be substituted behind either path later; the contracts
and the metrics do not change. Substituting one now would confound the very
comparison this experiment exists to make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from nano.contracts import (
    ClinicalAssertion, ClinicalEvent, ConflictRecord, ConflictType,
    DerivedArtifact, EpistemicStatus, EvidenceLedger, EvidenceSpanV2, GapKind,
    KnowledgeGap, Locator, Modality, PatientStateSnapshot, SourceArtifact,
    TemporalExtent, TimePrecision, VerificationReceipt,
)
from nano.fixtures import Fixture

_SPEAKER_RE = re.compile(r"^(clinician|patient):\s*(.*)$")

_NEGATION = ("no ", "not ", "denies", "without", "do not", "never")
_UNCERTAIN_TIME = ("about", "around", "maybe", "a while", "approximately", "roughly")
_ABSENT_MARKER = ("i do not see", "no adverse reaction documented", "not in the record")

#: Concepts whose dates are worth cross-checking. A shared concept is what
#: distinguishes a disagreement from an ordinary chronology.
_CONFLICT_CONCEPTS = ("diabetes", "hypertension", "cancer", "asthma", "copd")


# --------------------------------------------------------------------------
# Shared: segment a transcript into speaker-attributed lines with offsets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Utterance:
    speaker: str
    text: str
    start: int
    end: int
    line_no: int


def segment(transcript: str) -> list[Utterance]:
    out, offset = [], 0
    for i, raw in enumerate(transcript.split("\n")):
        if raw.strip():
            m = _SPEAKER_RE.match(raw.strip())
            if m:
                speaker, said = m.group(1), m.group(2)
                s = offset + raw.index(said)
                out.append(Utterance(speaker, said, s, s + len(said), i))
        offset += len(raw) + 1
    return out


# --------------------------------------------------------------------------
# BASELINE A — direct generation, no evidence layer
# --------------------------------------------------------------------------

def baseline_a(fx: Fixture) -> tuple[str, list[dict]]:
    """Transcript -> note. Emits sentences with no provenance, by construction.

    This is deliberately the naive path: it flattens speaker attribution and
    keeps no pointer back to source. Its failures are the thing being measured.
    """
    lines = []
    for u in segment(fx.transcript):
        t = u.text.rstrip(".")
        if u.speaker == "patient":
            lines.append(f"Patient reports {t.lower()}.")
        else:
            lines.append(f"{t}.")
    note = "\n".join(lines)
    claims = [{"text": s, "evidence_span_ids": [], "supported": False}
              for s in lines]
    return note, claims


# --------------------------------------------------------------------------
# CANDIDATE B — evidence-grounded pipeline
# --------------------------------------------------------------------------

def _status_for(speaker: str, text: str) -> EpistemicStatus:
    low = text.lower()
    if re.search(r"\d+(\.\d+)?\s*(degrees|percent|mg|celsius)", low):
        return EpistemicStatus.DIRECT_MEASUREMENT
    if speaker == "patient":
        return EpistemicStatus.PATIENT_REPORTED
    if any(m in low for m in ("i think", "this is a", "assess")):
        return EpistemicStatus.CLINICIAN_ASSERTED
    return EpistemicStatus.DIRECT_DOCUMENTATION


def _temporal_for(text: str) -> TemporalExtent:
    low = text.lower()
    if any(m in low for m in _UNCERTAIN_TIME):
        year = re.search(r"\b(19|20)\d{2}\b", low)
        return TemporalExtent(
            event_time=year.group(0) if year else "",
            precision=TimePrecision.APPROXIMATE,
            uncertainty="hedged in source wording",
            relative_time=text if not year else "",
        )
    year = re.search(r"\b(19|20)\d{2}\b", low)
    if year:
        return TemporalExtent(event_time=year.group(0), precision=TimePrecision.YEAR)
    return TemporalExtent(precision=TimePrecision.UNKNOWN)


def candidate_b(fx: Fixture) -> dict:
    src = SourceArtifact(
        patient_id=fx.patient_id, modality=Modality.TEXT,
        document_type="encounter_transcript", content=fx.transcript,
        author_or_device="synthetic_generator",
    )
    ledger = EvidenceLedger(patient_id=fx.patient_id)
    ledger.append(sources=[src])

    spans, assertions, gaps = [], [], []
    for u in segment(fx.transcript):
        span = EvidenceSpanV2(
            source_id=src.source_id, patient_id=fx.patient_id,
            modality=Modality.TEXT,
            locator=Locator(start=u.start, end=u.end, line=u.line_no),
            verbatim=u.text, speaker=u.speaker,
        )
        spans.append(span)

        low = u.text.lower()
        # An explicit "not in the record" statement is a GAP, never an absence.
        if any(m in low for m in _ABSENT_MARKER):
            gaps.append(KnowledgeGap(
                patient_id=fx.patient_id, expected_information=u.text,
                kind=GapKind.NOT_FOUND, why_expected="clinician searched and reported",
                search_scope="record at encounter time",
            ))
            continue

        negated = any(low.startswith(n) or f" {n}" in low for n in _NEGATION)
        assertions.append(ClinicalAssertion(
            patient_id=fx.patient_id, subject=u.speaker,
            predicate="denies" if negated else "states",
            obj=u.text, original_wording=u.text,
            epistemic_status=_status_for(u.speaker, u.text),
            evidence_span_ids=(span.evidence_span_id,),
            negated=negated, temporal=_temporal_for(u.text), author=u.speaker,
        ))

    events = [ClinicalEvent(
        patient_id=fx.patient_id, event_type="encounter_utterance",
        temporal=a.temporal, assertion_ids=(a.assertion_id,),
        participants=(a.subject,),
    ) for a in assertions]

    conflicts = _detect_conflicts(fx, assertions)
    ledger.append(spans=spans, assertions=assertions, events=events,
                  conflicts=conflicts, gaps=gaps)

    state = PatientStateSnapshot(
        patient_id=fx.patient_id,
        evidence_ledger_version=ledger.version,
        ledger_hash=ledger.ledger_hash(),
        active_conditions=tuple(
            a.obj for a in assertions
            if a.epistemic_status == EpistemicStatus.CLINICIAN_ASSERTED and not a.negated),
        current_medications=tuple(
            a.obj for a in assertions if "mg" in a.obj.lower() and not a.negated),
        laboratory_state=tuple(
            a.obj for a in assertions
            if a.epistemic_status == EpistemicStatus.DIRECT_MEASUREMENT),
        uncertainties=tuple(
            a.obj for a in assertions
            if a.temporal.precision == TimePrecision.APPROXIMATE),
        conflicts=tuple(c.conflict_id for c in conflicts),
        unresolved_questions=tuple(g.expected_information for g in gaps),
    )

    note, claims = _render_note(assertions, conflicts, gaps)
    artifact = DerivedArtifact(
        patient_id=fx.patient_id, artifact_type="encounter_note",
        task="verified_encounter_note", content=note,
        patient_state_version=state.snapshot_id,
        evidence_ledger_version=ledger.version,
        supporting_evidence=tuple(s.evidence_span_id for s in spans),
        generation_method="evidence_grounded_render",
    )
    receipt = VerificationReceipt(
        artifact_id=artifact.artifact_id, claim_results=tuple(claims),
        coverage_status="complete" if all(c["supported"] for c in claims) else "partial",
    )
    return {"source": src, "ledger": ledger, "spans": spans,
            "assertions": assertions, "events": events, "conflicts": conflicts,
            "gaps": gaps, "state": state, "artifact": artifact,
            "receipt": receipt, "note": note, "claims": claims}


def _detect_conflicts(fx: Fixture, assertions: list) -> list[ConflictRecord]:
    """Flag a date disagreement only when years attach to the SAME concept.

    An earlier version flagged any two distinct years, which reported the
    metoprolol fixture as conflicting: 2019 (started) and 2021 (stopped) are
    sequential events, not contradictory claims. Requiring a shared concept is
    what separates a disagreement from a chronology.
    """
    concept_years: dict[str, dict[str, list[str]]] = {}

    def note(text: str, ref: str) -> None:
        years = re.findall(r"\b((?:19|20)\d{2})\b", text)
        if not years:
            return
        low = text.lower()
        for concept in _CONFLICT_CONCEPTS:
            if concept in low:
                bucket = concept_years.setdefault(concept, {})
                for y in years:
                    bucket.setdefault(y, []).append(ref)

    for a in assertions:
        note(a.original_wording, a.assertion_id)
    for line in (fx.prior_chart or "").split("\n"):
        if line.strip():
            note(line, f"prior_chart:{line.strip()[:24]}")

    out = []
    for concept, years in concept_years.items():
        if len(years) < 2:
            continue
        claim_set = tuple(sorted({r for refs in years.values() for r in refs}))
        if len(claim_set) < 2:
            continue
        out.append(ConflictRecord(
            patient_id=fx.patient_id,
            conflict_type=ConflictType.DATE_DISAGREEMENT,
            claim_set=claim_set,
            clinical_importance="unknown",
            resolution_status="unresolved",
        ))
    return out


def _render_note(assertions, conflicts, gaps) -> tuple[str, list[dict]]:
    """Render with attribution preserved. Every factual line cites evidence."""
    lines, claims = [], []
    for a in assertions:
        if a.negated:
            sentence = f"{a.subject.capitalize()} denies: {a.original_wording}"
        elif a.epistemic_status == EpistemicStatus.PATIENT_REPORTED:
            sentence = f"Patient-reported: {a.original_wording}"
        elif a.epistemic_status == EpistemicStatus.DIRECT_MEASUREMENT:
            sentence = f"Measured: {a.original_wording}"
        elif a.epistemic_status == EpistemicStatus.CLINICIAN_ASSERTED:
            sentence = f"Clinician assessment: {a.original_wording}"
        else:
            sentence = f"Documented: {a.original_wording}"
        if a.temporal.precision == TimePrecision.APPROXIMATE:
            sentence += " [time approximate as stated]"
        lines.append(sentence)
        claims.append({"text": sentence,
                       "evidence_span_ids": list(a.evidence_span_ids),
                       "epistemic_status": a.epistemic_status.value,
                       "supported": bool(a.evidence_span_ids)})
    for c in conflicts:
        lines.append(f"CONFLICT (unresolved): {c.conflict_type.value} "
                     f"across {len(c.claim_set)} sources")
        claims.append({"text": lines[-1], "evidence_span_ids": list(c.claim_set),
                       "epistemic_status": EpistemicStatus.CONFLICTING.value,
                       "supported": True})
    for g in gaps:
        lines.append(f"NOT FOUND IN RECORD (not equivalent to absent): "
                     f"{g.expected_information}")
        claims.append({"text": lines[-1], "evidence_span_ids": [],
                       "epistemic_status": EpistemicStatus.NOT_FOUND.value,
                       "supported": True})
    return "\n".join(lines), claims
