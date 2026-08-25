"""Nano Capability & Architecture Specification — typed registry, not prose.

Each capability carries the eleven fields required by the specification request:

    capability -> internal representation -> module -> inputs -> outputs
    -> memory requirements -> tools -> training objective
    -> evaluation benchmark -> failure modes -> implementation stage

plus an honest `status` against the repository and, where status is not
PROPOSED, an `evidence` pointer to the code or ledger claim that supports it.

Why a registry and not a document: "is a capability missing?" and "is a status
claim backed by evidence?" become mechanical checks (`nano/test_capabilities.py`)
rather than a careful read. Prose cannot be regression-tested.

Status vocabulary is deliberately the same one the directive requires for the
public README, so the two cannot drift:

    IMPLEMENTED   exercised by code in this repository, with a pointer
    PARTIAL       some mechanism exists; the capability is not met
    PROPOSED      designed, not built
    ABSENT        not designed
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class Status(str, Enum):
    IMPLEMENTED = "IMPLEMENTED"
    PARTIAL = "PARTIAL"
    PROPOSED = "PROPOSED"
    ABSENT = "ABSENT"


class Stage(str, Enum):
    """Build order from D-NANO-2026-08-25 §8, plus core work that precedes it."""
    CORE = "CORE"           # Nano Core, domain-independent
    A_ENCOUNTER = "A"       # encounter truth
    B_CHART = "B"           # chart truth
    C_PRESENTATION = "C"    # presentation
    D_LONGITUDINAL = "D"    # longitudinal intelligence
    E_HIGHER_RISK = "E"     # gated, higher risk


@dataclass(frozen=True)
class Capability:
    capability_id: str
    domain: str
    capability: str
    internal_representation: str
    module: str
    inputs: tuple
    outputs: tuple
    memory_requirements: str
    tools: tuple
    training_objective: str
    evaluation_benchmark: str
    failure_modes: tuple
    implementation_stage: Stage
    status: Status
    evidence: str = ""

    def __post_init__(self):
        if not self.failure_modes:
            raise ValueError(
                f"{self.capability_id}: failure_modes required — a capability "
                "with no named failure mode cannot be evaluated")
        if not self.evaluation_benchmark:
            raise ValueError(f"{self.capability_id}: evaluation_benchmark required")
        if self.status in (Status.IMPLEMENTED, Status.PARTIAL) and not self.evidence:
            raise ValueError(
                f"{self.capability_id}: status={self.status.value} requires an "
                "evidence pointer (no unbacked status claims)")
        if self.status == Status.PROPOSED and self.evidence:
            raise ValueError(
                f"{self.capability_id}: PROPOSED must not cite evidence — "
                "citing evidence for unbuilt work is how a plan becomes a claim")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["implementation_stage"] = self.implementation_stage.value
        d["status"] = self.status.value
        return d


def _c(**kw) -> Capability:
    return Capability(**kw)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

CAPABILITIES: tuple[Capability, ...] = (

    # ---------------- Perception ----------------
    _c(capability_id="PER-TEXT", domain="Perception",
       capability="Ingest text sources (notes, letters, summaries)",
       internal_representation="SourceArtifact + EvidenceSpanV2(text locator)",
       module="nano.contracts / nano.pipeline.segment",
       inputs=("raw text", "document type"), outputs=("SourceArtifact", "EvidenceSpanV2"),
       memory_requirements="immutable source retention; content hash",
       tools=(), training_objective="none (deterministic segmentation)",
       evaluation_benchmark="LCRB-1 span precision/recall",
       failure_modes=("offset drift after normalisation", "speaker misattribution",
                      "section boundaries lost"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/pipeline.py::segment; nano/test_nano_clin_001.py"),

    _c(capability_id="PER-AUDIO", domain="Perception",
       capability="Ingest encounter audio with speaker diarisation",
       internal_representation="EvidenceSpanV2(interval locator, speaker)",
       module="not built", inputs=("waveform",),
       outputs=("timed spans", "speaker labels"),
       memory_requirements="audio retained or referenced; never discarded after ASR",
       tools=("ASR", "diarisation"),
       training_objective="none in Core; specialist model",
       evaluation_benchmark="LCRB-1 with audio fixtures; WER + diarisation error",
       failure_modes=("ASR error propagates as fact", "speaker swap inverts attribution",
                      "transcript treated as source, losing the audio"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PROPOSED),

    _c(capability_id="PER-TABLE", domain="Perception",
       capability="Ingest tabular data (labs, medication lists)",
       internal_representation="EvidenceSpanV2(table locator)",
       module="not built", inputs=("CSV/FHIR/EHR export",),
       outputs=("cell-located spans",),
       memory_requirements="original units retained alongside normalised",
       tools=("table parser",), training_objective="none",
       evaluation_benchmark="LCRB-2 lab trajectory completeness",
       failure_modes=("unit loss", "column misalignment", "header row as data"),
       implementation_stage=Stage.B_CHART, status=Status.PROPOSED),

    _c(capability_id="PER-IMAGE", domain="Perception",
       capability="Ingest imaging and its report as linked evidence",
       internal_representation="EvidenceSpanV2(image locator + bbox)",
       module="not built", inputs=("image", "radiology report"),
       outputs=("region-located spans",),
       memory_requirements="image referenced, not inlined into the graph",
       tools=("vision specialist",), training_objective="none in Core",
       evaluation_benchmark="LCRB-5 imaging evolution",
       failure_modes=("model asserts findings the report does not support",
                      "report and image diverge silently"),
       implementation_stage=Stage.E_HIGHER_RISK, status=Status.PROPOSED),

    _c(capability_id="PER-SIGNAL", domain="Perception",
       capability="Ingest waveforms and continuous device data",
       internal_representation="EvidenceSpanV2(interval locator, channel)",
       module="not built", inputs=("ECG/CGM/monitor stream",),
       outputs=("interval spans", "derived measurements"),
       memory_requirements="time-series store; downsampling must be recorded",
       tools=("signal analysis",), training_objective="none in Core",
       evaluation_benchmark="LCRB-5",
       failure_modes=("downsampling hides events", "artefact read as physiology"),
       implementation_stage=Stage.E_HIGHER_RISK, status=Status.PROPOSED),

    # ---------------- Source intelligence ----------------
    _c(capability_id="SRC-PROV", domain="Source intelligence",
       capability="Record authorship, source type, and bitemporal times",
       internal_representation="SourceArtifact fields + EvidenceSpanV2 times",
       module="nano.contracts.SourceArtifact",
       inputs=("source metadata",), outputs=("provenance-bearing artifact",),
       memory_requirements="immutable; content-addressed",
       tools=(), training_objective="none",
       evaluation_benchmark="provenance coverage on LCRB-1",
       failure_modes=("documentation time used as event time",
                      "copied text credited to the copier"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/contracts.py::SourceArtifact, EvidenceSpanV2"),

    _c(capability_id="SRC-EPIST", domain="Source intelligence",
       capability="Distinguish measured / observed / reported / inferred",
       internal_representation="EpistemicStatus (14 values, never a scalar)",
       module="nano.contracts.EpistemicStatus",
       inputs=("assertion", "speaker", "source type"),
       outputs=("epistemic status",),
       memory_requirements="status travels with the assertion permanently",
       tools=(), training_objective="classification of assertion provenance",
       evaluation_benchmark="LCRB-1 attribution accuracy",
       failure_modes=("patient report promoted to clinician confirmation",
                      "inference rendered as documented",
                      "collapsing the 14 statuses to one confidence number"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/contracts.py::EpistemicStatus; test_patient_report_is_not_promoted"),

    # ---------------- Extraction ----------------
    _c(capability_id="EXT-ASSERT", domain="Extraction",
       capability="Extract clinical assertions preserving original wording",
       internal_representation="ClinicalAssertion(original_wording + normalized)",
       module="nano.pipeline.candidate_b",
       inputs=("evidence spans",), outputs=("assertions",),
       memory_requirements="original wording never discarded",
       tools=(), training_objective="span-to-assertion extraction",
       evaluation_benchmark="LCRB-1 assertion precision/recall",
       failure_modes=("interrogatives extracted as assertions (OPEN defect)",
                      "epistemic non-recall classified as negation (OPEN defect)",
                      "normalisation overwrites source wording"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/pipeline.py; two defects recorded in artifacts/nano_clin_001/decision.md"),

    _c(capability_id="EXT-EVENT", domain="Extraction",
       capability="Derive clinical events from assertions",
       internal_representation="ClinicalEvent(assertion_ids, temporal)",
       module="nano.contracts.ClinicalEvent",
       inputs=("assertions",), outputs=("events",),
       memory_requirements="event must retain its assertion ancestry",
       tools=(), training_objective="event typing and participant binding",
       evaluation_benchmark="LCRB-1 event extraction accuracy",
       failure_modes=("event invented without an assertion",
                      "one utterance split into duplicate events"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/contracts.py::ClinicalEvent (1:1 with assertions only)"),

    _c(capability_id="EXT-NEG", domain="Extraction",
       capability="Preserve negation as first-class, not absence",
       internal_representation="ClinicalAssertion.negated",
       module="nano.pipeline", inputs=("assertion text",), outputs=("negation flag",),
       memory_requirements="none beyond the assertion",
       tools=(), training_objective="negation detection",
       evaluation_benchmark="LCRB-1 incorrect-negation rate",
       failure_modes=("denied finding rendered as present (safety)",
                      "'do not remember' treated as clinical denial (OPEN defect)"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/test_nano_clin_001.py::test_negation_is_preserved"),

    # ---------------- Normalization ----------------
    _c(capability_id="NRM-CONCEPT", domain="Normalization",
       capability="Map surface terms to concepts without losing the original",
       internal_representation="normalized_concept beside original_wording",
       module="field exists; no ontology bound",
       inputs=("surface term",), outputs=("concept id", "mapping confidence"),
       memory_requirements="ontology store, versioned",
       tools=("terminology service (SNOMED/LOINC/RxNorm)",),
       training_objective="concept linking",
       evaluation_benchmark="LCRB-2 entity-resolution accuracy",
       failure_modes=("metoprolol tartrate conflated with succinate",
                      "normalisation overwrites source", "ontology version drift"),
       implementation_stage=Stage.B_CHART, status=Status.PARTIAL,
       evidence="nano/contracts.py::ClinicalAssertion.normalized_concept (field only)"),

    _c(capability_id="NRM-UNITS", domain="Normalization",
       capability="Retain original and normalised units together",
       internal_representation="original_units + normalized_units",
       module="fields exist; no converter",
       inputs=("measurement",), outputs=("both unit forms",),
       memory_requirements="none", tools=("UCUM converter",),
       training_objective="none (deterministic)",
       evaluation_benchmark="LCRB-2 numerical verification",
       failure_modes=("silent unit conversion changes a dose",
                      "mg/mL confused with mg"),
       implementation_stage=Stage.B_CHART, status=Status.PARTIAL,
       evidence="nano/contracts.py::ClinicalAssertion original_units/normalized_units"),

    # ---------------- Temporal ----------------
    _c(capability_id="TMP-BITEMP", domain="Temporal",
       capability="Separate event / documentation / discovery / system time",
       internal_representation="TemporalExtent (11 fields)",
       module="nano.contracts.TemporalExtent",
       inputs=("assertion text", "source times"), outputs=("temporal extent",),
       memory_requirements="both time axes retained for replay",
       tools=("date parser",), training_objective="temporal expression extraction",
       evaluation_benchmark="LCRB-1/2 event-time accuracy, doc-vs-event separation",
       failure_modes=("note date becomes symptom onset",
                      "exact date manufactured from hedged language"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/contracts.py::TemporalExtent; test_approximate_time_cannot_carry_an_exact_date"),

    _c(capability_id="TMP-PRECISION", domain="Temporal",
       capability="Preserve time precision and refuse to invent it",
       internal_representation="TimePrecision enum, enforced in __post_init__",
       module="nano.contracts", inputs=("temporal expression",),
       outputs=("precision label",), memory_requirements="none",
       tools=(), training_objective="precision classification",
       evaluation_benchmark="uncertain-time calibration on LCRB-2",
       failure_modes=("'around 2021' rendered as 2021-01-01",
                      "approximate interval collapsed to a point"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/contracts.py::TemporalExtent.__post_init__ raises"),

    # ---------------- Longitudinal ----------------
    _c(capability_id="LNG-EPISODE", domain="Longitudinal",
       capability="Group related events into episodes",
       internal_representation="Episode node over events (not built)",
       module="not built", inputs=("events",), outputs=("episodes",),
       memory_requirements="episodic memory; addressable without rereading source",
       tools=("graph traversal",), training_objective="episode segmentation",
       evaluation_benchmark="LCRB-3 episode-linking accuracy",
       failure_modes=("two admissions merged", "one admission split",
                      "episode boundary set by document boundary"),
       implementation_stage=Stage.B_CHART, status=Status.ABSENT),

    _c(capability_id="LNG-TRAJ", domain="Longitudinal",
       capability="Build per-problem trajectories across years",
       internal_representation="trajectory over episodes (not built)",
       module="not built", inputs=("episodes", "measurements"),
       outputs=("trajectory series",),
       memory_requirements="longitudinal memory with descent to source",
       tools=("time-series analysis",), training_objective="trajectory construction",
       evaluation_benchmark="LCRB-4 parallel disease trajectories",
       failure_modes=("unrelated problems merged into one trajectory",
                      "trend asserted from two points"),
       implementation_stage=Stage.B_CHART, status=Status.ABSENT),

    _c(capability_id="LNG-ZOOM", domain="Longitudinal",
       capability="Descend from trajectory to exact source passage",
       internal_representation="L0..L6 zoom chain via ids",
       module="partially: ids chain span->assertion->event->state",
       inputs=("any level",), outputs=("next level down",),
       memory_requirements="every level must retain child ids",
       tools=(), training_objective="none (structural)",
       evaluation_benchmark="provenance coverage; source-reversible derivation",
       failure_modes=("compression without a descent path",
                      "summary that cannot be traced"),
       implementation_stage=Stage.B_CHART, status=Status.PARTIAL,
       evidence="nano/contracts.py id chain; artifacts/nano_clin_001/*.jsonl"),

    # ---------------- State ----------------
    _c(capability_id="STA-PROJ", domain="State",
       capability="Project patient state from the evidence ledger",
       internal_representation="PatientStateSnapshot (rebuildable projection)",
       module="nano.pipeline.candidate_b",
       inputs=("ledger",), outputs=("state snapshot",),
       memory_requirements="ledger authoritative; state derivable, never primary",
       tools=(), training_objective="none (deterministic projection)",
       evaluation_benchmark="current-state accuracy; rebuild determinism",
       failure_modes=("state edited directly, diverging from ledger",
                      "projection not reproducible"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/test_nano_clin_001.py::test_state_is_a_rebuildable_projection"),

    _c(capability_id="STA-VERSION", domain="State",
       capability="Append new evidence without mutating history",
       internal_representation="EvidenceLedger.version + ledger_hash",
       module="nano.contracts.EvidenceLedger",
       inputs=("new evidence",), outputs=("new version",),
       memory_requirements="append-only; prior versions recoverable",
       tools=(), training_objective="none",
       evaluation_benchmark="LCRB-7 incremental update and invalidation",
       failure_modes=("correction erases the earlier claim",
                      "dependent artifacts not marked stale"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="test_new_evidence_appends_and_does_not_overwrite (append yes; "
                "downstream invalidation not implemented)"),

    _c(capability_id="STA-DIFF", domain="State",
       capability="Diff two state versions into a change report",
       internal_representation="StateDelta (not built)",
       module="change_summary.md is a stub, not a diff",
       inputs=("state v_n", "state v_n+1"), outputs=("changed / new / resolved"),
       memory_requirements="both versions retained",
       tools=(), training_objective="none (structural diff)",
       evaluation_benchmark="LCRB-2 change detection",
       failure_modes=("change reported without cause",
                      "unchanged item reported as changed"),
       implementation_stage=Stage.B_CHART, status=Status.ABSENT),

    # ---------------- Conflict and gaps ----------------
    _c(capability_id="CFL-DETECT", domain="Conflict",
       capability="Detect contradictions and never resolve them silently",
       internal_representation="ConflictRecord(unresolved unless human disposition)",
       module="nano.pipeline._detect_conflicts",
       inputs=("assertions", "prior chart"), outputs=("conflict records",),
       memory_requirements="all sides of the conflict retained",
       tools=(), training_objective="contradiction detection",
       evaluation_benchmark="LCRB-2 contradiction recall; silent-conflict rate",
       failure_modes=("sequential events flagged as conflict (FIXED, pinned)",
                      "conflict resolved by picking the most recent source",
                      "conflict detected but not surfaced in the artifact"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/pipeline.py::_detect_conflicts; concept-scoped, 5 concepts only"),

    _c(capability_id="GAP-DETECT", domain="Conflict",
       capability="Distinguish not-found / unavailable / never-performed",
       internal_representation="KnowledgeGap(GapKind)",
       module="nano.pipeline", inputs=("assertions",), outputs=("gaps",),
       memory_requirements="search scope recorded with the gap",
       tools=(), training_objective="expectation modelling (what should be here?)",
       evaluation_benchmark="LCRB-2 missing-information detection",
       failure_modes=("'not mentioned' rendered as 'absent' (safety)",
                      "gap not raised because nothing prompted the search"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/test_nano_clin_001.py::test_not_found_is_distinguished_from_absent"),

    # ---------------- Memory ----------------
    _c(capability_id="MEM-HIER", domain="Memory",
       capability="Working / encounter / episodic / longitudinal / semantic memory",
       internal_representation="separate stores, not one vector index",
       module="not built", inputs=("state", "evidence"), outputs=("retrievable memory",),
       memory_requirements="the capability itself",
       tools=(), training_objective="learned write/retain/decay policy",
       evaluation_benchmark="B0 next-state prediction; recall at fixed budget",
       failure_modes=("one vector DB standing in for memory",
                      "write-everything, so retrieval degrades",
                      "patient memory merged with medical knowledge"),
       implementation_stage=Stage.CORE, status=Status.ABSENT),

    _c(capability_id="MEM-SELECT", domain="Memory",
       capability="Decide what is worth remembering",
       internal_representation="learned gate over observations",
       module="not built", inputs=("observation", "current state"),
       outputs=("write / discard",),
       memory_requirements="bounded store", tools=(),
       training_objective="importance x novelty x future usefulness",
       evaluation_benchmark="B0 recall at fixed memory budget",
       failure_modes=("discards the fact needed later",
                      "keeps everything, learning nothing"),
       implementation_stage=Stage.CORE, status=Status.ABSENT),

    # ---------------- Knowledge ----------------
    _c(capability_id="KNW-SEP", domain="Knowledge",
       capability="Keep patient / medical / institutional / model knowledge apart",
       internal_representation="four memories, never merged",
       module="documented invariant; not enforced in code",
       inputs=("claim",), outputs=("world-of-origin label",),
       memory_requirements="separate stores",
       tools=("literature retrieval", "guideline retrieval"),
       training_objective="none (architectural)",
       evaluation_benchmark="unsupported-claim rate by origin",
       failure_modes=("'the literature says X' becomes 'the patient has X'",
                      "institutional protocol asserted as medical fact"),
       implementation_stage=Stage.E_HIGHER_RISK, status=Status.PROPOSED),

    # ---------------- Retrieval ----------------
    _c(capability_id="RET-ROUTER", domain="Retrieval",
       capability="Route between semantic / exact / temporal / graph / episode retrieval",
       internal_representation="retrieval plan",
       module="not built", inputs=("question", "state"),
       outputs=("retrieval plan", "results"),
       memory_requirements="indexes per strategy",
       tools=("BM25", "embeddings", "graph traversal"),
       training_objective="routing accuracy vs oracle",
       evaluation_benchmark="LCRB-6 hidden questions; routing accuracy",
       failure_modes=("top-k vector search used for an exact-value question",
                      "retrieval succeeds, reasoning still ungrounded"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.ABSENT),

    # ---------------- Reasoning ----------------
    _c(capability_id="RSN-TEMPORAL", domain="Reasoning",
       capability="Reason over before/after/during/overlap/recurrence",
       internal_representation="temporal relations over events",
       module="not built", inputs=("events",), outputs=("ordering", "intervals"),
       memory_requirements="temporal index",
       tools=("date arithmetic",), training_objective="temporal relation prediction",
       evaluation_benchmark="B0/B1 ordering accuracy",
       failure_modes=("ordering by document order rather than event time",
                      "unknown time treated as latest"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.ABSENT),

    _c(capability_id="RSN-CAUSAL", domain="Reasoning",
       capability="Separate documented rationale / temporal association / inferred cause",
       internal_representation="typed causal edge with epistemic status",
       module="not built", inputs=("events", "trajectories"),
       outputs=("causal hypotheses, typed"),
       memory_requirements="hypotheses stored apart from observations",
       tools=("statistics",), training_objective="causal hypothesis generation",
       evaluation_benchmark="B4 correlation-vs-causation; LCRB-6",
       failure_modes=("temporal association reported as cause (safety)",
                      "clinician's stated reason treated as ground truth"),
       implementation_stage=Stage.E_HIGHER_RISK, status=Status.ABSENT),

    _c(capability_id="RSN-METACOG", domain="Reasoning",
       capability="Represent what is known / unknown / conflicting / needed",
       internal_representation="uncertainty dimensions + open questions in state",
       module="partially: uncertainties and unresolved_questions in state",
       inputs=("state",), outputs=("knowledge gaps", "next information need"),
       memory_requirements="none beyond state",
       tools=(), training_objective="calibration",
       evaluation_benchmark="abstention correctness; uncertain-time calibration",
       failure_modes=("confident answer over a gap",
                      "uncertainty reported as a single number"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.PARTIAL,
       evidence="nano/contracts.py::PatientStateSnapshot.uncertainties, unresolved_questions"),

    # ---------------- Planning and compute ----------------
    _c(capability_id="PLN-DECOMP", domain="Planning",
       capability="Decompose a question into an evidence and tool plan",
       internal_representation="task plan",
       module="not built", inputs=("question",), outputs=("plan",),
       memory_requirements="working memory",
       tools=("tool runtime",), training_objective="plan quality vs oracle",
       evaluation_benchmark="LCRB-6",
       failure_modes=("single generation call instead of a plan",
                      "plan not followed"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.ABSENT),

    _c(capability_id="PLN-ADAPTIVE", domain="Planning",
       capability="Spend computation in proportion to difficulty",
       internal_representation="halting signal over reasoning iterations",
       module="not built", inputs=("question", "uncertainty"),
       outputs=("continue / halt",), memory_requirements="none",
       tools=(), training_objective="halting policy",
       evaluation_benchmark="compute vs difficulty correlation (Program K1)",
       failure_modes=("uniform compute regardless of difficulty",
                      "halts while uncertainty is still high"),
       implementation_stage=Stage.CORE, status=Status.ABSENT),

    # ---------------- Tools ----------------
    _c(capability_id="TOL-DELEGATE", domain="Tools",
       capability="Delegate arithmetic and lookup rather than estimating",
       internal_representation="tool call with typed result",
       module="not built", inputs=("subtask",), outputs=("tool result",),
       memory_requirements="procedural memory of tool behaviour",
       tools=("calculator", "code", "terminology", "statistics"),
       training_objective="tool selection",
       evaluation_benchmark="numerical verification error rate",
       failure_modes=("model does arithmetic in-context and is wrong",
                      "tool result not verified before use"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.ABSENT),

    # ---------------- Visualization ----------------
    _c(capability_id="VIS-FROMSTATE", domain="Visualization",
       capability="Compile charts and timelines from state, not from prose",
       internal_representation="chart spec derived from projection",
       module="timeline.json emitted; no renderer",
       inputs=("state", "trajectories"), outputs=("timeline", "trend chart"),
       memory_requirements="none",
       tools=("plotting",), training_objective="none (deterministic compile)",
       evaluation_benchmark="does the chart match the underlying series?",
       failure_modes=("hallucinated chart generated from text",
                      "chart and note disagree"),
       implementation_stage=Stage.C_PRESENTATION, status=Status.PARTIAL,
       evidence="scripts/run_nano_clin_001.py emits timeline.json from events"),

    # ---------------- Generation ----------------
    _c(capability_id="GEN-COMPILE", domain="Generation",
       capability="Compile many artifact types from one patient state",
       internal_representation="DerivedArtifact with state + ledger version",
       module="nano.pipeline._render_note (one artifact type only)",
       inputs=("state", "task"), outputs=("note / summary / handoff / presentation"),
       memory_requirements="artifact records its input versions",
       tools=(), training_objective="artifact generation conditioned on state",
       evaluation_benchmark="LCRB-1..5 factual support, critical omission rate",
       failure_modes=("ten independent summarisers instead of one compiler",
                      "artifact not linked to the state version that produced it"),
       implementation_stage=Stage.C_PRESENTATION, status=Status.PARTIAL,
       evidence="nano/contracts.py::DerivedArtifact; one renderer implemented"),

    _c(capability_id="GEN-AUDIENCE", domain="Generation",
       capability="Same facts, different audience (clinician / patient / specialist)",
       internal_representation="audience parameter over the same state",
       module="not built", inputs=("state", "audience"), outputs=("artifact",),
       memory_requirements="none", tools=(),
       training_objective="audience-conditioned generation",
       evaluation_benchmark="factual support held constant across audiences",
       failure_modes=("simplification drops a critical fact",
                      "facts change with audience"),
       implementation_stage=Stage.C_PRESENTATION, status=Status.ABSENT),

    # ---------------- Verification ----------------
    _c(capability_id="VRF-CLAIM", domain="Verification",
       capability="Verify every factual claim against evidence",
       internal_representation="VerificationReceipt(claim_results)",
       module="nano.contracts.VerificationReceipt",
       inputs=("artifact", "ledger"), outputs=("receipt",),
       memory_requirements="none",
       tools=(), training_objective="none (checkable)",
       evaluation_benchmark="provenance coverage; unsupported-claim rate",
       failure_modes=("artifact-level verification hides a bad sentence",
                      "verification of the generator by the generator"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/contracts.py::VerificationReceipt; artifacts/nano_clin_001/*/verification_receipt.json"),

    _c(capability_id="VRF-ADVERSARIAL", domain="Verification",
       capability="Attack own output before emitting it",
       internal_representation="self-critique pass with revision",
       module="not built", inputs=("draft artifact",),
       outputs=("findings", "revision"),
       memory_requirements="none", tools=(),
       training_objective="critique quality, measured not assumed",
       evaluation_benchmark="does the critic catch injected errors?",
       failure_modes=("critic agrees with the generator",
                      "self-critique measured by self-score"),
       implementation_stage=Stage.C_PRESENTATION, status=Status.ABSENT),

    # ---------------- Uncertainty ----------------
    _c(capability_id="UNC-ABSTAIN", domain="Uncertainty",
       capability="Abstain as a first-class outcome",
       internal_representation="DecisionAction.ABSTAIN / REVIEW",
       module="fabric.schemas.DecisionAction",
       inputs=("claim", "verification"), outputs=("present / qualify / abstain / review"),
       memory_requirements="none", tools=(),
       training_objective="selective prediction",
       evaluation_benchmark="presented-error at coverage; C_FABRIC_SLICE",
       failure_modes=("abstention treated as failure and tuned away",
                      "abstains on everything, achieving trivial safety"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="fabric/schemas.py::DecisionAction; ledger claim C_FABRIC_SLICE"),

    # ---------------- Learning ----------------
    _c(capability_id="LRN-CORRECTION", domain="Learning",
       capability="Absorb a human correction and recompute what depends on it",
       internal_representation="HumanDisposition + dependency invalidation",
       module="not built", inputs=("correction",),
       outputs=("new state version", "stale artifact list"),
       memory_requirements="dependency graph artifact->state->evidence",
       tools=(), training_objective="none (structural)",
       evaluation_benchmark="LCRB-7 invalidation and freshness",
       failure_modes=("correction applied, dependent summary left stale",
                      "correction overwrites the original claim"),
       implementation_stage=Stage.D_LONGITUDINAL, status=Status.ABSENT),

    _c(capability_id="LRN-LOOPS", domain="Learning",
       capability="Fast loop touches state; slow loop touches weights, offline",
       internal_representation="two separated loops",
       module="documented invariant; enforced by absence of any training path",
       inputs=("interaction", "curated corpus"), outputs=("state", "model version"),
       memory_requirements="separate stores per loop",
       tools=(), training_objective="none",
       evaluation_benchmark="no live weight update from unvalidated data",
       failure_modes=("patient data reaches the training corpus",
                      "model version not recorded on the artifact"),
       implementation_stage=Stage.CORE, status=Status.PROPOSED),

    # ---------------- Safety ----------------
    _c(capability_id="SAF-ISOLATION", domain="Safety",
       capability="Patient isolation enforced in the type system",
       internal_representation="patient_id required on every evidence object",
       module="nano.contracts",
       inputs=("any evidence",), outputs=("scoped object",),
       memory_requirements="no cross-patient index",
       tools=(), training_objective="none",
       evaluation_benchmark="cross-patient contamination rate (must be 0)",
       failure_modes=("retrieval crosses patients",
                      "shared cache leaks between patients"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.IMPLEMENTED,
       evidence="nano/test_nano_clin_001.py::test_no_cross_patient_contamination"),

    _c(capability_id="SAF-ACTION", domain="Safety",
       capability="Human review is the action boundary",
       internal_representation="HumanDisposition required to resolve",
       module="nano.contracts.ConflictRecord enforces it for conflicts",
       inputs=("proposed output",), outputs=("accept / edit / reject / defer"),
       memory_requirements="disposition retained",
       tools=(), training_objective="none",
       evaluation_benchmark="no autonomous clinical action",
       failure_modes=("system resolves a conflict on its own",
                      "recommendation emitted without review"),
       implementation_stage=Stage.A_ENCOUNTER, status=Status.PARTIAL,
       evidence="nano/contracts.py::ConflictRecord requires human_disposition"),
)


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------

#: The 24 capability domains this specification is required to cover.
REQUIRED_DOMAINS = (
    "Perception", "Source intelligence", "Extraction", "Normalization",
    "Temporal", "Longitudinal", "State", "Conflict", "Memory", "Knowledge",
    "Retrieval", "Reasoning", "Planning", "Tools", "Visualization",
    "Generation", "Verification", "Uncertainty", "Learning", "Safety",
)


def by_status() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {s.value: [] for s in Status}
    for c in CAPABILITIES:
        out[c.status.value].append(c.capability_id)
    return out


def by_domain() -> dict[str, list[Capability]]:
    out: dict[str, list[Capability]] = {}
    for c in CAPABILITIES:
        out.setdefault(c.domain, []).append(c)
    return out


def coverage() -> dict:
    counts = {s.value: len(v) for s, v in
              zip(Status, [by_status()[s.value] for s in Status])}
    return {
        "total": len(CAPABILITIES),
        "by_status": counts,
        "domains_covered": len(by_domain()),
        "domains_required": len(REQUIRED_DOMAINS),
        "missing_domains": sorted(set(REQUIRED_DOMAINS) - set(by_domain())),
    }
