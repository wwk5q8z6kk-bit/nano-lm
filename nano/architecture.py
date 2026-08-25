"""The five Nano specifications, kept structurally separate.

XXIV states the reset: five things must not be conflated.

    1. ONTOLOGY              what exists / what can be represented
    2. COGNITIVE             how Nano transforms state and solves problems
    3. NEURAL                what learned mechanisms implement parts of cognition
    4. LEARNING              how those mechanisms acquire capability
    5. PROGRESSION           which integrated abilities we prove first

The failure this prevents is an implementation detail becoming an
architectural assumption -- "Transformer + SFT" silently promoted from
spec 3+4 into spec 2.

**The capability ladder is not the architecture.** The architecture is the
invariant substrate (spec 2); the ladder (spec 5) describes how we
progressively prove it. A single field carrying both is the same overloading
failure `nano/ontology.py` guards against at the primitive level -- so it is
guarded here at the specification level.
"""

from __future__ import annotations

from enum import Enum


class Spec(str, Enum):
    ONTOLOGY = "ontology"
    COGNITIVE = "cognitive"
    NEURAL = "neural"
    LEARNING = "learning"
    PROGRESSION = "progression"


class Layer(str, Enum):
    """Spec 2 -- the invariant cognitive substrate (XXXIX). Position, not order."""
    ONTOLOGY = "I_ontology"
    EPISTEMIC = "II_epistemic_contract"
    IDENTITY = "III_identity_authority"
    PERCEPTION = "IV_observation_perception"
    EVIDENCE = "V_evidence_provenance"
    WORLD = "VI_world_representation"
    TEMPORAL = "VII_temporal_causal"
    MEMORY = "VIII_memory_knowledge"
    KERNEL = "IX_cognitive_kernel"
    FABRIC = "X_capability_tool_fabric"
    ARTIFACT = "XI_artifact_compilation"
    VERIFICATION = "XII_verification_epistemic_control"
    DEPENDENCY = "XIII_dependency_invalidation"
    ADAPTATION = "XIV_learning_adaptation"


#: Cross-cutting concerns that belong to no single layer.
CROSS_CUTTING = ("observability", "security", "resource control", "evaluation",
                 "reproducibility", "science/evidence tracking")


class ProvingStage(str, Enum):
    """Spec 5 -- the order in which we PROVE capability. Never a layer."""
    A_EVIDENCE = "A_evidence"
    B_STATE = "B_state"
    C_MEMORY = "C_memory"
    D_RETRIEVAL = "D_retrieval"
    E_REASONING = "E_reasoning"
    F_ARTIFACTS = "F_artifacts"
    G_MULTIMODAL = "G_multimodal"
    H_GENERAL = "H_general_cognition"
    I_CONTINUAL = "I_continual"


class LearningLevel(str, Enum):
    """Spec 4 -- timescale at which a capability is acquired (XVII / L0-L6)."""
    L0_INFERENCE = "L0_inference_time"
    L1_WORKING = "L1_working_state"
    L2_PERSISTENT = "L2_persistent_memory"
    L3_PROCEDURAL = "L3_procedural_skill"
    L4_ADAPTER = "L4_adapter_posttrain"
    L5_BASE = "L5_base_retraining"
    L6_ARCHITECTURE = "L6_architecture_evolution"
    NONE = "none"


#: Spec 3 -- neural mechanisms are HYPOTHESES, never architectural commitments.
#: Listed so an experiment can name one; presence here implies nothing measured.
NEURAL_CANDIDATES = (
    "dense_transformer", "sparse_moe", "ssm_recurrent", "external_memory",
    "retrieval_conditioned", "latent_bottleneck", "cross_attention_memory",
    "modality_expert", "conditional_adapter", "specialised_head",
    "early_exit", "adaptive_depth", "learned_routing",
)


#: The Nano invariant (XXV). Stated once, referenced everywhere.
CANONICAL_CHAIN = (
    "WORLD", "OBSERVATION", "EVIDENCE", "REPRESENTATION",
    "STATE/BELIEF", "REASONING", "ARTIFACT",
)

NANO_INVARIANT = (
    "No generated artifact is the canonical representation of reality. "
    "A note is not the state; a summary is not memory; a timeline is not the "
    "ledger; a chart is not the data; an answer is not the world model; a "
    "latent state is not evidence; a prediction is not a historical fact."
)


# ---------------------------------------------------------------------------
# Extensibility declaration — the guard against today's understanding becoming
# tomorrow's ceiling.
# ---------------------------------------------------------------------------

EXTENSIBILITY_DECLARATION = (
    "All enumerated primitives, reasoning modes, memory classes, modalities, "
    "tools, experts, artifact types, and capability families are illustrative "
    "extensible sets unless explicitly designated as invariant. Their purpose is "
    "to demonstrate interfaces and integration behavior, not to bound the "
    "eventual intelligence architecture."
)

#: Sets that may grow freely. Adding a member is ordinary work.
EXTENSIBLE_SETS = {
    "Layer": "functional responsibilities, not mandatory services or networks",
    "ProvingStage": "current decomposition of how we prove capability",
    "LearningLevel": "current timescale decomposition",
    "NEURAL_CANDIDATES": "hypotheses competing for implementation, never commitments",
    "CROSS_CUTTING": "concerns identified so far",
    "PRIMITIVES": "the ontology is open by construction (nano/ontology.py)",
    "CAPABILITIES": "the capability registry is open (nano/capabilities.py)",
    "Modality": "future modalities plug in without redefining evidence",
    "EpistemicStatus": "more statuses may be needed; none may be collapsed",
}

#: The only things designated INVARIANT. Everything else is negotiable.
#: Kept deliberately tiny — an invariant is a promise, and promises are costly.
INVARIANT_SETS = {
    "CANONICAL_CHAIN": "world -> observation -> evidence -> representation -> "
                       "state -> reasoning -> artifact; the direction never reverses",
    "NANO_INVARIANT": "no generated artifact is the canonical representation of reality",
}


class IntegrationMaturity(str, Enum):
    """Spec 5 -- P1..P9 as INTEGRATION MATURITY, not module boundaries.

    A milestone does not create a capability. Temporal reasoning is already
    exercised at P1; compression reasoning at P2; longitudinal inference at P3.
    P6 is when previously isolated mechanisms become a broad integrated
    reasoning capability. Reading these as modules is the error this docstring
    exists to prevent.
    """
    P1_GROUNDED_ENCOUNTER = "P1_grounded_encounter_reconstruction"
    P2_FAITHFUL_COMPRESSION = "P2_faithful_compression"
    P3_LONGITUDINAL_STATE = "P3_persistent_longitudinal_state"
    P4_SYNTHESIS = "P4_cross_source_multimodal_synthesis"
    P5_EPISTEMIC = "P5_epistemic_intelligence"
    P6_REASONING = "P6_general_reasoning"
    P7_PLANNING = "P7_planning"
    P8_INTERACTION = "P8_tool_environment_interaction"
    P9_ADAPTIVE = "P9_adaptive_intelligence"


class ExperimentalObject(str, Enum):
    """Three distinct kinds of experimental object, each answering a different
    question. Conflating them is how a mechanism result gets misread as a
    capability result.
    """
    MECHANISM_MODEL = "mechanism_model"    # small, cheap, controlled -> causality
    CAPABILITY_MODEL = "capability_model"  # larger pretrained -> achievable ceilings
    NANO_SYSTEM = "nano_system"            # models + memory + tools + verification


#: Why the from-scratch line was not wasted, stated where it cannot be lost.
MECHANISM_MODEL_NOTE = (
    "The 3.15M/10M/160M/30M from-scratch models are MECHANISM_MODELS. They exist "
    "to manipulate data, architecture, objective and scale directly and observe "
    "causality — not to carry the eventual capability stack. The 2026-08 native "
    "wave is the case in point: the 30M model sat below the task capability "
    "floor, and the run still produced durable science on instrumentation, "
    "causal masking, target truncation, objective separation, tokenizer/context "
    "interaction and measurement reliability. Judging a mechanism model by "
    "capability is a category error."
)

#: Efficiency is measured over the SYSTEM, never the checkpoint (XL).
EFFICIENCY_OBJECTIVE = (
    "verified useful capability / (active compute, memory, latency, energy, "
    "money, human review) — a 100M controller with structured memory, retrieval, "
    "tools and deterministic verification may legitimately beat a 30B monolith."
)

#: What Nano is not. Each of these has been proposed as the whole thing.
NANO_IS_NOT = (
    "a scribe", "a chatbot", "a language model", "a knowledge graph",
    "an agent", "a workflow", "a collection of tools",
)


# ---------------------------------------------------------------------------
# The efficiency thesis — stated so it can die.
# ---------------------------------------------------------------------------

#: Verified Task Completion per Unit Compute. The metric that replaces
#: benchmark score as the program's primary target.
VTCPU = (
    "successful VERIFIED work / computation consumed, extended to latency, "
    "energy, memory, financial cost and human intervention. A system that "
    "produces a beautiful answer but does not complete the task scores zero."
)

PARAMETER_EFFICIENCY_HYPOTHESIS = (
    "System-level intelligence can substantially exceed what raw parameter "
    "count predicts, when a compact learned model is combined with structured "
    "state, retrieval, tools, memory, adaptive computation, verification and "
    "specialised computation."
)

#: Stated explicitly because this hypothesis is attractive and therefore
#: dangerous: it is the kind of claim a program adopts as an assumption.
PARAMETER_EFFICIENCY_FALSIFIER = (
    "A compact Nano-System matched against a larger monolithic model on "
    "verified task completion at EQUAL total cost (compute + tools + retrieval "
    "+ human review). If the monolith wins or ties, the hypothesis is dead — "
    "'more components' is not evidence, and the substrate must earn its "
    "complexity. Do not assume it is true; build experiments capable of "
    "falsifying it."
)

#: Three capability readings that must never be reported interchangeably.
CAPABILITY_READINGS = {
    "parametric": "what the learned weights can do alone",
    "system": "what Nano can do with weights + state + memory + retrieval + "
              "algorithms + tools + verification",
    "autonomous": "what Nano can reliably accomplish without rescue",
}

#: Local deployability is a PRIVACY property, not only an engineering one.
LOCAL_DEPLOYMENT_RATIONALE = (
    "Personal and medical data argue for local inference, local memory, "
    "explicit identity boundaries, data minimisation and auditable actions. "
    "The 1-7B local target is therefore strategic rather than a curiosity: a "
    "system that runs privately is a different product from one that ships "
    "every observation to a server."
)
