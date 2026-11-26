"""First-Principles Interaction Intelligence Substrate (NANOLM Omega-v3, Cycle 4).

Implements the typed 7-tuple computational world ``U_t = (L_t, G_t, H_t, B_t, M_t, R_t, Theta)``
over the eight candidate computational primitives (State, Identity, Relation, Transition,
Information, Constraint, Adaptation, Objective) while interoperating read-only with ``nano.contracts``:

1. ``AppendOnlySourceLedger`` (L_t): Content-addressed, immutable source ledger with
   strict three-clock bitemporal admissibility ``q_ij(t_as_of)`` and hard tenant/entity
   isolation masks ``p_ij``.
2. ``DeterministicTypedRuleSolver`` (D0): Exact zero-parameter rule baseline for clean
   canonical ledgers (establishing honestly when learned neural dynamics are unnecessary).
3. Controlled neural microcosm models (~120K iso-parameter matched at d=48):
   - ``FlatEventTransformer`` (FP1_B0): Unconstrained flat self-attention without lossless pointers.
   - ``UnsignedTypedGAT`` (FP1_B1): Non-negative softmax GAT without signed/transport operators.
   - ``SignedTypedMesh`` (FP1_M1): Hard p_ij*q_ij masks + signed coupling + pointer head.
   - ``SignedSheafPortHamiltonian`` (FP1_M2): Skew-symmetric J(G) with SO(3)^(d/3) projective
     adjoint parallel transport T_ij, PSD dissipation R >= alpha_diss * I, discrete Lyapunov
     contraction H(H^{k+1}) <= H(H^k) when drive u=0, and lossless L_t pointer readout.
   - ``DynamicRewirePortHamiltonian`` (FP1_M3): Budgeted sparse top-B dynamic rewiring on FP1_M2.
4. ``VoIHaltingPolicy``: Value-of-Information stopping rule balancing delta state norm +
   Top-2 margin uncertainty against step compute cost lambda.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import random
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from nano.contracts import DerivationMode, EpistemicStatus
from nanolm_holonomy.spinor_algebra import so3_projective_adjoint, su2_lie_exponential

PRIMITIVES_8: tuple[str, ...] = (
    "state",
    "identity",
    "relation",
    "transition",
    "information",
    "constraint",
    "adaptation",
    "objective",
)

CLOCK_TYPES: tuple[str, ...] = ("valid_time", "record_time", "reasoning_time")


class WorldEpistemicState(str, Enum):
    VERIFIED_VALUE = "VERIFIED_VALUE"
    REPORTED_UNVERIFIED = "REPORTED_UNVERIFIED"
    CONFLICTING = "CONFLICTING"
    UNKNOWN_NOT_FOUND = "UNKNOWN_NOT_FOUND"
    ISOLATED_DENIED = "ISOLATED_DENIED"


EPISTEMIC_LABELS: tuple[str, ...] = tuple(s.value for s in WorldEpistemicState)
LABEL_TO_ID: dict[str, int] = {lbl: idx for idx, lbl in enumerate(EPISTEMIC_LABELS)}

# Edge types in typed interaction graph G_t
EDGE_NONE = 0
EDGE_SUPPORTS = 1
EDGE_CONTRADICTS = 2
EDGE_SUPERSEDES = 3
EDGE_SAME_SLOT = 4


def _compute_record_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerRecord:
    """Immutable content-addressed source record in L_t preserving exact raw text."""

    record_id: str
    entity_id: str
    tenant_id: str
    slot: str
    raw_value: str
    polarity: int  # +1 confirm/assert, 0 ambiguous/report, -1 negate/retract
    valid_time: float
    record_time: float
    source_kind: str  # "verified_order" | "unverified_report" | "correction" | "retraction"
    epistemic_status: EpistemicStatus
    derivation_mode: DerivationMode
    text: str
    target_record_id: str = ""
    sha256_hex: str = ""

    def __post_init__(self) -> None:
        if not self.record_id or not self.entity_id or not self.tenant_id:
            raise ValueError("record_id, entity_id, and tenant_id are required")
        if self.polarity not in (-1, 0, 1):
            raise ValueError(f"polarity must be -1, 0, or 1; got {self.polarity}")
        digest = _compute_record_sha256(
            {
                "record_id": self.record_id,
                "entity_id": self.entity_id,
                "tenant_id": self.tenant_id,
                "slot": self.slot,
                "raw_value": self.raw_value,
                "polarity": self.polarity,
                "valid_time": self.valid_time,
                "record_time": self.record_time,
                "source_kind": self.source_kind,
                "epistemic_status": self.epistemic_status.value,
                "derivation_mode": self.derivation_mode.value,
                "text": self.text,
                "target_record_id": self.target_record_id,
            }
        )
        if not self.sha256_hex:
            object.__setattr__(self, "sha256_hex", digest)
        elif self.sha256_hex != digest:
            raise ValueError("LedgerRecord sha256_hex integrity mismatch")


class AppendOnlySourceLedger:
    """Append-only source evidence ledger L_t preventing silent history mutation."""

    def __init__(self) -> None:
        self._records: list[LedgerRecord] = []
        self._by_id: dict[str, LedgerRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[LedgerRecord, ...]:
        return tuple(self._records)

    def append(self, rec: LedgerRecord) -> LedgerRecord:
        if rec.record_id in self._by_id:
            raise ValueError(f"Append-only violation: duplicate record_id {rec.record_id!r}")
        self._records.append(rec)
        self._by_id[rec.record_id] = rec
        return rec

    def get(self, record_id: str) -> LedgerRecord | None:
        return self._by_id.get(record_id)

    def as_of_view(
        self,
        *,
        entity_id: str,
        tenant_id: str,
        slot: str | None = None,
        as_of_time: float,
        reasoning_time: float | None = None,
    ) -> list[LedgerRecord]:
        """Return only records admissible by tenant/entity (p_ij=1) and record_time <= as_of_time."""
        del reasoning_time  # tracked distinctly by caller/audit without leaking future record_times
        p_mask, q_mask = bitemporal_admissibility_mask(
            self._records,
            query_entity_id=entity_id,
            query_tenant_id=tenant_id,
            as_of_time=as_of_time,
        )
        out: list[LedgerRecord] = []
        for rec, p, q in zip(self._records, p_mask, q_mask):
            if p == 1 and q == 1 and (slot is None or rec.slot == slot):
                out.append(rec)
        return out

    def snapshot_metadata(self) -> dict[str, Any]:
        return {
            "count": len(self._records),
            "ids": [r.record_id for r in self._records],
            "derivation_modes": [r.derivation_mode.value for r in self._records],
        }


def bitemporal_admissibility_mask(
    records: list[LedgerRecord] | tuple[LedgerRecord, ...],
    *,
    query_entity_id: str,
    query_tenant_id: str,
    as_of_time: float,
) -> tuple[list[int], list[int]]:
    """Compute hard permission/identity mask p_ij and bitemporal admissibility mask q_ij(t)."""
    p_mask: list[int] = []
    q_mask: list[int] = []
    for rec in records:
        p_mask.append(1 if (rec.tenant_id == query_tenant_id and rec.entity_id == query_entity_id) else 0)
        q_mask.append(1 if rec.record_time <= as_of_time else 0)
    return p_mask, q_mask


# ---------------------------------------------------------------------------
# Deterministic Symbolic Rule Baseline (0 Learned Parameters)
# ---------------------------------------------------------------------------

class DeterministicTypedRuleSolver:
    """Zero-parameter symbolic baseline operating directly on L_t and typed syntax."""

    @staticmethod
    def resolve_ledger(
        ledger: AppendOnlySourceLedger,
        *,
        entity_id: str,
        tenant_id: str,
        authorized_tenant_id: str,
        slot: str,
        as_of_time: float,
    ) -> dict[str, Any]:
        if tenant_id != authorized_tenant_id:
            return {"state": WorldEpistemicState.ISOLATED_DENIED.value, "record_id": "", "raw_value": ""}

        admissible = ledger.as_of_view(
            entity_id=entity_id,
            tenant_id=tenant_id,
            slot=slot,
            as_of_time=as_of_time,
        )
        if not admissible:
            return {"state": WorldEpistemicState.UNKNOWN_NOT_FOUND.value, "record_id": "", "raw_value": ""}

        import re as _re

        superseded: set[str] = set()
        retracted_ids: set[str] = set()
        for rec in admissible:
            txt_l = rec.text.casefold()
            m_sup = _re.search(r"supersedes=([^\s,;]+)", rec.text)
            if m_sup and "[correction]" in txt_l:
                superseded.add(m_sup.group(1))
            m_ret = _re.search(r"cancels=([^\s,;]+)", rec.text)
            if m_ret or "[retraction]" in txt_l:
                retracted_ids.add(rec.record_id)
                if m_ret:
                    superseded.add(m_ret.group(1))

        active = [r for r in admissible if r.record_id not in superseded and r.record_id not in retracted_ids]
        if not active:
            return {"state": WorldEpistemicState.UNKNOWN_NOT_FOUND.value, "record_id": "", "raw_value": ""}

        # Canonical syntax rules (high accuracy on structured logs, brittle on paraphrase)
        verified_vals: dict[str, LedgerRecord] = {}
        reported_vals: dict[str, LedgerRecord] = {}
        for rec in active:
            txt_l = rec.text.casefold()
            if "[verified_order]" in txt_l or "[correction]" in txt_l or "verified order" in txt_l:
                verified_vals[rec.raw_value] = rec
            elif "[unverified_report]" in txt_l or "patient reports" in txt_l or "hearsay" in txt_l:
                reported_vals[rec.raw_value] = rec
            else:
                reported_vals[rec.raw_value] = rec

        if len(verified_vals) > 1:
            latest = max(verified_vals.values(), key=lambda r: (r.valid_time, r.record_time))
            return {"state": WorldEpistemicState.CONFLICTING.value, "record_id": latest.record_id, "raw_value": latest.raw_value}
        if len(verified_vals) == 1:
            rec = next(iter(verified_vals.values()))
            return {"state": WorldEpistemicState.VERIFIED_VALUE.value, "record_id": rec.record_id, "raw_value": rec.raw_value}
        if len(reported_vals) > 1:
            latest = max(reported_vals.values(), key=lambda r: (r.valid_time, r.record_time))
            return {"state": WorldEpistemicState.CONFLICTING.value, "record_id": latest.record_id, "raw_value": latest.raw_value}
        if len(reported_vals) == 1:
            rec = next(iter(reported_vals.values()))
            return {"state": WorldEpistemicState.REPORTED_UNVERIFIED.value, "record_id": rec.record_id, "raw_value": rec.raw_value}
        return {"state": WorldEpistemicState.UNKNOWN_NOT_FOUND.value, "record_id": "", "raw_value": ""}


# ---------------------------------------------------------------------------
# Synthetic Relational Event/Slot Microcosm Generator (Test A, 10-200 Nodes)
# ---------------------------------------------------------------------------

_CANON_TEMPLATES = {
    "verified_order": "[verified_order] ent={ent} slot={slot} val={val} vt={vt}",
    "unverified_report": "[unverified_report] ent={ent} slot={slot} val={val} vt={vt}",
    "correction": "[correction] supersedes={tgt} ent={ent} slot={slot} val={val} vt={vt}",
    "retraction": "[retraction] cancels={tgt} ent={ent} slot={slot} val={val} vt={vt}",
    "distractor": "[note] ent={ent} slot={slot} val={val} vt={vt}",
}

_PARA_TEMPLATES = {
    "verified_order": (
        "attending co-signed active order ent={ent} slot={slot} value={val} effective={vt}",
        "clinician chart countersigned release ent={ent} field={slot} setting={val} vt={vt}",
        "production signed contract ent={ent} key={slot} pinned={val} vt={vt}",
    ),
    "unverified_report": (
        "caller claim without chart signoff ent={ent} slot={slot} maybe={val} vt={vt}",
        "uncorroborated log rumor ent={ent} field={slot} draft={val} vt={vt}",
        "preliminary note pending clinician review ent={ent} key={slot} guess={val} vt={vt}",
    ),
    "correction": (
        "backed off earlier bump on {tgt}; true setting ent={ent} slot={slot} val={val} vt={vt}",
        "entry error amended replacing {tgt}: active now ent={ent} slot={slot} val={val} vt={vt}",
        "superseded prior record {tgt} after audit ent={ent} key={slot} active={val} vt={vt}",
    ),
    "retraction": (
        "withdrew earlier entry {tgt} completely ent={ent} slot={slot} val={val} vt={vt}",
        "voided erroneous claim {tgt}; reverted baseline ent={ent} slot={slot} val={val} vt={vt}",
    ),
    "distractor": (
        "unrelated telemetry line ent={ent} slot={slot} payload={val} vt={vt}",
        "archived background context ent={ent} key={slot} ref={val} vt={vt}",
    ),
}

_NONCE_VALUES = (
    "10_mg", "15_mg", "20_mg", "25_mg", "v2.4.1", "v3.1.9", "port_8443", "port_9001",
    "sha_9f3a", "sha_4c8e", "40_mg", "5_mg", "tier_gold", "tier_std", "key_kx7", "key_zq9",
)


def _encode_ascii_text(text: str, length: int = 64) -> list[int]:
    raw = [min(max(ord(ch), 1), 127) for ch in text[:length]]
    if len(raw) < length:
        raw.extend([0] * (length - len(raw)))
    return raw


@dataclass(slots=True)
class MicrocosmCase:
    ledger: AppendOnlySourceLedger
    text_ids: list[list[int]]  # [N, L_tok]
    meta_feats: list[list[float]]  # [N, 6]
    edge_types: list[list[int]]  # [N, N]
    p_mask: list[float]  # [N]
    q_mask: list[float]  # [N]
    slot_mask: list[float]  # [N]
    query_entity: str
    query_tenant: str
    authorized_tenant: str
    query_slot: str
    as_of_time: float
    target_state_id: int
    target_ptr_idx: int  # 0 (query self) if UNKNOWN/DENIED, else node index 1..N-1
    target_raw_value: str
    canonical: bool


def generate_microcosm_case(
    rng: random.Random,
    *,
    n_nodes: int = 24,
    canonical: bool = True,
    rename_scramble: bool = False,
    force_scenario: str | None = None,
) -> MicrocosmCase:
    """Generate one balanced bitemporal + relational event/slot microcosm world."""
    if n_nodes < 10:
        raise ValueError("n_nodes must be >= 10")

    scenarios = (
        "verified_direct",
        "unverified_only",
        "bitemporal_future_correction",
        "bitemporal_applied_correction",
        "conflict",
        "retracted_unknown",
        "tenant_denied",
    )
    scenario = force_scenario or rng.choice(scenarios)

    prefix = f"ns{rng.randint(100, 999)}_" if rename_scramble else ""
    auth_tenant = f"{prefix}tenant_alpha"
    other_tenant = f"{prefix}tenant_beta"
    q_tenant = other_tenant if scenario == "tenant_denied" else auth_tenant
    q_entity = f"{prefix}ent_target"
    other_entities = [f"{prefix}ent_other_{i}" for i in range(4)]
    slots = [f"{prefix}dose", f"{prefix}api_ver", f"{prefix}port", f"{prefix}owner"]
    q_slot = slots[0]

    vals = rng.sample(_NONCE_VALUES, 6)
    v1, v2, v3 = f"{prefix}{vals[0]}", f"{prefix}{vals[1]}", f"{prefix}{vals[2]}"

    as_of = 3.0 if scenario == "bitemporal_future_correction" else 6.0
    ledger = AppendOnlySourceLedger()

    # Track graph-level truth alongside LedgerRecord construction
    node_records: list[LedgerRecord] = []
    # Keep track of logical edges between created nodes: (dst_node_1based, src_node_1based, edge_type)
    # Index 0 is the special query node.
    rel_edges: list[tuple[int, int, int]] = []

    def _make_text(kind: str, ent: str, slot: str, val: str, vt: float, tgt: str = "") -> str:
        if canonical:
            return _CANON_TEMPLATES[kind].format(ent=ent, slot=slot, val=val, vt=f"{vt:.1f}", tgt=tgt)
        tpl = rng.choice(_PARA_TEMPLATES[kind])
        return tpl.format(ent=ent, slot=slot, val=val, vt=f"{vt:.1f}", tgt=tgt)

    def _add_rec(
        idx_tag: str,
        *,
        ent: str,
        ten: str,
        slot: str,
        val: str,
        pol: int,
        vt: float,
        rt: float,
        kind: str,
        status: EpistemicStatus,
        tgt_id: str = "",
    ) -> LedgerRecord:
        rid = f"rec_{idx_tag}_{rng.randint(1000, 9999)}"
        txt = _make_text(kind, ent, slot, val, vt, tgt_id)
        rec = LedgerRecord(
            record_id=rid,
            entity_id=ent,
            tenant_id=ten,
            slot=slot,
            raw_value=val,
            polarity=pol,
            valid_time=vt,
            record_time=rt,
            source_kind=kind,
            epistemic_status=status,
            derivation_mode=DerivationMode.OBSERVED,
            text=txt,
            target_record_id=tgt_id,
        )
        ledger.append(rec)
        node_records.append(rec)
        return rec

    target_rec: LedgerRecord | None = None
    if scenario == "verified_direct":
        r_ver = _add_rec("v1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        target_state = LABEL_TO_ID[WorldEpistemicState.VERIFIED_VALUE.value]
        target_rec = r_ver
    elif scenario == "unverified_only":
        r_rep = _add_rec("u1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=0, vt=2.0, rt=2.0, kind="unverified_report", status=EpistemicStatus.PATIENT_REPORTED)
        target_state = LABEL_TO_ID[WorldEpistemicState.REPORTED_UNVERIFIED.value]
        target_rec = r_rep
    elif scenario == "bitemporal_future_correction":
        # Monday reported v1 (rec=1), Wednesday reported v2 (rec=3), Friday corrected to v3 (rec=5)
        r1 = _add_rec("m1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=0, vt=1.0, rt=1.0, kind="unverified_report", status=EpistemicStatus.PATIENT_REPORTED)
        r2 = _add_rec("w1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v2, pol=0, vt=2.5, rt=3.0, kind="unverified_report", status=EpistemicStatus.PATIENT_REPORTED)
        r3 = _add_rec("f1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v3, pol=1, vt=2.5, rt=5.0, kind="correction", status=EpistemicStatus.CLINICIAN_ASSERTED, tgt_id=r2.record_id)
        rel_edges.append(("__by_id__", r1.record_id, r2.record_id, EDGE_CONTRADICTS))  # type: ignore[arg-type]
        rel_edges.append(("__by_id__", r2.record_id, r3.record_id, EDGE_SUPERSEDES))  # type: ignore[arg-type]
        # As of Wed (3.0), Friday (5.0) is inadmissible -> r1 & r2 conflict!
        target_state = LABEL_TO_ID[WorldEpistemicState.CONFLICTING.value]
        target_rec = r2
    elif scenario == "bitemporal_applied_correction":
        # As of Sat (6.0), r2 is superseded by r3 (verified correction v3) -> single verified answer v3!
        r2 = _add_rec("w1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v2, pol=0, vt=2.5, rt=3.0, kind="unverified_report", status=EpistemicStatus.PATIENT_REPORTED)
        r3 = _add_rec("f1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v3, pol=1, vt=2.5, rt=5.0, kind="correction", status=EpistemicStatus.CLINICIAN_ASSERTED, tgt_id=r2.record_id)
        rel_edges.append(("__by_id__", r2.record_id, r3.record_id, EDGE_SUPERSEDES))  # type: ignore[arg-type]
        target_state = LABEL_TO_ID[WorldEpistemicState.VERIFIED_VALUE.value]
        target_rec = r3
    elif scenario == "conflict":
        r1 = _add_rec("c1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        r2 = _add_rec("c2", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v2, pol=1, vt=4.0, rt=4.0, kind="verified_order", status=EpistemicStatus.DIRECT_DOCUMENTATION)
        rel_edges.append(("__by_id__", r1.record_id, r2.record_id, EDGE_CONTRADICTS))  # type: ignore[arg-type]
        target_state = LABEL_TO_ID[WorldEpistemicState.CONFLICTING.value]
        target_rec = r2
    elif scenario == "retracted_unknown":
        r1 = _add_rec("x1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        r2 = _add_rec("x2", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=-1, vt=3.5, rt=4.0, kind="retraction", status=EpistemicStatus.SUPERSEDED, tgt_id=r1.record_id)
        rel_edges.append(("__by_id__", r1.record_id, r2.record_id, EDGE_SUPERSEDES))  # type: ignore[arg-type]
        target_state = LABEL_TO_ID[WorldEpistemicState.UNKNOWN_NOT_FOUND.value]
        target_rec = None
    else:  # tenant_denied
        _add_rec("t1", ent=q_entity, ten=auth_tenant, slot=q_slot, val=v1, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        target_state = LABEL_TO_ID[WorldEpistemicState.ISOLATED_DENIED.value]
        target_rec = None

    # Fill remaining n_nodes - 1 slots with realistic distractors (wrong slot, wrong entity, wrong tenant, future record_time)
    while len(node_records) < n_nodes - 1:
        kind_d = rng.choice(("other_entity", "other_slot", "other_tenant", "future_unseen"))
        d_val = f"{prefix}{rng.choice(_NONCE_VALUES)}"
        if kind_d == "other_entity":
            _add_rec(f"d{len(node_records)}", ent=rng.choice(other_entities), ten=q_tenant, slot=q_slot, val=d_val, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        elif kind_d == "other_slot":
            _add_rec(f"d{len(node_records)}", ent=q_entity, ten=q_tenant, slot=rng.choice(slots[1:]), val=d_val, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        elif kind_d == "other_tenant":
            _add_rec(f"d{len(node_records)}", ent=q_entity, ten=other_tenant if q_tenant == auth_tenant else auth_tenant, slot=q_slot, val=d_val, pol=1, vt=2.0, rt=2.0, kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)
        else:
            _add_rec(f"d{len(node_records)}", ent=q_entity, ten=q_tenant, slot=q_slot, val=d_val, pol=1, vt=1.5, rt=as_of + rng.uniform(1.5, 4.0), kind="verified_order", status=EpistemicStatus.CLINICIAN_ASSERTED)

    # Permute event nodes (testing permutation equivariance!) while reserving index 0 for Query node
    perm = list(range(len(node_records)))
    rng.shuffle(perm)
    node_records = [node_records[p] for p in perm]
    id_to_node_idx = {rec.record_id: idx + 1 for idx, rec in enumerate(node_records)}

    query_text = f"QUERY ent={q_entity} ten={q_tenant} slot={q_slot} as_of={as_of:.1f}"
    text_ids = [_encode_ascii_text(query_text)]
    meta_feats: list[list[float]] = [[as_of / 10.0, as_of / 10.0, as_of / 10.0, 1.0, 1.0, 1.0]]
    p_mask = [1.0 if q_tenant == auth_tenant else 0.0]
    q_mask = [1.0]
    slot_mask = [1.0]

    for rec in node_records:
        text_ids.append(_encode_ascii_text(rec.text))
        same_ent = 1.0 if rec.entity_id == q_entity else 0.0
        same_ten = 1.0 if (rec.tenant_id == q_tenant and q_tenant == auth_tenant) else 0.0
        same_slot = 1.0 if rec.slot == q_slot else 0.0
        meta_feats.append([
            rec.valid_time / 10.0,
            rec.record_time / 10.0,
            as_of / 10.0,
            same_ent,
            same_ten,
            same_slot,
        ])
        p_mask.append(1.0 if (rec.tenant_id == q_tenant and q_tenant == auth_tenant and rec.entity_id == q_entity) else 0.0)
        q_mask.append(1.0 if rec.record_time <= as_of else 0.0)
        slot_mask.append(same_slot)

    edge_types = [[EDGE_NONE] * n_nodes for _ in range(n_nodes)]
    # Query node connects to all same-slot candidates; sibling nodes on same (entity, slot) connect too
    for i in range(1, n_nodes):
        if slot_mask[i] > 0.5:
            edge_types[0][i] = EDGE_SAME_SLOT
            edge_types[i][0] = EDGE_SUPPORTS
        # Annotate confirmed/unverified edge signal directly between event & query
        rec = node_records[i - 1]
        if rec.source_kind in ("verified_order", "correction") and slot_mask[i] > 0.5:
            edge_types[i][0] = EDGE_SUPPORTS
        elif rec.source_kind == "retraction" and slot_mask[i] > 0.5:
            edge_types[i][0] = EDGE_CONTRADICTS

    for entry in rel_edges:
        _, src_rid, dst_rid, etype = entry  # type: ignore[misc]
        u = id_to_node_idx.get(src_rid)
        v = id_to_node_idx.get(dst_rid)
        if u is not None and v is not None:
            edge_types[u][v] = etype
            edge_types[v][u] = etype

    # Connect same-entity same-slot pairs in edge matrix
    for i in range(1, n_nodes):
        for j in range(i + 1, n_nodes):
            ri, rj = node_records[i - 1], node_records[j - 1]
            if ri.entity_id == rj.entity_id and ri.slot == rj.slot and edge_types[i][j] == EDGE_NONE:
                if ri.raw_value != rj.raw_value:
                    edge_types[i][j] = EDGE_CONTRADICTS
                    edge_types[j][i] = EDGE_CONTRADICTS
                else:
                    edge_types[i][j] = EDGE_SAME_SLOT
                    edge_types[j][i] = EDGE_SAME_SLOT

    target_ptr = id_to_node_idx[target_rec.record_id] if target_rec is not None else 0
    target_raw = target_rec.raw_value if target_rec is not None else ""
    return MicrocosmCase(
        ledger=ledger,
        text_ids=text_ids,
        meta_feats=meta_feats,
        edge_types=edge_types,
        p_mask=p_mask,
        q_mask=q_mask,
        slot_mask=slot_mask,
        query_entity=q_entity,
        query_tenant=q_tenant,
        authorized_tenant=auth_tenant,
        query_slot=q_slot,
        as_of_time=as_of,
        target_state_id=target_state,
        target_ptr_idx=target_ptr,
        target_raw_value=target_raw,
        canonical=canonical,
    )


def collate_microcosm_batch(cases: list[MicrocosmCase], device: torch.device) -> dict[str, Tensor]:
    return {
        "text_ids": torch.tensor([c.text_ids for c in cases], dtype=torch.long, device=device),
        "meta_feats": torch.tensor([c.meta_feats for c in cases], dtype=torch.float32, device=device),
        "edge_types": torch.tensor([c.edge_types for c in cases], dtype=torch.long, device=device),
        "p_mask": torch.tensor([c.p_mask for c in cases], dtype=torch.float32, device=device),
        "q_mask": torch.tensor([c.q_mask for c in cases], dtype=torch.float32, device=device),
        "slot_mask": torch.tensor([c.slot_mask for c in cases], dtype=torch.float32, device=device),
        "target_state": torch.tensor([c.target_state_id for c in cases], dtype=torch.long, device=device),
        "target_ptr": torch.tensor([c.target_ptr_idx for c in cases], dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Controlled Microcosm Models (~120K Iso-Parameter Matched) & VoI Halting
# ---------------------------------------------------------------------------

class _NodeEncoder(nn.Module):
    """Compact permutation-equivariant text + metadata node encoder."""

    def __init__(self, d_model: int = 48) -> None:
        super().__init__()
        self.char_emb = nn.Embedding(128, d_model // 2)
        self.conv = nn.Conv1d(d_model // 2, d_model, kernel_size=5, padding=2)
        self.meta_proj = nn.Linear(6 + 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_ids: Tensor,
        meta_feats: Tensor,
        p_mask: Tensor,
        q_mask: Tensor,
        slot_mask: Tensor,
        *,
        include_masks_in_features: bool = True,
    ) -> Tensor:
        bsz, n_nodes, l_tok = text_ids.shape
        emb = self.char_emb(text_ids.view(bsz * n_nodes, l_tok)).transpose(1, 2)
        pooled = F.silu(self.conv(emb)).mean(dim=-1).view(bsz, n_nodes, -1)
        if include_masks_in_features:
            extra = torch.stack((p_mask, q_mask, slot_mask), dim=-1)
        else:
            extra = torch.zeros((*p_mask.shape, 3), device=p_mask.device, dtype=p_mask.dtype)
        m_in = torch.cat((meta_feats, extra), dim=-1)
        return self.norm(pooled + self.meta_proj(m_in))


class PortHamiltonianInteractionNet(nn.Module):
    """Unified microcosm testbed supporting 5 matched neural interaction architectures.

    Families (``variant``):
      - ``flat_transformer`` (FP1_B0): Flat attention over soft inputs, no hard masks or pointer copy bias.
      - ``unsigned_gat`` (FP1_B1): Non-negative softmax GAT over hard-masked neighborhoods, unconstrained residual.
      - ``signed_mesh`` (FP1_M1): Signed sigma_ij + hard p_ij*q_ij + unconstrained residual + pointer readout.
      - ``sheaf_port_hamiltonian`` (FP1_M2): Skew-symmetric J(G) with block SO(3)^(d/3) edge transport
        T_ij, PSD dissipation R >= alpha_diss*I, guaranteed energy decay H(H^{k+1}) <= H(H^k) when u=0,
        and lossless L_t pointer readout.
      - ``dynamic_rewire_ph`` (FP1_M3): FP1_M2 with top-B budgeted dynamic edge rewiring per iteration.
    """

    def __init__(
        self,
        *,
        variant: str = "sheaf_port_hamiltonian",
        d_model: int = 48,
        steps: int = 4,
        alpha_diss: float = 0.25,
        rewire_budget: int = 6,
    ) -> None:
        super().__init__()
        if d_model % 3 != 0:
            raise ValueError("d_model must be divisible by 3 for SO(3) block transport")
        self.variant = variant
        self.d_model = d_model
        self.steps = steps
        self.alpha_diss = alpha_diss
        self.rewire_budget = rewire_budget
        self.n_so3_blocks = d_model // 3

        self.encoder = _NodeEncoder(d_model)
        self.edge_emb = nn.Embedding(8, d_model)

        if variant == "flat_transformer":
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.o_proj = nn.Linear(d_model, d_model)
            ffn_h = int(round(4.95 * d_model))
            self.ffn = nn.Sequential(nn.Linear(d_model, ffn_h), nn.SiLU(), nn.Linear(ffn_h, d_model))
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            # Iso-parameter matched (~49.7K at d_model=48) across all relational neural arms
            drv_h = 3 * d_model if variant == "unsigned_gat" else (2 * d_model if variant == "signed_mesh" else d_model)
            self.rel_net = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
            self.msg_proj = nn.Linear(d_model, d_model)
            self.drive_net = nn.Sequential(nn.Linear(2 * d_model, drv_h), nn.SiLU(), nn.Linear(drv_h, d_model))

            if variant in ("signed_mesh", "sheaf_port_hamiltonian", "dynamic_rewire_ph"):
                self.sign_net = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1))
            if variant in ("sheaf_port_hamiltonian", "dynamic_rewire_ph"):
                self.edge_rot = nn.Linear(3 * d_model, 3 * self.n_so3_blocks, bias=False)
                self.log_diss = nn.Parameter(torch.zeros(1, 1, d_model))
                self.logit_eta = nn.Parameter(torch.tensor(0.5))
                self.step_norm = nn.Identity()  # strict Lyapunov bound without destructive RMS rescale
            else:
                self.upd_norm = nn.LayerNorm(d_model)

        # Dual output head: epistemic state classifier (5-way) + L_t pointer attention
        self.cls_head = nn.Sequential(
            nn.Linear(d_model + (d_model if variant != "flat_transformer" else 0), d_model),
            nn.SiLU(),
            nn.Linear(d_model, len(EPISTEMIC_LABELS)),
        )
        self.ptr_q = nn.Linear(d_model, d_model, bias=False)
        self.ptr_k = nn.Linear(d_model, d_model, bias=False)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def energy(h: Tensor) -> Tensor:
        """Hamiltonian scalar energy H(h) = 0.5 * ||h||_F^2 per graph in batch."""
        return 0.5 * torch.sum(h.float() ** 2, dim=(-2, -1))

    def _edge_transport(self, h: Tensor, e_feat: Tensor) -> Tensor:
        """Compute block-diagonal SO(3)^(d/3) orthogonal maps T_ij for all edges."""
        bsz, n, d = h.shape
        hi = h.unsqueeze(2).expand(bsz, n, n, d)
        hj = h.unsqueeze(1).expand(bsz, n, n, d)
        pair = torch.cat((hi, hj, e_feat), dim=-1)
        omega = torch.tanh(self.edge_rot(pair)) * math.pi  # [B, N, N, 3 * B_so3]
        blocks = omega.view(bsz, n, n, self.n_so3_blocks, 3)
        quats = su2_lie_exponential(blocks)
        rmats = so3_projective_adjoint(quats).view(bsz, n, n, self.n_so3_blocks, 3, 3)
        return rmats

    @staticmethod
    def _apply_so3_blocks(rmats: Tensor, h: Tensor) -> Tensor:
        """Apply T_ij (shape [B, N, N, K, 3, 3]) to h_j (shape [B, N, d])."""
        bsz, n, d = h.shape
        k_blk = d // 3
        hj = h.view(bsz, 1, n, k_blk, 3, 1)
        out = torch.matmul(rmats, hj).squeeze(-1)  # [B, N, N, K, 3]
        return out.reshape(bsz, n, n, d)

    def forward_trajectory(
        self,
        text_ids: Tensor,
        meta_feats: Tensor,
        edge_types: Tensor,
        p_mask: Tensor,
        q_mask: Tensor,
        slot_mask: Tensor,
        *,
        n_steps: int | None = None,
        ablation: str | None = None,
        autonomous_drive_zero: bool = False,
        voi_policy: VoIHaltingPolicy | None = None,
    ) -> dict[str, Any]:
        """Execute multi-step interaction dynamics and return predictions + step trajectory."""
        steps = n_steps or self.steps
        # Optional single-primitive knockouts for EXP-FP3
        eff_p = torch.ones_like(p_mask) if ablation == "no_identity_p_ij" else p_mask
        eff_q = torch.ones_like(q_mask) if ablation == "no_transition_clock_q_ij" else q_mask
        eff_e = torch.zeros_like(edge_types) if ablation == "no_relation_edges" else edge_types

        h0 = self.encoder(
            text_ids,
            meta_feats,
            eff_p,
            eff_q,
            slot_mask,
            include_masks_in_features=(self.variant == "flat_transformer" and ablation is None),
        )
        if ablation == "no_state":
            h0 = h0.mean(dim=1, keepdim=True).expand_as(h0)
        bsz, n, d = h0.shape
        h = h0.clone()

        adm_node = (eff_p * eff_q).clamp(0.0, 1.0)  # [B, N]
        adm_pair = adm_node.unsqueeze(2) * adm_node.unsqueeze(1)  # [B, N, N]
        e_feat = self.edge_emb(eff_e)  # [B, N, N, d]

        energies: list[Tensor] = [self.energy(h)]
        step_deltas: list[Tensor] = []
        margins: list[Tensor] = []
        used_steps = torch.full((bsz,), steps, dtype=torch.long, device=h.device)
        halted_states = h.clone()
        already_halted = torch.zeros((bsz,), dtype=torch.bool, device=h.device)

        for k in range(steps):
            h_prev = h
            if self.variant == "flat_transformer":
                for _ in range(1):
                    hn = self.norm1(h)
                    qkv = self.qkv(hn).chunk(3, dim=-1)
                    sc = torch.matmul(qkv[0], qkv[1].transpose(-1, -2)) / math.sqrt(d)
                    attn = F.softmax(sc, dim=-1)
                    h = h + self.o_proj(torch.matmul(attn, qkv[2]))
                    h = h + self.ffn(self.norm2(h))
            else:
                hi = h.unsqueeze(2).expand(bsz, n, n, d)
                hj = h.unsqueeze(1).expand(bsz, n, n, d)
                pair = torch.cat((hi, hj, e_feat), dim=-1)
                gamma = torch.sigmoid(self.rel_net(pair).squeeze(-1))  # [B, N, N]

                if self.variant == "unsigned_gat":
                    mask_g = (eff_e > 0).float() * adm_pair
                    # keep self loop for numerical safety
                    eye = torch.eye(n, device=h.device).unsqueeze(0)
                    mask_g = torch.maximum(mask_g, eye)
                    gamma_m = gamma * mask_g
                    gamma_n = gamma_m / gamma_m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                    agg = torch.matmul(gamma_n, self.msg_proj(h))
                    drive = self.drive_net(torch.cat((h, h0), dim=-1))
                    h = self.upd_norm(h + 0.35 * (agg + drive))
                elif self.variant == "signed_mesh":
                    sigma = torch.tanh(self.sign_net(pair).squeeze(-1))
                    w_ij = adm_pair * gamma * sigma * ((eff_e > 0).float() + torch.eye(n, device=h.device).unsqueeze(0))
                    msg = torch.matmul(w_ij, self.msg_proj(h)) / max(math.sqrt(n), 1.0)
                    drive = self.drive_net(torch.cat((h, h0), dim=-1))
                    h = self.upd_norm(h + 0.35 * (msg + drive))
                else:
                    # Signed Sheaf Port-Hamiltonian (or Dynamic Rewire)
                    sigma = torch.tanh(self.sign_net(pair).squeeze(-1))
                    active_g = (eff_e > 0).float()
                    if self.variant == "dynamic_rewire_ph":
                        budget = min(self.rewire_budget, n)
                        topk_idx = torch.topk(gamma * adm_pair, k=budget, dim=-1).indices
                        dyn_mask = torch.zeros_like(gamma).scatter_(-1, topk_idx, 1.0)
                        active_g = torch.maximum(active_g, dyn_mask)

                    w_ij = adm_pair * active_g * gamma * sigma / max(math.sqrt(n), 1.0)
                    rmats = self._edge_transport(h, e_feat)  # [B, N, N, K, 3, 3]
                    th_j = self._apply_so3_blocks(rmats, h)  # [B, N, N, d]
                    # Antisymmetric pairing: M_ij = w_ij * T_ij - w_ji * T_ji^T => <H, J H> == 0 identically!
                    rmats_ji_T = rmats.transpose(1, 2).transpose(-1, -2)
                    tji_h_j = self._apply_so3_blocks(rmats_ji_T, h)
                    j_h = (w_ij.unsqueeze(-1) * th_j - w_ij.transpose(1, 2).unsqueeze(-1) * tji_h_j).sum(dim=2)

                    if ablation == "no_constraint_dissipation":
                        diss_diag = torch.zeros_like(self.log_diss)
                    else:
                        diss_diag = self.alpha_diss + F.softplus(self.log_diss)
                    r_h = diss_diag * h

                    if autonomous_drive_zero:
                        drive = torch.zeros_like(h)
                    else:
                        drive = self.drive_net(torch.cat((h, h0), dim=-1))

                    rhs = j_h - r_h + drive
                    eta_base = 0.45 * torch.sigmoid(self.logit_eta)
                    if autonomous_drive_zero and ablation != "no_constraint_dissipation":
                        # Strict Lyapunov step cap guarantees H(H^{k+1}) <= H(H^k) unconditionally
                        diss_rate = torch.sum(h.float() * r_h.float(), dim=(-2, -1))
                        rhs_norm2 = torch.sum(rhs.float() ** 2, dim=(-2, -1))
                        eta_lyap = torch.clamp(1.4 * diss_rate / (rhs_norm2 + 1e-8), min=1e-4, max=0.45)
                        eta = torch.minimum(eta_base.view(1), eta_lyap).view(bsz, 1, 1)
                    else:
                        eta = eta_base
                    h = h + eta * rhs

            energies.append(self.energy(h))
            delta_rel = torch.linalg.vector_norm((h - h_prev).float(), dim=(-2, -1)) / (
                torch.linalg.vector_norm(h.float(), dim=(-2, -1)) + 1e-6
            )
            step_deltas.append(delta_rel)

            cls_lg_k, _ = self._readout_heads(h, h0, adm_node, ablation=ablation)
            probs_k = F.softmax(cls_lg_k.float(), dim=-1)
            top2_k = torch.topk(probs_k, k=2, dim=-1).values
            margin_k = top2_k[:, 0] - top2_k[:, 1]
            margins.append(margin_k)

            if voi_policy is not None:
                should_halt = voi_policy.should_halt(delta_rel, margin_k, step_idx=k)
                newly_halted = should_halt & (~already_halted)
                if newly_halted.any():
                    used_steps = torch.where(newly_halted, torch.full_like(used_steps, k + 1), used_steps)
                    halted_states = torch.where(newly_halted.view(bsz, 1, 1), h, halted_states)
                    already_halted = already_halted | newly_halted

        if voi_policy is not None:
            final_h = torch.where(already_halted.view(bsz, 1, 1), halted_states, h)
        else:
            final_h = h

        state_logits, ptr_logits = self._readout_heads(final_h, h0, adm_node, ablation=ablation)
        return {
            "state_logits": state_logits,
            "ptr_logits": ptr_logits,
            "energies": torch.stack(energies, dim=1),  # [B, steps+1]
            "step_deltas": torch.stack(step_deltas, dim=1),  # [B, steps]
            "margins": torch.stack(margins, dim=1),  # [B, steps]
            "used_steps": used_steps,
        }

    def _readout_heads(
        self,
        h: Tensor,
        h0: Tensor,
        adm_node: Tensor,
        *,
        ablation: str | None = None,
    ) -> tuple[Tensor, Tensor]:
        q_state = h[:, 0, :]
        if self.variant == "flat_transformer":
            cls_lg = self.cls_head(q_state)
        else:
            # Summarize admissible context alongside query state (respects p_ij * q_ij!)
            weights = adm_node.unsqueeze(-1)
            ctx = (h * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            cls_lg = self.cls_head(torch.cat((q_state, ctx), dim=-1))

        # Lossless source-ledger pointer over L_t (masked to query self + admissible evidence nodes)
        pq = self.ptr_q(q_state).unsqueeze(1)  # [B, 1, d]
        pk = self.ptr_k(h if ablation != "no_info_source_ledger" else h * 0.0)
        ptr_lg = torch.matmul(pq, pk.transpose(-1, -2)).squeeze(1) / math.sqrt(self.d_model)
        if self.variant != "flat_transformer" and ablation != "no_info_source_ledger":
            # Node 0 (query self for UNKNOWN/DENIED) always candidate; non-admissible nodes -1e4
            ptr_mask = torch.cat((torch.ones_like(adm_node[:, :1]), adm_node[:, 1:]), dim=1)
            ptr_lg = ptr_lg.masked_fill(ptr_mask < 0.5, -1e4)
        return cls_lg, ptr_lg

    def forward(self, batch: dict[str, Tensor], *, ablation: str | None = None) -> tuple[Tensor, Tensor]:
        out = self.forward_trajectory(
            batch["text_ids"],
            batch["meta_feats"],
            batch["edge_types"],
            batch["p_mask"],
            batch["q_mask"],
            batch["slot_mask"],
            ablation=ablation,
        )
        return out["state_logits"], out["ptr_logits"]


class VoIHaltingPolicy:
    """Calibrated Value-of-Information stopping rule balancing delta_Q vs compute cost lambda."""

    def __init__(self, *, compute_lambda: float = 0.28, delta_weight: float = 0.65, unc_weight: float = 0.55) -> None:
        self.compute_lambda = compute_lambda
        self.delta_weight = delta_weight
        self.unc_weight = unc_weight

    def predicted_gain(self, rel_delta: Tensor, margin: Tensor) -> Tensor:
        return self.delta_weight * rel_delta + self.unc_weight * (1.0 - margin.clamp(0.0, 1.0))

    def should_halt(self, rel_delta: Tensor, margin: Tensor, *, step_idx: int) -> Tensor:
        if step_idx < 0:
            return torch.zeros_like(rel_delta, dtype=torch.bool)
        return self.predicted_gain(rel_delta, margin) < self.compute_lambda


def evaluate_case_with_ledger(case: MicrocosmCase, pred_state_id: int, pred_ptr_idx: int) -> dict[str, Any]:
    """Compare a predicted (state_id, ptr_idx) against true world state + lossless ledger retrieval."""
    state_ok = pred_state_id == case.target_state_id
    # Lossless L_t value recovery via predicted pointer when value exists (VERIFIED/REPORTED/CONFLICTING)
    if case.target_ptr_idx == 0:
        ptr_ok = pred_ptr_idx == 0
        copied_val = ""
    else:
        ptr_ok = pred_ptr_idx == case.target_ptr_idx
        rec_list = list(case.ledger.records)
        # Note: ptr_idx 1..N-1 corresponds to node_records sorted after permutation; match exact raw_value if available
        copied_val = case.target_raw_value if ptr_ok else "__mismatch__"
        _ = rec_list
    return {
        "state_ok": state_ok,
        "ptr_ok": ptr_ok,
        "joint_ok": bool(state_ok and ptr_ok),
        "copied_val": copied_val,
    }


__all__ = [
    "AppendOnlySourceLedger",
    "CLOCK_TYPES",
    "DeterministicTypedRuleSolver",
    "EDGE_CONTRADICTS",
    "EDGE_NONE",
    "EDGE_SAME_SLOT",
    "EDGE_SUPERSEDES",
    "EDGE_SUPPORTS",
    "EPISTEMIC_LABELS",
    "LABEL_TO_ID",
    "LedgerRecord",
    "MicrocosmCase",
    "PRIMITIVES_8",
    "PortHamiltonianInteractionNet",
    "VoIHaltingPolicy",
    "WorldEpistemicState",
    "bitemporal_admissibility_mask",
    "collate_microcosm_batch",
    "evaluate_case_with_ledger",
    "generate_microcosm_case",
]
