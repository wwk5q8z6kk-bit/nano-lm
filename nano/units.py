"""Modular Denomination, Dimensional Unit & Scale Algebra — Layer VI, domain-independent.

Replaces brittle domain-specific value string normalization with explicit typed
quantities, interval bounds, scale/multiplier prefixes, dimensional equivalence,
ordinal mappings, role separation (`stock` vs `flow`, `duration` vs `timepoint`),
and evidence-backed conversion edges (never guessing FX rates or densities).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import re
from typing import Iterable


class DenomRelation(str, Enum):
    EQUIVALENT = "equivalent"
    OVERLAPPING = "overlapping"
    CONFLICTING = "conflicting"
    ROLE_MISMATCH = "role_mismatch"
    INCOMPARABLE_DIMENSION = "incomparable_dimension"
    INCOMPARABLE_DENOMINATION = "incomparable_denomination"


@dataclass(frozen=True)
class QuantityValue:
    """Canonical typed measurement, monetary amount, duration, ratio, or ordinal band."""
    raw_text: str
    dimension: str
    unit: str
    denomination: str = ""
    role: str = "scalar"           # scalar | stock | flow | duration | timepoint | rate | ordinal | limit
    base_unit: str = ""
    base_min: float = 0.0
    base_max: float = 0.0
    approximate: bool = False
    canonical_label: str = ""
    conversion_lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.dimension:
            raise ValueError("QuantityValue requires a non-empty dimension")
        if self.base_min > self.base_max + 1e-12:
            raise ValueError(f"invalid interval [{self.base_min}, {self.base_max}]")

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.base_min + self.base_max)

    @property
    def is_point(self) -> bool:
        return abs(self.base_max - self.base_min) <= 1e-9 * max(1.0, abs(self.base_min))


@dataclass(frozen=True)
class ConversionEdge:
    """Observed, verifiable conversion factor between denominations or custom units."""
    source_denom: str
    target_denom: str
    factor: float
    evidence_id: str = ""
    valid_time: str = ""
    source_dimension: str = ""
    target_dimension: str = ""

    def __post_init__(self) -> None:
        if self.factor <= 0.0 or not math.isfinite(self.factor):
            raise ValueError("conversion factor must be finite and positive")


@dataclass(frozen=True)
class DenomComparison:
    relation: DenomRelation
    reason: str
    delta: float | None = None


_WORD_NUMBERS: dict[str, float] = {
    "zero": 0.0, "a": 1.0, "an": 1.0, "half": 0.5, "half a": 0.5, "half an": 0.5,
    "one": 1.0, "two": 2.0, "a couple": 2.0, "a couple of": 2.0, "couple": 2.0,
    "three": 3.0, "a few": 3.0, "several": 4.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0,
    "fifteen": 15.0, "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0,
    "nineteen": 19.0, "twenty": 20.0, "thirty": 30.0, "forty": 40.0,
    "fifty": 50.0, "sixty": 60.0, "seventy": 70.0, "eighty": 80.0,
    "ninety": 90.0, "hundred": 100.0,
}

_MAG_MULTIPLIERS: dict[str, float] = {
    "": 1.0,
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6, "mil": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
}

_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD", "usd": "USD", "us$": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
    "¥": "JPY", "jpy": "JPY", "yen": "JPY",
    "chf": "CHF", "cad": "CAD", "aud": "AUD", "btc": "BTC", "eth": "ETH",
}

_CURRENCY_SUBUNITS: dict[str, tuple[str, float]] = {
    "cent": ("USD", 1e-2), "cents": ("USD", 1e-2), "¢": ("USD", 1e-2),
    "pence": ("GBP", 1e-2), "penny": ("GBP", 1e-2),
    "sat": ("BTC", 1e-8), "sats": ("BTC", 1e-8), "satoshi": ("BTC", 1e-8),
}

# Exact physical / time / binary / ratio unit definitions: unit_key -> (dimension, base_unit, factor, default_role)
_BUILTIN_UNITS: dict[str, tuple[str, str, float, str]] = {
    # Mass (base: g)
    "pg": ("mass", "g", 1e-12, "scalar"),
    "ng": ("mass", "g", 1e-9, "scalar"),
    "ug": ("mass", "g", 1e-6, "scalar"), "mcg": ("mass", "g", 1e-6, "scalar"),
    "mg": ("mass", "g", 1e-3, "scalar"),
    "g": ("mass", "g", 1.0, "scalar"), "gram": ("mass", "g", 1.0, "scalar"), "grams": ("mass", "g", 1.0, "scalar"),
    "kg": ("mass", "g", 1e3, "scalar"), "kilogram": ("mass", "g", 1e3, "scalar"), "kilograms": ("mass", "g", 1e3, "scalar"),
    "t": ("mass", "g", 1e6, "scalar"), "tonne": ("mass", "g", 1e6, "scalar"), "tonnes": ("mass", "g", 1e6, "scalar"),
    "lb": ("mass", "g", 453.59237, "scalar"), "lbs": ("mass", "g", 453.59237, "scalar"),
    "oz": ("mass", "g", 28.349523125, "scalar"),
    # Volume (base: L)
    "ul": ("volume", "L", 1e-6, "scalar"), "mcl": ("volume", "L", 1e-6, "scalar"),
    "ml": ("volume", "L", 1e-3, "scalar"),
    "l": ("volume", "L", 1.0, "scalar"), "liter": ("volume", "L", 1.0, "scalar"), "liters": ("volume", "L", 1.0, "scalar"),
    # Length (base: m)
    "mm": ("length", "m", 1e-3, "scalar"),
    "cm": ("length", "m", 1e-2, "scalar"),
    "m": ("length", "m", 1.0, "scalar"), "meter": ("length", "m", 1.0, "scalar"), "meters": ("length", "m", 1.0, "scalar"),
    "km": ("length", "m", 1e3, "scalar"),
    # Data size (base: byte; explicit SI 1000^k vs IEC 1024^k)
    "b": ("data_size", "byte", 1.0, "scalar"), "byte": ("data_size", "byte", 1.0, "scalar"), "bytes": ("data_size", "byte", 1.0, "scalar"),
    "kb": ("data_size", "byte", 1e3, "scalar"),
    "mb": ("data_size", "byte", 1e6, "scalar"),
    "gb": ("data_size", "byte", 1e9, "scalar"),
    "tb": ("data_size", "byte", 1e12, "scalar"),
    "kib": ("data_size", "byte", float(1024), "scalar"),
    "mib": ("data_size", "byte", float(1024 ** 2), "scalar"),
    "gib": ("data_size", "byte", float(1024 ** 3), "scalar"),
    "tib": ("data_size", "byte", float(1024 ** 4), "scalar"),
    # Time duration (base: second)
    "ms": ("time_duration", "s", 1e-3, "duration"), "millisecond": ("time_duration", "s", 1e-3, "duration"), "milliseconds": ("time_duration", "s", 1e-3, "duration"),
    "s": ("time_duration", "s", 1.0, "duration"), "sec": ("time_duration", "s", 1.0, "duration"), "secs": ("time_duration", "s", 1.0, "duration"),
    "second": ("time_duration", "s", 1.0, "duration"), "seconds": ("time_duration", "s", 1.0, "duration"),
    "min": ("time_duration", "s", 60.0, "duration"), "mins": ("time_duration", "s", 60.0, "duration"),
    "minute": ("time_duration", "s", 60.0, "duration"), "minutes": ("time_duration", "s", 60.0, "duration"),
    "h": ("time_duration", "s", 3600.0, "duration"), "hr": ("time_duration", "s", 3600.0, "duration"), "hrs": ("time_duration", "s", 3600.0, "duration"),
    "hour": ("time_duration", "s", 3600.0, "duration"), "hours": ("time_duration", "s", 3600.0, "duration"),
    "d": ("time_duration", "s", 86400.0, "duration"), "day": ("time_duration", "s", 86400.0, "duration"), "days": ("time_duration", "s", 86400.0, "duration"),
    "wk": ("time_duration", "s", 7 * 86400.0, "duration"), "wks": ("time_duration", "s", 7 * 86400.0, "duration"),
    "week": ("time_duration", "s", 7 * 86400.0, "duration"), "weeks": ("time_duration", "s", 7 * 86400.0, "duration"),
    "mo": ("time_duration", "s", 30.0 * 86400.0, "duration"), "mos": ("time_duration", "s", 30.0 * 86400.0, "duration"),
    "month": ("time_duration", "s", 30.0 * 86400.0, "duration"), "months": ("time_duration", "s", 30.0 * 86400.0, "duration"),
    "yr": ("time_duration", "s", 365.0 * 86400.0, "duration"), "yrs": ("time_duration", "s", 365.0 * 86400.0, "duration"),
    "year": ("time_duration", "s", 365.0 * 86400.0, "duration"), "years": ("time_duration", "s", 365.0 * 86400.0, "duration"),
    # Dimensionless ratios (base: ratio)
    "%": ("ratio", "ratio", 1e-2, "rate"), "pct": ("ratio", "ratio", 1e-2, "rate"), "percent": ("ratio", "ratio", 1e-2, "rate"),
    "bps": ("ratio", "ratio", 1e-4, "rate"), "bp": ("ratio", "ratio", 1e-4, "rate"),
    "permille": ("ratio", "ratio", 1e-3, "rate"), "‰": ("ratio", "ratio", 1e-3, "rate"),
}

_APPROX_PREFIX_RE = re.compile(r"^(?:about|around|roughly|nearly|almost|over|approx(?:\.|imately)?|circa|~)\s+", re.IGNORECASE)
_BOUND_SUFFIX_RE = re.compile(r"(?:\s+(?:ago|so far|now|total|approx(?:\.|imately)?))+$", re.IGNORECASE)
_TIMEPOINT_RE = re.compile(
    r"^(?:since\s+|from\s+|as of\s+|on\s+)?"
    r"(?P<anchor>yesterday(?:\s+(?:morning|afternoon|evening|night))?"
    r"|today(?:\s+(?:morning|afternoon|evening))?"
    r"|tonight|last night|this (?:morning|afternoon|evening)"
    r"|(?:the\s+)?(?:morning|afternoon|evening|night)\s+before\s+last"
    r"|(?:last\s+|this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?"
    r"|after\s+(?:breakfast|lunch|dinner|supper|waking\s+up|eating)(?:\s+[^.!?;]+)?"
    r"|noon|midnight|\d{4}-\d{2}-\d{2}|q[1-4]\s*\d{4}"
    r"|[a-z0-9 ,'-]{2,40})"
    r"(?:,\s*(?:after|before|when|while)\s+[^.!?;]+)?$",
    re.IGNORECASE,
)


def _parse_number_token(text: str) -> float | None:
    cleaned = text.strip().lower().replace(",", "")
    if cleaned in _WORD_NUMBERS:
        return _WORD_NUMBERS[cleaned]
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class DenominationRegistry:
    """Modular, runtime-extensible unit, currency, ordinal, and conversion-edge engine."""
    custom_units: dict[str, tuple[str, str, float, str]] = field(default_factory=dict)
    custom_currencies: dict[str, str] = field(default_factory=dict)
    conversions: list[ConversionEdge] = field(default_factory=list)

    def register_unit(self, symbol: str, dimension: str, base_unit: str, factor: float, role: str = "scalar") -> None:
        if factor <= 0.0:
            raise ValueError("unit factor must be positive")
        self.custom_units[symbol.strip().lower()] = (dimension, base_unit, float(factor), role)

    def register_currency(self, token: str, code: str) -> None:
        self.custom_currencies[token.strip().lower()] = code.strip().upper()

    def register_conversion(self, edge: ConversionEdge) -> None:
        self.conversions.append(edge)

    def _lookup_unit(self, token: str) -> tuple[str, str, float, str] | None:
        key = token.strip().lower()
        return self.custom_units.get(key) or _BUILTIN_UNITS.get(key)

    def _lookup_currency(self, token: str) -> str | None:
        key = token.strip().lower()
        return self.custom_currencies.get(key) or _CURRENCY_SYMBOLS.get(key)

    def _currency_prefix_pattern(self) -> str:
        """Build regex matching ONLY registered currency symbols or codes (never verbs/prepositions like 'is')."""
        tokens = set(_CURRENCY_SYMBOLS.keys()) | set(self.custom_currencies.keys())
        sym_toks: list[str] = []
        word_toks: list[str] = []
        for tok in tokens:
            if not tok:
                continue
            if tok.isalpha():
                word_toks.append(re.escape(tok))
            else:
                sym_toks.append(re.escape(tok))
        sym_toks.sort(key=len, reverse=True)
        word_toks.sort(key=len, reverse=True)
        parts: list[str] = []
        if sym_toks:
            parts.append("(?:%s)" % "|".join(sym_toks))
        if word_toks:
            parts.append(r"\b(?:%s)\b(?=\s+)" % "|".join(word_toks))
        return "|".join(parts) if parts else r"[$€£¥]"

    def parse(
        self,
        text: str,
        *,
        expected_dimension: str = "",
        role: str = "",
        ordinal_scale: Iterable[tuple[str, float, float]] = (),
    ) -> QuantityValue | None:
        """Parse formatted quantities, currencies/denominations, temporal values, or ordinals."""
        raw = text.strip()
        if not raw:
            return None
        allowed_dims = {d.strip() for d in expected_dimension.split("|") if d.strip()}

        work = raw
        if not allowed_dims or "time_duration" in allowed_dims:
            work = re.sub(r"^for\s+", "", work, flags=re.IGNORECASE)
        approx = False
        m_ap = _APPROX_PREFIX_RE.match(work)
        if m_ap:
            approx = True
            work = work[m_ap.end():].strip()
        work = _BOUND_SUFFIX_RE.sub("", work).strip()
        work_low = work.lower()

        # 1. Ordinal scale check if caller/schema provides one (e.g. severity or priority)
        if ordinal_scale or "ordinal" in allowed_dims:
            m_out_of = re.fullmatch(
                r"(?P<num>\d+(?:\.\d+)?|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
                r"\s*(?:out of|/)\s*(?P<den>\d+(?:\.\d+)?|ten|five|hundred)"
                r"(?:\s*(?:,\s*[^.!?;]+|\s+(?:when|while|especially|mostly|if|at)\s+[^.!?;]+))?",
                work_low,
            )
            if m_out_of:
                n_val = _parse_number_token(m_out_of.group("num"))
                d_val = _parse_number_token(m_out_of.group("den"))
                if n_val is not None and d_val and d_val > 0:
                    ratio = n_val / d_val
                    return QuantityValue(
                        raw_text=raw, dimension="ordinal", unit=f"/{int(d_val)}",
                        denomination="", role=role or "ordinal",
                        base_unit="norm01", base_min=ratio, base_max=ratio,
                        approximate=approx, canonical_label=f"{n_val:g}/{d_val:g}",
                    )
            for label_pat, lo, hi in ordinal_scale:
                if re.search(rf"\b{label_pat}\b", work_low):
                    return QuantityValue(
                        raw_text=raw, dimension="ordinal", unit="band",
                        denomination="", role=role or "ordinal",
                        base_unit="norm01", base_min=float(lo), base_max=float(hi),
                        approximate=True, canonical_label=label_pat,
                    )

        # 2. Timepoint check (e.g. "Yesterday", "since yesterday", "Q1 2026", "last night", "since eating lunch")
        if not allowed_dims or "time_point" in allowed_dims:
            m_tp = _TIMEPOINT_RE.fullmatch(work_low)
            if m_tp:
                # Arbitrary `[a-z0-9 ,'-]{2,40}` anchor ONLY valid when explicitly introduced by `since|from|as of`
                has_tp_prep = bool(re.match(r"^(?:since|from|as of)\s+", work_low))
                anchor_raw = re.sub(r"\s+", " ", m_tp.group("anchor").strip().lower())
                is_named_anchor = bool(
                    re.fullmatch(
                        r"yesterday(?:\s+(?:morning|afternoon|evening|night))?|today(?:\s+(?:morning|afternoon|evening))?"
                        r"|tonight|last night|this (?:morning|afternoon|evening)|(?:the\s+)?(?:morning|afternoon|evening|night)\s+before\s+last"
                        r"|(?:last\s+|this\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?"
                        r"|after\s+(?:breakfast|lunch|dinner|supper|waking\s+up|eating)(?:\s+[^.!?;]+)?"
                        r"|noon|midnight|\d{4}-\d{2}-\d{2}|q[1-4]\s*\d{4}",
                        anchor_raw,
                    )
                )
                if is_named_anchor or has_tp_prep:
                    return QuantityValue(
                        raw_text=raw, dimension="time_point", unit="anchor",
                        denomination="", role=role or "timepoint",
                        base_unit="anchor", base_min=0.0, base_max=0.0,
                        approximate=approx, canonical_label=anchor_raw,
                    )

        # 3. Currency / Denomination amounts: "$1.5M", "USD 1,200k", "1500k EUR", "250 cents"
        cur_pre_pat = self._currency_prefix_pattern()
        cur_pat = (
            rf"^(?P<pre>{cur_pre_pat})?\s*"
            r"(?P<num>-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|a couple of|half an?|a few|several|[a-z]+)"
            r"(?:(?<=\d)(?P<mult_tight>[kmbt])|\s+(?P<mult_word>thousand|million|billion|trillion|mil|mn|bn))?"
            r"\s*(?P<post>us\$|[$€£¥]|cents?|pence|penny|sats?|satoshi|dollars?|euros?|pounds?|yen|[A-Za-z%‰]+)?"
            r"(?:\s*/\s*(?P<per>[A-Za-z]+))?$"
        )
        m_q = re.fullmatch(cur_pat, work_low)
        if m_q:
            num_val = _parse_number_token(m_q.group("num"))
            pre = (m_q.group("pre") or "").strip()
            mult_tok = (m_q.group("mult_tight") or m_q.group("mult_word") or "").strip()
            post = (m_q.group("post") or "").strip()
            per = (m_q.group("per") or "").strip()

            if num_val is not None:
                # Check currency subunit first ("250 cents")
                if not pre and post in _CURRENCY_SUBUNITS:
                    denom, sub_factor = _CURRENCY_SUBUNITS[post]
                    mult = _MAG_MULTIPLIERS.get(mult_tok, 1.0)
                    val = num_val * mult * sub_factor
                    lo, hi = (val * 0.8, val * 1.2) if approx else (val, val)
                    return QuantityValue(
                        raw_text=raw, dimension="currency", unit=post, denomination=denom,
                        role=role or "scalar", base_unit=denom, base_min=lo, base_max=hi,
                        approximate=approx, canonical_label=f"{val:g} {denom}",
                    )

                # Check currency symbol before or after ("$1.5M", "1500k EUR", "USD 100")
                denom_pre = self._lookup_currency(pre) if pre else None
                denom_post = self._lookup_currency(post) if post else None
                if denom_pre or denom_post:
                    denom = denom_pre or denom_post or ""
                    if denom_pre and denom_post and denom_pre != denom_post:
                        return None
                    mult = _MAG_MULTIPLIERS.get(mult_tok, 1.0)
                    val = num_val * mult
                    lo, hi = (val * 0.85, val * 1.15) if approx else (val, val)
                    return QuantityValue(
                        raw_text=raw, dimension="currency", unit=denom, denomination=denom,
                        role=role or "scalar", base_unit=denom, base_min=lo, base_max=hi,
                        approximate=approx, canonical_label=f"{val:g} {denom}",
                    )

                # Check physical / binary / duration / ratio unit (only if no currency prefix!)
                if not pre:
                    unit_tok = post
                    uinfo = self._lookup_unit(unit_tok) if unit_tok else None
                    mult = _MAG_MULTIPLIERS.get(mult_tok, 1.0) if (mult_tok and unit_tok) else 1.0
                    if uinfo is not None:
                        dim, base_u, u_factor, def_role = uinfo
                        if allowed_dims and dim not in allowed_dims:
                            pass
                        else:
                            val = num_val * mult * u_factor
                            lo, hi = (val * 0.8, val * 1.2) if approx else (val, val)
                            final_role = role or def_role
                            if per:
                                dim = f"{dim}/{per}"
                                base_u = f"{base_u}/{per}"
                            return QuantityValue(
                                raw_text=raw, dimension=dim, unit=unit_tok, denomination="",
                                role=final_role, base_unit=base_u, base_min=lo, base_max=hi,
                                approximate=approx, canonical_label=f"{val:g} {base_u}",
                            )
                    elif "count" in allowed_dims and not post:
                        val = num_val * _MAG_MULTIPLIERS.get(mult_tok, 1.0)
                        return QuantityValue(
                            raw_text=raw, dimension="count", unit="count", denomination="",
                            role=role or "scalar", base_unit="count", base_min=val, base_max=val,
                            approximate=approx, canonical_label=f"{val:g}",
                        )

        return None

    def extract_spans(
        self,
        text: str,
        *,
        expected_dimension: str = "",
        role: str = "",
        ordinal_scale: Iterable[tuple[str, float, float]] = (),
    ) -> list[tuple[int, int, QuantityValue]]:
        """Find and parse candidate quantity substrings `(start, end, QuantityValue)` inside free prose."""
        found: list[tuple[int, int, QuantityValue]] = []
        # Scan general quantitative/currency/temporal/ordinal token spans without hardcoded domain nouns
        allowed_dims = {d.strip() for d in expected_dimension.split("|") if d.strip()}
        cur_pre_pat = self._currency_prefix_pattern()
        ord_alts = [rf"\b(?:{pat})\b" for pat, _, _ in ordinal_scale if pat]
        ord_block = ("|" + "|".join(ord_alts)) if ord_alts else ""
        tp_extra = (
            r"|\b(?:since|from)\s+[^.!?;]+"
            r"|\bafter\s+(?:breakfast|lunch|dinner|supper|waking\s+up|eating)(?:\s+(?:this\s+\w+|today|yesterday))?"
            r"|\b(?:(?:last|this)\s+)?(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)(?:\s+(?:morning|afternoon|evening|night))?\b"
            r"|\b(?:the\s+)?(?:morning|afternoon|evening|night)\s+before\s+last\b"
        ) if (not allowed_dims or "time_point" in allowed_dims) else ""
        cand_pat = re.compile(
            r"(?:(?:about|around|roughly|nearly|almost|over|approx\.?|~)\s+)?"
            rf"(?:(?:{cur_pre_pat})\s*)?"
            r"(?:(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)|a couple of|half an?|a few|several|"
            r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|an?)\b|"
            r"\bsince\s+(?:yesterday|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b|"
            r"\byesterday(?:\s+(?:morning|afternoon|evening|night))?\b|\btoday(?:\s+(?:morning|afternoon|evening))?\b|"
            r"\b(?:tonight|last night|this morning|this afternoon|this evening|noon|midnight)\b|"
            rf"\b(?:mild(?:ly uncomfortable)?|moderate|severe)\b{tp_extra}{ord_block})"
            r"(?:(?<=\d)[kmbt]|\s+(?:thousand|million|billion|trillion|mil|mn|bn))?"
            r"(?:\s*(?:out of\s*(?:10|5|100|ten)|/\s*(?:10|5|100)|us\$|[$€£¥]|cents?|pence|sats?|dollars?|euros?|pounds?|yen|[A-Za-z%‰]+(?:\s*/\s*[A-Za-z]+)?))?"
            r"(?:\s+(?:ago|so far|now))?",
            re.IGNORECASE,
        )
        occupied: list[tuple[int, int]] = []
        for m in cand_pat.finditer(text):
            s0, e0 = m.start(), m.end()
            sub = m.group().strip(" ,.;:!?")
            if not sub:
                continue
            # Adjust offsets cleanly after trim
            lead = len(m.group()) - len(m.group().lstrip(" ,.;:!?"))
            trail = len(m.group()) - len(m.group().rstrip(" ,.;:!?"))
            ns, ne = s0 + lead, e0 - trail
            if any(a <= ns and ne <= b for a, b in occupied):
                continue
            q = self.parse(sub, expected_dimension=expected_dimension, role=role, ordinal_scale=ordinal_scale)
            if q is not None:
                found.append((ns, ne, q))
                occupied.append((ns, ne))
        return found

    def convert(self, q: QuantityValue, target_denom: str) -> QuantityValue | None:
        """Convert `q` only through an explicit observed ConversionEdge (never guess FX)."""
        tgt = target_denom.strip().upper()
        if q.denomination == tgt or not tgt:
            return q
        for edge in self.conversions:
            if edge.source_denom.upper() == q.denomination.upper() and edge.target_denom.upper() == tgt:
                if edge.source_dimension and edge.source_dimension != q.dimension:
                    continue
                new_dim = edge.target_dimension or q.dimension
                return QuantityValue(
                    raw_text=q.raw_text, dimension=new_dim, unit=tgt, denomination=tgt,
                    role=q.role, base_unit=tgt,
                    base_min=q.base_min * edge.factor, base_max=q.base_max * edge.factor,
                    approximate=q.approximate, canonical_label=f"{q.midpoint * edge.factor:g} {tgt}",
                    conversion_lineage=(*q.conversion_lineage, edge.evidence_id or f"{edge.source_denom}->{tgt}"),
                )
            if edge.target_denom.upper() == q.denomination.upper() and edge.source_denom.upper() == tgt:
                if edge.target_dimension and edge.target_dimension != q.dimension:
                    continue
                new_dim = edge.source_dimension or q.dimension
                inv = 1.0 / edge.factor
                return QuantityValue(
                    raw_text=q.raw_text, dimension=new_dim, unit=tgt, denomination=tgt,
                    role=q.role, base_unit=tgt,
                    base_min=q.base_min * inv, base_max=q.base_max * inv,
                    approximate=q.approximate, canonical_label=f"{q.midpoint * inv:g} {tgt}",
                    conversion_lineage=(*q.conversion_lineage, edge.evidence_id or f"{edge.target_denom}->{tgt}"),
                )
        return None

    def compare(
        self,
        a: QuantityValue,
        b: QuantityValue,
        *,
        rel_tol: float = 1e-4,
        allow_mixed_roles: tuple[frozenset[str], ...] = (),
    ) -> DenomComparison:
        """Compare two parsed QuantityValues cleanly across scales & denominations."""
        if a.dimension != b.dimension:
            return DenomComparison(
                DenomRelation.INCOMPARABLE_DIMENSION,
                f"dimension mismatch: {a.dimension} vs {b.dimension}",
            )

        if a.role != b.role and not any({a.role, b.role} <= grp for grp in allow_mixed_roles):
            return DenomComparison(
                DenomRelation.ROLE_MISMATCH,
                f"role mismatch: {a.role} vs {b.role}",
            )

        b_conv = b
        if a.denomination != b.denomination:
            converted = self.convert(b, a.denomination)
            if converted is None:
                return DenomComparison(
                    DenomRelation.INCOMPARABLE_DENOMINATION,
                    f"no observed conversion edge between {a.denomination!r} and {b.denomination!r}",
                )
            b_conv = converted

        if a.dimension == "time_point":
            if a.canonical_label == b_conv.canonical_label:
                return DenomComparison(DenomRelation.EQUIVALENT, f"same timepoint {a.canonical_label!r}", delta=0.0)
            return DenomComparison(
                DenomRelation.CONFLICTING,
                f"timepoint mismatch: {a.canonical_label!r} vs {b_conv.canonical_label!r}",
            )

        # Interval equivalence & overlap with numeric tolerance
        scale = max(1.0, abs(a.midpoint), abs(b_conv.midpoint))
        tol = rel_tol * scale
        if abs(a.base_min - b_conv.base_min) <= tol and abs(a.base_max - b_conv.base_max) <= tol:
            return DenomComparison(DenomRelation.EQUIVALENT, "values equivalent within tolerance", delta=abs(a.midpoint - b_conv.midpoint))
        if min(a.base_max, b_conv.base_max) + tol >= max(a.base_min, b_conv.base_min):
            return DenomComparison(DenomRelation.OVERLAPPING, "intervals overlap", delta=abs(a.midpoint - b_conv.midpoint))
        return DenomComparison(
            DenomRelation.CONFLICTING,
            f"disjoint intervals [{a.base_min:g}, {a.base_max:g}] vs [{b_conv.base_min:g}, {b_conv.base_max:g}] {a.base_unit}",
            delta=abs(a.midpoint - b_conv.midpoint),
        )

    def sum_quantities(
        self,
        items: Iterable[QuantityValue],
        *,
        target_denomination: str = "",
    ) -> tuple[QuantityValue | None, str]:
        """Sum compatible quantities across mixed scales/denominations, failing closed on missing edges."""
        seq = list(items)
        if not seq:
            return None, "empty_quantity_list"
        ref = seq[0]
        tgt_denom = target_denomination or ref.denomination
        total_min = 0.0
        total_max = 0.0
        lineage: list[str] = []
        any_approx = False
        for q in seq:
            if q.dimension != ref.dimension:
                return None, f"dimension_mismatch:{ref.dimension}:{q.dimension}"
            qc = self.convert(q, tgt_denom) if q.denomination != tgt_denom else q
            if qc is None:
                return None, f"missing_conversion:{q.denomination}->{tgt_denom}"
            total_min += qc.base_min
            total_max += qc.base_max
            any_approx = any_approx or qc.approximate
            lineage.extend(qc.conversion_lineage)
        return QuantityValue(
            raw_text="+".join(x.raw_text for x in seq),
            dimension=ref.dimension,
            unit=tgt_denom or ref.base_unit,
            denomination=tgt_denom,
            role=ref.role,
            base_unit=tgt_denom or ref.base_unit,
            base_min=total_min,
            base_max=total_max,
            approximate=any_approx,
            canonical_label=f"{0.5 * (total_min + total_max):g} {tgt_denom or ref.base_unit}".strip(),
            conversion_lineage=tuple(dict.fromkeys(lineage)),
        ), "ok"
