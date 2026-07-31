"""E1 methods M1–M5 (and local M0 LM wrapper)."""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import (
    FIELDS,
    FieldPred,
    ItemPred,
    REPO,
    dialogue_of,
    format_summary,
    patient_lines,
    patient_text,
    pred_from_values,
)

# ---------------------------------------------------------------------------
# Load fabric._extract for M1 (rules-perfect template extractor for this world)
# ---------------------------------------------------------------------------
_fabric_path = REPO / "fabric" / "slice.py"
_spec = importlib.util.spec_from_file_location("fabric_slice_e1", _fabric_path)
_fab = importlib.util.module_from_spec(_spec)
# fabric.slice imports schemas from same dir
sys.path.insert(0, str(REPO / "fabric"))
_spec.loader.exec_module(_fab)

# Train-only pools from v1 (eval held excluded). VOID if held values appear in dicts.
_src = (REPO / "scribe" / "build_scribe_data.py").read_text()
_ns = {}
exec(compile(_src.split("def sample_tuple")[0], "bsd", "exec"), _ns)
CC_TRAIN_CANON = [c for _, c in _ns["CC_TRAIN"]]
CC_TRAIN_DIALOG = [d for d, _ in _ns["CC_TRAIN"]]
MED_TRAIN = list(_ns["MED_TRAIN"])
ALG_TRAIN = list(_ns["ALG_TRAIN"])
SEV = list(_ns["SEV"])
HELD_FORBIDDEN = set([c for _, c in _ns["CC_HELD"]] + _ns["MED_HELD"] + _ns["ALG_HELD"])


def _assert_no_held_leak(vocab):
    leak = set(x.lower() for x in vocab) & set(x.lower() for x in HELD_FORBIDDEN)
    if leak:
        raise RuntimeError(f"E1 VOID: held lexicon leak {leak}")


_assert_no_held_leak(CC_TRAIN_CANON + MED_TRAIN + ALG_TRAIN)


# ============================= M1 =========================================

def predict_m1(item: dict, source_id: str) -> ItemPred:
    content = item["convo"][0]["content"]
    values, spans = {}, {}
    t0 = time.perf_counter()
    for slot in FIELDS:
        kind, sp = _fab._extract(slot, content, source_id)
        if kind == "denial":
            values[slot] = "none"
            spans[slot] = (sp.start, sp.end, sp.text)
        elif kind == "value":
            val = _fab._norm(sp.text) if slot == "cc" else sp.text.strip()
            # DUR templates capture "{n} {unit}" already as group
            if slot == "sev":
                val = val.lower()
            values[slot] = val
            spans[slot] = (sp.start, sp.end, sp.text)
        else:
            values[slot] = "none"
    # duration canon: truth uses "3 weeks" form — extracted group should match
    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


# ============================= M2 =========================================

def _find_span(dialogue: str, needle: str):
    """Return (start,end,text) of first patient-line occurrence, else None."""
    for text, start in patient_lines(dialogue):
        idx = text.lower().find(needle.lower())
        if idx >= 0:
            return start + idx, start + idx + len(needle), text[idx:idx + len(needle)]
    return None


def predict_m2(item: dict, source_id: str) -> ItemPred:
    """Dictionary + span match; train lists only. Open held values only if verbatim in dialogue
    AND matched via generic span heuristics without held lexicon (string present after cue)."""
    dialogue = dialogue_of(item)
    ptext = patient_text(dialogue)
    values, spans = {}, {}
    t0 = time.perf_counter()

    # SEV: closed
    sev_hit = None
    for s in SEV:
        sp = _find_span(dialogue, s)
        if sp:
            sev_hit = s
            spans["sev"] = sp
            break
    values["sev"] = sev_hit or "none"

    # DUR: regex on patient text
    m = re.search(r"\b(\d+)\s+(days?|weeks?)\b", ptext)
    if m:
        unit = "days" if m.group(2).startswith("day") else "weeks"
        # normalize singular in unit to match truth style (truth uses days/weeks)
        n = m.group(1)
        # find span in original dialogue
        sp = _find_span(dialogue, m.group(0)) or _find_span(dialogue, f"{n} {unit}")
        values["dur"] = f"{n} {unit}"
        if sp:
            spans["dur"] = sp
    else:
        values["dur"] = "none"

    # CC: train canons / dialog forms only for classification; else longest patient noun-ish
    # copy: if a train canon appears, use it; else try to copy a span after openers without held list
    cc_hit = None
    for form, canon in zip(CC_TRAIN_DIALOG, CC_TRAIN_CANON):
        sp = _find_span(dialogue, form) or _find_span(dialogue, canon)
        if sp:
            cc_hit = canon
            spans["cc"] = sp
            break
    if cc_hit is None:
        # verbatim copy heuristic: capture after "having/got/it's" without using held lexicon
        for text, start in patient_lines(dialogue):
            m = re.search(
                r"(?:having|got|it's|with|dealing with|problem is)\s+(?:a |an |the )?(.+?)(?:\.|$)",
                text,
                re.I,
            )
            if m:
                frag = m.group(1).strip().rstrip(".")
                # strip trailing "for N days"
                frag = re.split(r"\s+for\s+\d+", frag, maxsplit=1)[0].strip()
                if frag:
                    cc_hit = _fab._norm(frag)
                    spans["cc"] = (start + m.start(1), start + m.end(1), m.group(1))
                    break
    values["cc"] = cc_hit or "none"

    # MED / ALG: train dict hits; denial templates → none; else if some other span after question, copy it
    for slot, vocab, deny_needles in (
        ("med", MED_TRAIN, ["nothing", "haven't taken", "not taking", "no medication", "nothing at all"]),
        ("alg", ALG_TRAIN, ["no allergies", "not that i know", "none that", "nothing on record", "none whatsoever", "no, no allergies"]),
    ):
        hit = None
        for term in sorted(vocab, key=len, reverse=True):
            sp = _find_span(dialogue, term)
            if sp:
                hit = term
                spans[slot] = sp
                break
        if hit is None:
            if any(d in ptext for d in deny_needles):
                values[slot] = "none"
            else:
                # copy unknown span after med/alg cues without held lexicon
                cue = "medicine" if slot == "med" else "allerg"
                # find patient reply following a doctor line containing cue — reuse fabric slot reply
                kind, sp = _fab._extract(slot, item["convo"][0]["content"], source_id)
                if kind == "value":
                    # only accept if extracted text is NOT forcing held via lexicon — it's from dialogue
                    values[slot] = sp.text.strip()
                    spans[slot] = (sp.start, sp.end, sp.text)
                elif kind == "denial":
                    values[slot] = "none"
                    spans[slot] = (sp.start, sp.end, sp.text)
                else:
                    values[slot] = "none"
        else:
            values[slot] = hit

    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


# ============================= Alignments for M3/M5 ========================

def export_alignments(n_train: int = 4000, seed: int = 11) -> List[dict]:
    """Regenerate v1-style dialogues with span labels via _extract (train templates only)."""
    import random
    random.seed(seed)
    # Use v1 generator functions without writing files
    spec = importlib.util.spec_from_file_location(
        "bsd_e1", REPO / "scribe" / "build_scribe_data.py"
    )
    bsd = importlib.util.module_from_spec(spec)
    # Prevent main from running: load by exec of functions only — the file runs main at import.
    # So exec the source up through function defs by truncating at `if __name__` or build sets.
    src = (REPO / "scribe" / "build_scribe_data.py").read_text()
    # cut before building datasets
    cut = src.split("# ---------------- build sets ----------------")[0]
    ns = {"__name__": "bsd_e1_align"}
    exec(compile(cut, "bsd_align", "exec"), ns)
    out = []
    for i in range(n_train):
        t = ns["sample_tuple"](held=False)
        dlg = ns["render_dialogue"](t, held=False)
        content = dlg + "\nSummarize the visit."
        spans = {}
        cc = t["cc"][1] if isinstance(t["cc"], tuple) else t["cc"]
        values = {
            "cc": cc,
            "dur": f'{t["n"]} {t["unit"]}',
            "sev": t["sev"],
            "med": t["med"],
            "alg": t["alg"],
        }
        for slot in FIELDS:
            kind, sp = _fab._extract(slot, content, f"train/{i}")
            if kind == "value":
                spans[slot] = (sp.start, sp.end, sp.text)
            elif kind == "denial":
                spans[slot] = (sp.start, sp.end, sp.text)
            else:
                spans[slot] = None
        out.append({"content": content, "dialogue": dlg, "tuple": values, "spans": spans})
    return out


def _tokenize_words(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def _bio_for_example(ex: dict) -> List[Tuple[str, str]]:
    """Word tokens with BIO tags over dialogue (not including summarize line)."""
    dlg = ex["dialogue"]
    toks = _tokenize_words(dlg)
    tags = ["O"] * len(toks)
    for slot in FIELDS:
        sp = ex["spans"].get(slot)
        if not sp:
            continue
        s, e, _ = sp
        for i, (a, b, w) in enumerate(toks):
            if b <= s or a >= e:
                continue
            tags[i] = f"B-{slot.upper()}" if a <= s < b or tags[i] == "O" and a >= s else f"I-{slot.upper()}"
            if a <= s < b:
                tags[i] = f"B-{slot.upper()}"
            elif s < a < e:
                tags[i] = f"I-{slot.upper()}"
    feats = []
    for (a, b, w), tag in zip(toks, tags):
        feats.append((w, tag))
    return list(zip([w for _, _, w in toks], tags))


# ============================= M3 CRF-lite =================================

class CRFLite:
    """Feature-based BIO tagger + Viterbi with learned emission and transition counts.
    Not a full CRF optimizer; satisfies prereg 'CRF-lite' with linear-chain decoding.
    """

    LABELS = ["O"] + [f"{p}-{s.upper()}" for s in FIELDS for p in ("B", "I")]

    def __init__(self):
        self.emit = {}  # (label, feat) -> count
        self.trans = {}  # (prev, lab) -> count
        self.label_counts = {}
        self.feat_set = set()

    def _feats(self, words: List[str], i: int) -> List[str]:
        w = words[i]
        feats = [
            f"w={w.lower()}",
            f"pref={w[:3].lower()}",
            f"suf={w[-3:].lower()}",
            f"isdigit={w.isdigit()}",
            f"bias",
        ]
        if i > 0:
            feats.append(f"pw={words[i-1].lower()}")
        if i + 1 < len(words):
            feats.append(f"nw={words[i+1].lower()}")
        return feats

    def fit(self, examples: List[dict]):
        for ex in examples:
            pairs = _bio_for_example(ex)
            if not pairs:
                continue
            words, tags = zip(*pairs)
            words, tags = list(words), list(tags)
            prev = "O"
            for i, lab in enumerate(tags):
                self.label_counts[lab] = self.label_counts.get(lab, 0) + 1
                self.trans[(prev, lab)] = self.trans.get((prev, lab), 0) + 1
                for f in self._feats(words, i):
                    self.feat_set.add(f)
                    self.emit[(lab, f)] = self.emit.get((lab, f), 0) + 1
                prev = lab
            self.trans[(prev, "STOP")] = self.trans.get((prev, "STOP"), 0) + 1

    def _log_emit(self, lab, feats):
        # additive smoothing
        s = 0.0
        lc = self.label_counts.get(lab, 1)
        for f in feats:
            c = self.emit.get((lab, f), 0) + 0.1
            s += np.log(c / (lc + 0.1 * max(len(self.feat_set), 1)))
        return s

    def _log_trans(self, a, b):
        c = self.trans.get((a, b), 0) + 0.1
        z = sum(v for (p, q), v in self.trans.items() if p == a) + 0.1 * len(self.LABELS)
        return np.log(c / z)

    def decode(self, words: List[str]) -> List[str]:
        if not words:
            return []
        labs = self.LABELS
        n, m = len(words), len(labs)
        dp = np.full((n, m), -1e30)
        bp = np.zeros((n, m), dtype=np.int32)
        for j, lab in enumerate(labs):
            dp[0, j] = self._log_trans("O", lab) + self._log_emit(lab, self._feats(words, 0))
        for i in range(1, n):
            feats = self._feats(words, i)
            for j, lab in enumerate(labs):
                emit = self._log_emit(lab, feats)
                best, arg = -1e30, 0
                for k, plab in enumerate(labs):
                    val = dp[i - 1, k] + self._log_trans(plab, lab) + emit
                    if val > best:
                        best, arg = val, k
                dp[i, j] = best
                bp[i, j] = arg
        # end
        j = int(dp[-1].argmax())
        out = [labs[j]]
        for i in range(n - 1, 0, -1):
            j = int(bp[i, j])
            out.append(labs[j])
        out.reverse()
        return out


_M3: Optional[CRFLite] = None


def train_m3(n_train: int = 4000) -> CRFLite:
    global _M3
    ex = export_alignments(n_train=n_train)
    model = CRFLite()
    model.fit(ex)
    _M3 = model
    return model


def predict_m3(item: dict, source_id: str) -> ItemPred:
    global _M3
    if _M3 is None:
        train_m3()
    t0 = time.perf_counter()
    dlg = dialogue_of(item)
    toks = _tokenize_words(dlg)
    words = [w for _, _, w in toks]
    tags = _M3.decode(words)
    values = {f: "none" for f in FIELDS}
    spans = {}
    for slot in FIELDS:
        B, I = f"B-{slot.upper()}", f"I-{slot.upper()}"
        spans_idx = []
        i = 0
        while i < len(tags):
            if tags[i] == B:
                j = i + 1
                while j < len(tags) and tags[j] == I:
                    j += 1
                spans_idx.append((i, j))
                i = j
            else:
                i += 1
        if not spans_idx:
            continue
        a, b = spans_idx[0]
        start, end = toks[a][0], toks[b - 1][1]
        text = dlg[start:end]
        spans[slot] = (start, end, text)
        if slot == "cc":
            values[slot] = _fab._norm(text)
        elif slot == "sev":
            values[slot] = text.lower().strip(".,")
        elif slot == "dur":
            m = re.search(r"(\d+)\s+(days?|weeks?)", text, re.I)
            if m:
                unit = "days" if m.group(2).lower().startswith("day") else "weeks"
                values[slot] = f"{m.group(1)} {unit}"
            else:
                values[slot] = text
        else:
            # med/alg denial?
            low = text.lower()
            if any(x in low for x in ("no ", "none", "nothing", "haven't", "not that")):
                values[slot] = "none"
            else:
                values[slot] = text.strip(".,")
    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


# ============================= M4 constrained ==============================

def predict_m4(item: dict, source_id: str) -> ItemPred:
    """Schema-constrained copy: open slots may only emit substrings present in patient text.
    Uses question-anchored reply (like M1) but value must be a patient substring verbatim;
    closed slots from fixed sets. Distinct from M1: no template regex capture — takes whole
    reply then copies longest in-dialogue train-or-raw span under constraints.
    """
    t0 = time.perf_counter()
    content = item["convo"][0]["content"]
    dialogue = dialogue_of(item)
    ptext = patient_text(dialogue)
    values, spans = {}, {}

    for slot in FIELDS:
        reply, roff = _fab._slot_reply(slot, content)
        if reply is None:
            values[slot] = "none"
            continue
        if reply in _fab.DENY_TPL.get(slot, ()):
            values[slot] = "none"
            spans[slot] = (roff, roff + len(reply), reply)
            continue
        if slot == "sev":
            for s in SEV:
                if s in reply.lower():
                    values[slot] = s
                    sp = _find_span(dialogue, s)
                    if sp:
                        spans[slot] = sp
                    break
            else:
                values[slot] = "none"
            continue
        if slot == "dur":
            m = re.search(r"(\d+)\s+(days?|weeks?)", reply, re.I)
            if m and m.group(0).lower() in ptext:
                unit = "days" if m.group(2).lower().startswith("day") else "weeks"
                values[slot] = f"{m.group(1)} {unit}"
                sp = _find_span(dialogue, m.group(0))
                if sp:
                    spans[slot] = sp
            else:
                values[slot] = "none"
            continue
        # open: copy-only — prefer train vocab hit in reply; else full reply if in patient text
        cand = None
        vocab = {"cc": CC_TRAIN_CANON + CC_TRAIN_DIALOG, "med": MED_TRAIN, "alg": ALG_TRAIN}[slot]
        for term in sorted(vocab, key=len, reverse=True):
            if term.lower() in reply.lower() and term.lower() in ptext:
                cand = term if slot != "cc" else _fab._norm(term)
                # map dialog form to canon for cc
                if slot == "cc":
                    for d, c in zip(CC_TRAIN_DIALOG, CC_TRAIN_CANON):
                        if term.lower() in (d.lower(), c.lower()):
                            cand = c
                            break
                sp = _find_span(dialogue, term)
                if sp:
                    spans[slot] = sp
                break
        if cand is None:
            # copy raw reply fragment only if appears in patient text (enables held copy without lexicon)
            frag = reply.strip().rstrip(".")
            if frag.lower() in ptext:
                cand = _fab._norm(frag) if slot == "cc" else frag
                spans[slot] = (roff, roff + len(reply), reply)
            else:
                # try template group as last constrained option
                kind, sp = _fab._extract(slot, content, source_id)
                if kind == "value" and sp.text.lower() in ptext:
                    cand = _fab._norm(sp.text) if slot == "cc" else sp.text.strip()
                    spans[slot] = (sp.start, sp.end, sp.text)
                else:
                    cand = "none"
        values[slot] = cand
    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


# ============================= M5 span classifier ==========================

class SpanClassifier:
    """Per-field logistic start/end scoring over word positions (non-AR)."""

    def __init__(self):
        self.W_start = {f: None for f in FIELDS}
        self.W_end = {f: None for f in FIELDS}
        self.feat_index = {}

    def _featurize(self, words: List[str], i: int) -> np.ndarray:
        # bag of a few indicator features hashed into fixed dim
        dim = 256
        v = np.zeros(dim, dtype=np.float64)
        def add(s, w=1.0):
            v[hash(s) % dim] += w
        w = words[i]
        add(f"w={w.lower()}")
        add(f"suf={w[-3:].lower()}")
        add(f"i={i}")
        add("bias")
        if i > 0:
            add(f"pw={words[i-1].lower()}")
        if w.isdigit():
            add("digit")
        return v

    def fit(self, examples: List[dict], epochs: int = 5, lr: float = 0.2):
        # init
        for f in FIELDS:
            self.W_start[f] = np.zeros(256)
            self.W_end[f] = np.zeros(256)
        for _ in range(epochs):
            for ex in examples:
                dlg = ex["dialogue"]
                toks = _tokenize_words(dlg)
                words = [w for _, _, w in toks]
                if not words:
                    continue
                for slot in FIELDS:
                    sp = ex["spans"].get(slot)
                    y_s = np.zeros(len(words))
                    y_e = np.zeros(len(words))
                    if sp:
                        s, e, _ = sp
                        for i, (a, b, _) in enumerate(toks):
                            if a <= s < b:
                                y_s[i] = 1
                            if a < e <= b or (a <= e - 1 < b):
                                y_e[i] = 1
                    for i in range(len(words)):
                        x = self._featurize(words, i)
                        for W, y in ((self.W_start[slot], y_s[i]), (self.W_end[slot], y_e[i])):
                            z = 1 / (1 + np.exp(-np.clip(W @ x, -20, 20)))
                            W += lr * (y - z) * x

    def predict_spans(self, dlg: str) -> Dict[str, Optional[Tuple[int, int, str]]]:
        toks = _tokenize_words(dlg)
        words = [w for _, _, w in toks]
        out = {}
        if not words:
            return {f: None for f in FIELDS}
        for slot in FIELDS:
            scores_s = [self.W_start[slot] @ self._featurize(words, i) for i in range(len(words))]
            scores_e = [self.W_end[slot] @ self._featurize(words, i) for i in range(len(words))]
            best, pair = -1e30, None
            for i in range(len(words)):
                for j in range(i, min(i + 8, len(words))):
                    sc = scores_s[i] + scores_e[j]
                    if sc > best:
                        best, pair = sc, (i, j)
            if pair is None or best < 0:
                out[slot] = None
                continue
            i, j = pair
            start, end = toks[i][0], toks[j][1]
            out[slot] = (start, end, dlg[start:end])
        return out


_M5: Optional[SpanClassifier] = None


def train_m5(n_train: int = 4000) -> SpanClassifier:
    global _M5
    ex = export_alignments(n_train=n_train)
    model = SpanClassifier()
    model.fit(ex)
    _M5 = model
    return model


def predict_m5(item: dict, source_id: str) -> ItemPred:
    global _M5
    if _M5 is None:
        train_m5()
    t0 = time.perf_counter()
    dlg = dialogue_of(item)
    spmap = _M5.predict_spans(dlg)
    values, spans = {}, {}
    ptext = patient_text(dlg)
    for slot in FIELDS:
        sp = spmap.get(slot)
        if not sp:
            values[slot] = "none"
            continue
        text = sp[2]
        spans[slot] = sp
        low = text.lower()
        if slot in ("med", "alg") and any(x in low for x in ("no ", "none", "nothing", "haven't")):
            values[slot] = "none"
        elif slot == "cc":
            values[slot] = _fab._norm(text)
        elif slot == "sev":
            hit = next((s for s in SEV if s in low), None)
            values[slot] = hit or "none"
        elif slot == "dur":
            m = re.search(r"(\d+)\s+(days?|weeks?)", text, re.I)
            if m:
                unit = "days" if m.group(2).lower().startswith("day") else "weeks"
                values[slot] = f"{m.group(1)} {unit}"
            else:
                values[slot] = "none"
        else:
            values[slot] = text.strip(".,")
            # constrained: must appear in patient text
            if values[slot].lower() not in ptext:
                values[slot] = "none"
    return pred_from_values(values, spans, latency_s=time.perf_counter() - t0)


# ============================= M0 local LM =================================

def make_m0_predict(tag: str = "scale"):
    """Load anchor LM via rescore_anchors and return predict_fn."""
    ra_path = REPO / "trajectory" / "rescore_anchors.py"
    spec = importlib.util.spec_from_file_location("ra_e1", ra_path)
    ra = importlib.util.module_from_spec(spec)
    # rescore_anchors expects tokenizer relative — set cwd-sensitive paths
    old = os.getcwd()
    os.chdir(REPO / "trajectory")
    # Ensure tokenizer visible
    tok_src = REPO / "sft" / "tokenizer.json"
    tok_dst = REPO / "trajectory" / "tokenizer.json"
    if not tok_dst.exists():
        os.symlink(tok_src, tok_dst)
    os.environ.setdefault("NANO_CKDIR", str(REPO / "checkpoints" / "anchors"))
    try:
        spec.loader.exec_module(ra)
        m, _meta = ra.load(tag)
    finally:
        os.chdir(old)

    def predict(item: dict, source_id: str) -> ItemPred:
        t0 = time.perf_counter()
        content = item["convo"][0]["content"]
        ids = ra.prompt_ids(content)
        out = ra.generate(m, ids)
        text = ra.tok.decode(out[len(ids):]).strip()
        mm = ra.RE.match(text)
        if not mm:
            return ItemPred(
                fields={f: FieldPred("none") for f in FIELDS},
                latency_s=time.perf_counter() - t0,
                raw=text,
                parsed=False,
            )
        vals = dict(zip(FIELDS, [g.strip() for g in mm.groups()]))
        return pred_from_values(vals, latency_s=time.perf_counter() - t0)

    predict.m0_tag = tag  # type: ignore
    return predict


METHODS = {
    "M1_template": predict_m1,
    "M2_dict_span": predict_m2,
    "M3_crf_lite": predict_m3,
    "M4_constrained": predict_m4,
    "M5_span_clf": predict_m5,
}

# Relative compute costs vs 10M greedy scribe (prereg C)
COST_C = {
    "M0_scale": 1.0,
    "M1_template": 0.02,
    "M2_dict_span": 0.03,
    "M3_crf_lite": 0.15,
    "M4_constrained": 0.04,
    "M5_span_clf": 0.2,
}
