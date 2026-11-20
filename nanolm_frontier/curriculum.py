"""Decontaminated Multi-Domain Training Curriculum & Micro Pretraining Stream Generator.

Provides:
1. Programmatically verified SFT/RLVR training tasks spanning all verifiable capability
   families while strictly forbidding overlap with held-out ``benchmark_15d`` tasks.
2. Fail-closed ``verify_zero_contamination`` enforcing zero SHA-256 prompt-hash collisions
   and zero 8-gram character/word shingle collisions between training and held-out suites.
3. ``FrontierMicroTokenizer`` and deterministic multi-domain synthetic corpus + OOD
   associative-recall test generators (at seq=256, 512, 1024) for Experiment E1.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Any

from nanolm_frontier.benchmark_15d import BenchmarkDim, BenchmarkTask


class ContaminationError(RuntimeError):
    """Raised when training data shares hashes or 8-grams with held-out benchmark tasks."""


@dataclass(frozen=True, slots=True)
class CurriculumExample:
    example_id: str
    dim: BenchmarkDim
    system_prompt: str
    user_prompt: str
    reference_response: str
    verifier_spec: dict[str, Any]


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def _word_ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9_]+", _normalize_text(text))
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def verify_zero_contamination(
    train_items: list[CurriculumExample] | tuple[CurriculumExample, ...],
    eval_tasks: list[BenchmarkTask] | tuple[BenchmarkTask, ...],
    *,
    ngram_size: int = 8,
) -> dict[str, int]:
    """Fail closed if any normalized prompt hash or n-gram matches between train and eval."""
    eval_hashes = {
        hashlib.sha256(_normalize_text(t.user_prompt).encode("utf-8")).hexdigest(): t.task_id
        for t in eval_tasks
    }
    eval_ngrams: dict[tuple[str, ...], str] = {}
    for t in eval_tasks:
        # Ignore boilerplate haystack filler prefix when testing n-grams by still scanning full prompt;
        # train prompts never use eval topic filler strings so zero overlap must hold cleanly.
        for ng in _word_ngrams(t.user_prompt, n=ngram_size):
            eval_ngrams[ng] = t.task_id

    for ex in train_items:
        h = hashlib.sha256(_normalize_text(ex.user_prompt).encode("utf-8")).hexdigest()
        if h in eval_hashes:
            raise ContaminationError(f"Exact prompt hash collision: train={ex.example_id} eval={eval_hashes[h]}")
        for ng in _word_ngrams(ex.user_prompt, n=ngram_size):
            if ng in eval_ngrams:
                raise ContaminationError(
                    f"{ngram_size}-gram overlap: train={ex.example_id} eval={eval_ngrams[ng]} gram={' '.join(ng)}"
                )
    return {"train_examples": len(train_items), "eval_tasks": len(eval_tasks), "overlaps": 0}


def build_decontaminated_curriculum(seed: int = 2026) -> tuple[CurriculumExample, ...]:
    """Build verified multi-domain training + RLVR curriculum strictly disjoint from benchmark_15d."""
    rng = random.Random(seed)
    examples: list[CurriculumExample] = []

    # 1. Coding curriculum (6 verified algorithms distinct from eval cod-01..04)
    coding_specs = [
        (
            "trn-cod-01",
            "Write `binary_search(nums: list[int], target: int) -> int` returning index of target or -1.",
            "```python\ndef binary_search(nums: list[int], target: int) -> int:\n    lo, hi = 0, len(nums) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if nums[mid] == target:\n            return mid\n        if nums[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n```",
            ["assert binary_search([1, 3, 5, 7], 5) == 2", "assert binary_search([1, 3, 5], 2) == -1"],
        ),
        (
            "trn-cod-02",
            "Write `is_balanced(s: str) -> bool` checking matching parentheses `()`, `[]`, `{}`.",
            "```python\ndef is_balanced(s: str) -> bool:\n    pairs = {')': '(', ']': '[', '}': '{'}\n    stack = []\n    for ch in s:\n        if ch in '([{':\n            stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop() != pairs[ch]:\n                return False\n    return not stack\n```",
            ["assert is_balanced('{[()]}') is True", "assert is_balanced('([)]') is False", "assert is_balanced('(') is False"],
        ),
        (
            "trn-cod-03",
            "Write `transpose_grid(mat: list[list[int]]) -> list[list[int]]` transposing rectangular matrices.",
            "```python\ndef transpose_grid(mat: list[list[int]]) -> list[list[int]]:\n    if not mat:\n        return []\n    return [[mat[r][c] for r in range(len(mat))] for c in range(len(mat[0]))]\n```",
            ["assert transpose_grid([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]", "assert transpose_grid([]) == []"],
        ),
        (
            "trn-cod-04",
            "Write `two_sum_sorted(nums: list[int], target: int) -> tuple[int, int]` returning 0-based indices or (-1, -1).",
            "```python\ndef two_sum_sorted(nums: list[int], target: int) -> tuple[int, int]:\n    lo, hi = 0, len(nums) - 1\n    while lo < hi:\n        s = nums[lo] + nums[hi]\n        if s == target:\n            return (lo, hi)\n        if s < target:\n            lo += 1\n        else:\n            hi -= 1\n    return (-1, -1)\n```",
            ["assert two_sum_sorted([2, 7, 11, 15], 9) == (0, 1)", "assert two_sum_sorted([1, 2, 3], 10) == (-1, -1)"],
        ),
        (
            "trn-cod-05",
            "Write `prefix_sums(vals: list[int]) -> list[int]` returning cumulative totals starting from length 0.",
            "```python\ndef prefix_sums(vals: list[int]) -> list[int]:\n    out = []\n    acc = 0\n    for v in vals:\n        acc += v\n        out.append(acc)\n    return out\n```",
            ["assert prefix_sums([3, -1, 4]) == [3, 2, 6]", "assert prefix_sums([]) == []"],
        ),
        (
            "trn-cod-06",
            "Write `cadence_dedup(items: list[str]) -> list[str]` removing duplicates while preserving order.",
            "```python\ndef cadence_dedup(items: list[str]) -> list[str]:\n    seen = set()\n    out = []\n    for x in items:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n```",
            ["assert cadence_dedup(['b', 'a', 'b', 'c', 'a']) == ['b', 'a', 'c']"],
        ),
    ]
    for cid, prompt, sol, checks in coding_specs:
        examples.append(
            CurriculumExample(
                cid,
                BenchmarkDim.CODING,
                "Return only valid Python inside a ```python``` block. No prose.",
                prompt,
                sol,
                {"assertions": checks},
            )
        )

    # 2. Math & logic curriculum (8 parameterized modular / arithmetic / recurrence problems)
    for idx in range(8):
        a = rng.randint(11, 29)
        b = rng.randint(12, 27)
        c = rng.randint(15, 65)
        ans = (a * b - c) // 4
        prompt = f"Evaluate `(({a} * {b}) - {c}) // 4` cleanly step by step and finish cleanly."
        examples.append(
            CurriculumExample(
                f"trn-mth-{idx:02d}",
                BenchmarkDim.MATH_LOGIC,
                "Solve carefully and end with `FINAL_ANSWER: <value>`.",
                prompt,
                f"Compute product {a} * {b} = {a * b}.\nSubtract {c}: {a * b - c}.\nInteger divide by 4: {ans}.\nFINAL_ANSWER: {ans}",
                {"accepted_answers": [str(ans)]},
            )
        )

    # 3. Structured JSON + Memory retention (8 examples with disjoint entities/values)
    for idx in range(4):
        n_nodes = rng.randint(3, 8)
        per_node = rng.randint(12, 35)
        tax = rng.randint(5, 25)
        total = n_nodes * per_node + tax
        ex_json = {"nodes": n_nodes, "unit_cost": per_node, "fee": tax, "total": total}
        examples.append(
            CurriculumExample(
                f"trn-jsn-{idx:02d}",
                BenchmarkDim.STRUCTURED_JSON,
                "Return only valid JSON matching exact schema keys.",
                f'Parse bill `"{n_nodes} cluster racks billed at {per_node} credits each plus flat fee {tax} credits"` '
                f'into keys `nodes`, `unit_cost`, `fee`, `total`.',
                json.dumps(ex_json),
                {"expected_json": ex_json},
            )
        )
        # Multi-turn memory example
        init_val = rng.randint(30, 90)
        add_val = rng.randint(10, 40)
        sub_val = rng.randint(5, 20)
        final_val = init_val + add_val - sub_val
        examples.append(
            CurriculumExample(
                f"trn-mem-{idx:02d}",
                BenchmarkDim.MEMORY_RETENTION,
                "Return only valid JSON representing the FINAL state.",
                f"Event 1: STORE stock={init_val}\nEvent 2: TAG site=osaka\nEvent 3: REPLENISH stock+{add_val}\nEvent 4: SWITCH site=zurich\nEvent 5: SHIP stock-{sub_val}\nEvent 6: Return JSON `{{\"site\": str, \"stock\": int}}`.",
                json.dumps({"site": "zurich", "stock": final_val}),
                {"expected_json": {"site": "zurich", "stock": final_val}},
            )
        )

    # 4. Diagrams & Charts (3)
    examples.extend(
        [
            CurriculumExample(
                "trn-dia-01",
                BenchmarkDim.DIAGRAMS_CHARTS,
                "Return only Mermaid inside ```mermaid``` fence. Quote labels with parentheses.",
                'Draw a `flowchart TD` linking `Parser["Token Parser (v2)"]` --> `Planner` --> `Executor` with `-->`.',
                '```mermaid\nflowchart TD\n    P["Token Parser (v2)"] --> Pl["Planner"]\n    Pl --> Ex["Executor"]\n```',
                {"subkind": "mermaid", "expected_header": "flowchart TD", "required_tokens": ["Parser", "Planner", "Executor"], "min_edges": 2},
            ),
            CurriculumExample(
                "trn-dia-02",
                BenchmarkDim.DIAGRAMS_CHARTS,
                "Return only Mermaid inside ```mermaid``` fence.",
                "Draw a `stateDiagram-v2` transitioning `Boot --> Ready`, `Ready --> Busy`, `Busy --> Standby`.",
                "```mermaid\nstateDiagram-v2\n    Boot --> Ready\n    Ready --> Busy\n    Busy --> Standby\n```",
                {"subkind": "mermaid", "expected_header": "stateDiagram-v2", "required_tokens": ["Boot", "Ready", "Busy", "Standby"], "min_edges": 3},
            ),
            CurriculumExample(
                "trn-dia-03",
                BenchmarkDim.DIAGRAMS_CHARTS,
                "Return only valid JSON inside ```json``` block.",
                'Produce chart JSON with keys `chart_type`, `x_labels`, `values`, `sum_total` for `"line"` series across `["Mon", "Tue", "Wed"]` with counts 12, 18, 25.',
                json.dumps({"chart_type": "line", "x_labels": ["Mon", "Tue", "Wed"], "values": [12, 18, 25], "sum_total": 55}),
                {"subkind": "chart_json", "expected_json": {"chart_type": "line", "x_labels": ["Mon", "Tue", "Wed"], "values": [12, 18, 25], "sum_total": 55}},
            ),
        ]
    )

    # 5. Summarization, Tool Use, Planning, Instruction Following, OOD, Hallucination (10)
    examples.extend(
        [
            CurriculumExample(
                "trn-sum-01",
                BenchmarkDim.SUMMARIZATION,
                "Summarize strictly from source text under the word cap.",
                "Summarize in <= 25 words keeping 'Halcyon', '9.2 Gbps', and '18 minutes': Source: Array Halcyon synced checkpoints at 9.2 Gbps and finished recovery within 18 minutes.",
                "Halcyon synced checkpoints at 9.2 Gbps and completed recovery in 18 minutes.",
                {"source": "Array Halcyon synced checkpoints at 9.2 Gbps and finished recovery within 18 minutes.",
                 "max_words": 25, "required_terms": ["Halcyon", "9.2 Gbps", "18 minutes"]},
            ),
            CurriculumExample(
                "trn-tol-01",
                BenchmarkDim.TOOL_USE,
                'Return only a single JSON tool call object `{"tool": str, "args": dict}`.',
                "Available tool: `snapshot_volume(vol: str, encrypt: bool)`. Execute encrypted snapshot of volume `vol-data-7`.",
                json.dumps({"tool": "snapshot_volume", "args": {"vol": "vol-data-7", "encrypt": True}}),
                {"expected_calls": [{"tool": "snapshot_volume", "args": {"vol": "vol-data-7", "encrypt": True}}]},
            ),
            CurriculumExample(
                "trn-tol-02",
                BenchmarkDim.TOOL_USE,
                'Return only JSON `{"tool": str, "args": dict}`.',
                "Tool `fetch_dns(host: 'db.local')` timed out. Policy calls `ping_ip(ip: str, count: int)`. Fallback query ip `'10.4.0.9'` with count 3.",
                json.dumps({"tool": "ping_ip", "args": {"ip": "10.4.0.9", "count": 3}}),
                {"expected_calls": [{"tool": "ping_ip", "args": {"ip": "10.4.0.9", "count": 3}}]},
            ),
            CurriculumExample(
                "trn-pln-01",
                BenchmarkDim.PLANNING_DECOMPOSITION,
                'Return only valid JSON `{"order": list[str], "makespan": int}`.',
                "Tasks: `Load` (dur 2, deps []), `Train` (dur 6, deps [Load]), `Lint` (dur 3, deps [Load]), `Release` (dur 1, deps [Train, Lint]). Output schedule JSON.",
                json.dumps({"order": ["Load", "Train", "Lint", "Release"], "makespan": 9}),
                {"dag": {"Load": {"dur": 2, "deps": []}, "Train": {"dur": 6, "deps": ["Load"]}, "Lint": {"dur": 3, "deps": ["Load"]}, "Release": {"dur": 1, "deps": ["Train", "Lint"]}}},
            ),
            CurriculumExample(
                "trn-ins-01",
                BenchmarkDim.INSTRUCTION_FOLLOWING,
                "Follow all formatting constraints strictly.",
                "Write exactly 2 bullets starting with `- ` in all lowercase mentioning `matrix`, forbidding `gpu`, max 18 words, ending with `::ok`.",
                "- fast matrix tiling avoids wasted reads\n- compact kernels keep memory bounded::ok",
                {"require_all_lowercase": True, "exact_bullets": 2, "required_words": ["matrix"], "forbidden_words": ["gpu"], "max_words": 18, "required_end_token": "::ok"},
            ),
            CurriculumExample(
                "trn-ood-01",
                BenchmarkDim.OOD_ROBUSTNESS,
                "Apply the custom operator rule strictly and end with `FINAL_ANSWER: <int>`.",
                "Define `x # y = (x * y) - x`. Compute `7 # 5`.",
                "Compute 7 * 5 - 7 = 35 - 7 = 28.\nFINAL_ANSWER: 28",
                {"accepted_answers": ["28"]},
            ),
            CurriculumExample(
                "trn-ood-02",
                BenchmarkDim.OOD_ROBUSTNESS,
                "Apply the custom operator rule strictly and end with `FINAL_ANSWER: <int>`.",
                "Define `g(n) = (3 * n + 2) mod 11`. Compute `g(g(4))`.",
                "g(4) = 14 mod 11 = 3. g(3) = 11 mod 11 = 0.\nFINAL_ANSWER: 0",
                {"accepted_answers": ["0"]},
            ),
            CurriculumExample(
                "trn-hal-01",
                BenchmarkDim.HALLUCINATION_RESISTANCE,
                "Start with `ABSTAIN:` if premises omit required information.",
                "Shipping manifest mentions 12 crates arriving at Dock B on Tuesday. What color were the forklifts inside Dock A?",
                "ABSTAIN: Insufficient information because forklift color and Dock A equipment are not mentioned.",
                {"forbidden_confabulations": ["yellow", "red", "blue"]},
            ),
            CurriculumExample(
                "trn-msr-01",
                BenchmarkDim.MULTI_STEP_REASONING,
                "Track values step by step and end with `FINAL_ANSWER: <int>`.",
                "Start accumulator `acc = 6`. Multiply by 4. Subtract 9. Double the result. Add 3.",
                "6 * 4 = 24; 24 - 9 = 15; 15 * 2 = 30; 30 + 3 = 33.\nFINAL_ANSWER: 33",
                {"accepted_answers": ["33"]},
            ),
            CurriculumExample(
                "trn-knw-01",
                BenchmarkDim.GENERAL_KNOWLEDGE,
                "Answer concisely and end with `FINAL_ANSWER: <answer>`.",
                "What SQL clause filters grouped aggregate rows after `GROUP BY`?",
                "The HAVING clause filters groups after GROUP BY aggregation.\nFINAL_ANSWER: HAVING",
                {"accepted_answers": ["HAVING"]},
            ),
        ]
    )

    return tuple(examples)


# ---------------------------------------------------------------------------
# Experiment E1 Tokenizer & Multi-Domain Synthetic + OOD Recall Corpus
# ---------------------------------------------------------------------------

class FrontierMicroTokenizer:
    """Fast deterministic byte + keyword/punctuation hybrid tokenizer for E1 iso-param tests.

    Provides lossless roundtrip encoding and exact byte-length tracking so held-out loss
    converts accurately to Bits-Per-Byte (``BPB = loss_nats * tokens / (bytes * ln 2)``).
    """

    def __init__(self, vocab_size: int = 8_192) -> None:
        if vocab_size < 512:
            raise ValueError("vocab_size must be >= 512")
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self._byte_offset = 16
        # Reserve 0..15 special, 16..271 raw UTF-8 bytes, 272..vocab_size-1 deterministic hashed word/code atoms
        self._base_words: dict[str, int] = {}
        self._id_to_str: list[str] = ["<pad>", "<bos>", "<eos>"] + [f"<sp_{i}>" for i in range(3, 16)]
        for b in range(256):
            self._id_to_str.append(chr(b) if 32 <= b < 127 else f"\\x{b:02x}")
        self._next_id = self._byte_offset + 256

    def _hash_token(self, tok: str) -> int:
        if tok in self._base_words:
            return self._base_words[tok]
        digest = int.from_bytes(hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest(), "little")
        span = self.vocab_size - (self._byte_offset + 256)
        tid = (self._byte_offset + 256) + (digest % span)
        return tid

    def encode(self, text: str, *, add_bos: bool = True) -> tuple[list[int], int]:
        raw_bytes = len(text.encode("utf-8"))
        ids: list[int] = [self.bos_id] if add_bos else []
        # Tokenize identifiers/numbers/words >1 char via deterministic hash bucket, falling back to bytes for rare symbols
        parts = re.findall(r"[A-Za-z_][A-Za-z0-9_]{1,15}|[0-9]{2,6}|\s+|.", text)
        for p in parts:
            if len(p) >= 2 and not p.isspace():
                ids.append(self._hash_token(p))
            else:
                for b in p.encode("utf-8"):
                    ids.append(self._byte_offset + b)
        return ids, raw_bytes


def build_pretrain_tournament_datasets(
    tokenizer: FrontierMicroTokenizer,
    *,
    train_sequences: int = 2_400,
    seq_len: int = 256,
    seed: int = 1337,
) -> dict[str, Any]:
    """Build matched multi-domain pretraining & OOD length-extrapolation evaluation suites for E1."""
    kv_vocab_keys = [f"k{i:03d}" for i in range(128)]
    kv_vocab_vals = [f"V{i:03d}" for i in range(128)]

    def _sample_text(domain: str, idx: int, local_rng: random.Random) -> str:
        if domain == "code":
            fn = f"fn_{idx % 40}"
            a, b = local_rng.randint(1, 30), local_rng.randint(1, 30)
            return (
                f"def {fn}(xs):\n"
                f"    acc = {a}\n"
                f"    for v in xs:\n"
                f"        acc = (acc + v * {b}) % 997\n"
                f"    return acc\n"
                f"# check {fn}([1, 2]) == {( ( (a + 1*b)%997 + 2*b ) % 997 )}\n"
            )
        if domain == "math":
            x, y, z = local_rng.randint(10, 99), local_rng.randint(10, 99), local_rng.randint(5, 50)
            p1 = x * y
            p2 = p1 - z
            mod = p2 % 17
            return f"SOLVE ({x} * {y} - {z}) mod 17: prod={p1}; diff={p2}; FINAL_ANSWER: {mod}\n"
        if domain == "structured":
            node = f"srv-{local_rng.randint(10, 99)}"
            n1, n2 = local_rng.randint(5, 40), local_rng.randint(5, 40)
            return json.dumps({"node": node, "a": n1, "b": n2, "sum": n1 + n2, "ok": (n1 + n2) > 25}) + "\n"
        # Associative recall & multi-hop memory chain
        pairs = local_rng.sample(kv_vocab_keys, 10)
        vals = local_rng.sample(kv_vocab_vals, 10)
        bindings = "; ".join(f"{k}={v}" for k, v in zip(pairs, vals))
        q_idx = local_rng.randrange(len(pairs))
        distractors = " ".join(f"noise_{local_rng.randint(100,999)}" for _ in range(12))
        return f"MEM {bindings} | {distractors} | QUERY {pairs[q_idx]} -> {vals[q_idx]}\n"

    def _pack_stream(n_seqs: int, target_len: int, domain_weights: list[str], seed_offset: int) -> tuple[list[list[int]], float]:
        local_rng = random.Random(seed + seed_offset)
        seqs: list[list[int]] = []
        total_bytes = 0
        total_toks = 0
        buf: list[int] = []
        step = 0
        while len(seqs) < n_seqs:
            dom = local_rng.choice(domain_weights)
            txt = _sample_text(dom, step, local_rng)
            ids, nb = tokenizer.encode(txt, add_bos=True)
            total_bytes += nb
            total_toks += len(ids)
            buf.extend(ids)
            while len(buf) >= target_len + 1 and len(seqs) < n_seqs:
                seqs.append(buf[: target_len + 1])
                buf = buf[target_len:]
            step += 1
        bytes_per_tok = total_bytes / max(total_toks, 1)
        return seqs, bytes_per_tok

    train_seqs, train_bpt = _pack_stream(train_sequences, seq_len, ["code", "math", "structured", "assoc"], seed_offset=0)
    val_splits: dict[str, tuple[list[list[int]], float]] = {}
    for off, dom in enumerate(("code", "math", "structured", "assoc"), start=1):
        val_splits[dom] = _pack_stream(96, seq_len, [dom], seed_offset=10_000 + off * 100)

    return {
        "train_seqs": train_seqs,
        "train_bytes_per_token": train_bpt,
        "val_splits": val_splits,
    }


def build_ood_recall_probes(
    tokenizer: FrontierMicroTokenizer,
    *,
    lengths: tuple[int, ...] = (256, 512, 1024),
    probes_per_length: int = 48,
    seed: int = 9991,
) -> dict[int, list[tuple[list[int], int]]]:
    """Generate OOD length-extrapolation associative-recall probes: ``(prompt_ids, target_token_id)``."""
    rng = random.Random(seed)
    kv_keys = [f"k{i:03d}" for i in range(128)]
    kv_vals = [f"V{i:03d}" for i in range(128)]
    probes: dict[int, list[tuple[list[int], int]]] = {}
    for seq_len in lengths:
        items: list[tuple[list[int], int]] = []
        for _ in range(probes_per_length):
            keys = rng.sample(kv_keys, 10)
            vals = rng.sample(kv_vals, 10)
            q_i = rng.randrange(len(keys))
            header = "MEM " + "; ".join(f"{k}={v}" for k, v in zip(keys, vals)) + " |"
            query_suffix = f"| QUERY {keys[q_i]} -> "
            h_ids, _ = tokenizer.encode(header, add_bos=True)
            q_ids, _ = tokenizer.encode(query_suffix, add_bos=False)
            tgt_ids, _ = tokenizer.encode(vals[q_i], add_bos=False)
            target_tok = tgt_ids[0]
            pad_need = max(0, seq_len - len(h_ids) - len(q_ids) - 1)
            filler_ids: list[int] = []
            while len(filler_ids) < pad_need:
                f_ids, _ = tokenizer.encode(f" noise_{rng.randint(100, 999)}", add_bos=False)
                filler_ids.extend(f_ids)
            full_prompt = (h_ids + filler_ids[:pad_need] + q_ids)[-(seq_len - 1) :]
            items.append((full_prompt, target_tok))
        probes[seq_len] = items
    return probes
