"""Fast instrument checks, independent of the multi-gigabyte model inputs."""

import mmap
from pathlib import Path
import tempfile

import numpy as np
import pytest

pytest.importorskip("tokenizers")
from corpus import extract
from evaluate import EOS, aligned_match, blocks, shuffled_tokens


def test_context_resets_preserve_every_target():
    ids = list(range(3, 518))
    windows = list(blocks(ids))
    assert [len(target) for _, target in windows] == [256, 256, 3]
    assert [token for _, target in windows for token in target] == ids
    assert [inputs[0] for inputs, _ in windows] == [EOS, ids[255], ids[511]]
    assert all(inputs[1:] == target[:-1] for inputs, target in windows)


def test_shuffle_changes_order_with_identical_block_multisets():
    ids = list(range(777))
    one = shuffled_tokens(ids, "doc-1")
    assert one == shuffled_tokens(ids, "doc-1")
    assert one != shuffled_tokens(ids, "doc-2")
    for i in range(0, len(ids), 256):
        assert sorted(one[i:i + 256]) == ids[i:i + 256]


def test_overlap_requires_token_alignment():
    ids = [17, 4000, 511]
    needle = np.asarray(ids, dtype="<u2").tobytes()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokens.bin"
        path.write_bytes(b"\x80" + needle + b"\x80")
        with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            assert not aligned_match(data, ids)
        path.write_bytes(b"\x80" + needle + b"\x80" + needle)
        with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            assert aligned_match(data, ids)


def test_publication_boundaries():
    assert extract("fed", "<strong>Staff Review of the Economic Situation</strong><br>Output rose."
                   "<strong>Financial Situation</strong>Excluded.") == "Output rose."
    pre = "<pre>Preface\nResolved by the Senate &lt;&lt;NOTE: date.&gt;&gt; to approve. Approved, April</pre>"
    assert extract("law", pre) == "Resolved by the Senate to approve."
    pre = "<pre>1200 PM AST Mon Oct 6 2025\nThe storm weakened.\nFORECAST POSITIONS AND MAX WINDS\nTable</pre>"
    assert extract("weather", pre) == "The storm weakened."
    with pytest.raises(ValueError):
        extract("fed", "<p>Missing section</p>")


def test_loss_counts_each_target_at_correct_shift():
    import math
    torch = pytest.importorskip("torch")
    from evaluate import loss_bits

    class TransitionModel:
        def __call__(self, x):
            # Token (previous + 1) mod 4 has probability 3/6, all others 1/6.
            logits = torch.zeros((*x.shape, 4))
            return logits.scatter_(-1, ((x + 1) % 4).unsqueeze(-1), math.log(3))

    target = [2, 3, 0, 1] * 70
    assert loss_bits(TransitionModel(), target) == pytest.approx(len(target), abs=1e-4)
    wrong = [0] * 260
    assert loss_bits(TransitionModel(), wrong) == pytest.approx(len(wrong) * math.log2(6), abs=1e-4)


def test_common_byte_chunks_preserve_whitespace_and_multibyte_text():
    from reference.compare import split_bytes
    for text in ("A group of words around a new boundary.", "三四五六七八九", "一丁 一丁 " * 5,
                 "Ends with a space. ", " alpha"):
        original = text.encode()
        chunks = list(split_bytes(original, limit=7))
        assert "".join(c[2] for c in chunks) == text
        assert all(0 < end - start <= 7 and original[start:end] == s.encode() for start, end, s in chunks)
        assert [a[1] for a in chunks[:-1]] == [b[0] for b in chunks[1:]]


def test_reference_batch_scores_leading_targets_and_ignores_padding():
    import math
    torch = pytest.importorskip("torch")
    from reference.compare import measure

    def predict(x):
        logits = torch.zeros((*x.shape, 4))
        return logits.scatter_(-1, ((x + 1) % 4).unsqueeze(-1), math.log(3))

    sample = [({"doc": "b", "start": 12}, [2, 3, 0], [1, 2, 3, 3]),
              ({"doc": "a", "start": 0}, [2], [3])]
    native = measure(sample, "M0", predict)
    public = measure(sample, "Pythia-step256", predict)
    assert [r["doc"] for r in public] == ["a", "b"]
    assert [r["bits"] for r in native] == pytest.approx([1, 3], abs=1e-6)
    assert [r["bits"] for r in public] == pytest.approx([math.log2(6), 3 + math.log2(6)], abs=1e-6)
