import pytest

from sft import grpo


def test_verify_not_stopped():
    assert grpo.verify([1, 2, 3], False) == 0.0


@pytest.mark.parametrize(("text", "reward"), [
    ("", 0.0),
    ("hello world", 1.0),
    ("one two three four five six seven eight", 1.0),
    ("one two three four five six seven eight nine", 0.0),
])
def test_verify_word_count(text, reward):
    ids = grpo.M.tok.encode(text, add_special_tokens=False).ids
    assert grpo.verify(ids, True) == reward
