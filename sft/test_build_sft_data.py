import sys
import os

# Add the parent directory to sys.path so we can import build_sft_data without "sft." prefix
# This avoids importing "sft" module which seems to run side-effects or load missing files
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import build_sft_data

class DummyEncoding:
    def __init__(self, ids):
        self.ids = ids

class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        # simple mock: just return list of ordinals for each character
        return DummyEncoding([ord(c) for c in text])

@pytest.fixture(autouse=True)
def setup_globals(monkeypatch):
    monkeypatch.setattr(build_sft_data, "tok", DummyTokenizer())
    monkeypatch.setattr(build_sft_data, "ims_id", 999)
    monkeypatch.setattr(build_sft_data, "ime_id", 1000)

def test_render_happy_path():
    convo = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"}
    ]
    ids, mask = build_sft_data.render(convo)

    # User message
    user_head = [ord(c) for c in "user\n"]
    user_body = [ord(c) for c in "hello"]
    user_seg = [999] + user_head + user_body + [1000]

    # Assistant message
    asst_head = [ord(c) for c in "assistant\n"]
    asst_body = [ord(c) for c in "world"]
    asst_seg = [999] + asst_head + asst_body + [1000]

    expected_ids = user_seg + asst_seg
    assert ids == expected_ids

    # user mask is all 0
    expected_mask = [0] * len(user_seg)

    # assistant mask: 0 for 999 and head, 1 for body and 1000
    asst_mask = [0] + [0] * len(asst_head) + [1] * len(asst_body) + [1]
    expected_mask += asst_mask

    assert mask == expected_mask

def test_render_multi_turn():
    convo = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"}
    ]
    ids, mask = build_sft_data.render(convo)

    # The mask should have 1s only for the two assistant bodies and their <|im_end|>s.
    assert sum(mask) == len("a1") + 1 + len("a2") + 1

def test_render_no_assistant():
    convo = [
        {"role": "user", "content": "hello"},
        {"role": "user", "content": "world"}
    ]
    ids, mask = build_sft_data.render(convo)

    # Entire mask should be 0s
    assert sum(mask) == 0
