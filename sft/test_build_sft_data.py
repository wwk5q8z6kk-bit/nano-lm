import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_data import ok

def test_ok_happy_path():
    msgs = [
        {"role": "user", "content": "Hello, how are you today?"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]
    assert ok(msgs) is True

def test_ok_max_messages():
    msgs = [
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "12345"},
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "12345"},
        {"role": "user", "content": "12345"},
        {"role": "assistant", "content": "12345"}
    ]
    assert ok(msgs) is True

def test_ok_too_few_messages():
    msgs = [
        {"role": "user", "content": "Hello, how are you today?"}
    ]
    assert ok(msgs) is False

def test_ok_too_many_messages():
    msgs = [{"role": "user", "content": "12345"} for _ in range(7)]
    assert ok(msgs) is False

def test_ok_wrong_starting_role():
    msgs = [
        {"role": "assistant", "content": "How can I help you?"},
        {"role": "user", "content": "I need some help."}
    ]
    assert ok(msgs) is False

def test_ok_content_too_short():
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]
    assert ok(msgs) is False

def test_ok_content_too_long():
    long_content = "a" * 901
    msgs = [
        {"role": "user", "content": long_content},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]
    assert ok(msgs) is False

def test_ok_high_symbol_ratio():
    # Length is 10, so 1 symbol is 0.1 ratio > 0.06
    msgs = [
        {"role": "user", "content": "123456789\\"},
        {"role": "assistant", "content": "I am fine, thank you!"}
    ]
    assert ok(msgs) is False

def test_ok_empty_msgs():
    assert ok([]) is False
