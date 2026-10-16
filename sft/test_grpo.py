import sys
import os

# Ensure the app directory is in path so we can import 'sft' package properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from unittest.mock import patch
import torch

original_load = torch.load
def mock_load(path, *args, **kwargs):
    if isinstance(path, str) and ('ckpt.pt' in path or 'sft.pt' in path):
        return {"emb.weight": torch.zeros((4096, 192))}
    return original_load(path, *args, **kwargs)

import tokenizers
original_from_file = tokenizers.Tokenizer.from_file

def mock_from_file(path, *args, **kwargs):
    if 'tokenizer.json' in path and not os.path.exists(path):
        # Find the correct path
        sft_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer.json')
        if os.path.exists(sft_path):
            return original_from_file(sft_path, *args, **kwargs)
        else:
            # Fallback mock if completely missing
            mock_tok = type('MockTokenizer', (), {'encode': lambda self, x, *a, **kw: type('MockRes', (), {'ids': []})(), 'token_to_id': lambda self, x: 1, 'decode': lambda self, x: ""})()
            return mock_tok
    return original_from_file(path, *args, **kwargs)

@pytest.fixture(autouse=True)
def mock_dependencies_fixture(monkeypatch):
    monkeypatch.setattr(torch, 'load', mock_load)
    monkeypatch.setattr(tokenizers.Tokenizer, 'from_file', mock_from_file)
    import sft.model_nano as model_nano
    monkeypatch.setattr(model_nano.GPT, 'load_state_dict', lambda *args, **kwargs: None)


# Apply patching temporarily just for import
old_torch_load = torch.load
torch.load = mock_load

old_from_file = tokenizers.Tokenizer.from_file
tokenizers.Tokenizer.from_file = mock_from_file

import model_nano
old_load_state_dict = model_nano.GPT.load_state_dict
model_nano.GPT.load_state_dict = lambda *args, **kwargs: None

import grpo

torch.load = old_torch_load
tokenizers.Tokenizer.from_file = old_from_file
model_nano.GPT.load_state_dict = old_load_state_dict

def test_verify_not_stopped():
    assert grpo.verify([1, 2, 3], False) == 0.0

@patch('model_nano.tok.decode')
def test_verify_too_short(mock_decode):
    mock_decode.return_value = ""
    assert grpo.verify([1], True) == 0.0
    mock_decode.assert_called_once_with([1])

@patch('model_nano.tok.decode')
def test_verify_just_right(mock_decode):
    mock_decode.return_value = "hello world"
    assert grpo.verify([1, 2], True) == 1.0
    mock_decode.assert_called_once_with([1, 2])

@patch('model_nano.tok.decode')
def test_verify_max_words(mock_decode):
    mock_decode.return_value = "one two three four five six seven eight"
    assert grpo.verify([1, 2, 3, 4, 5, 6, 7, 8], True) == 1.0
    mock_decode.assert_called_once_with([1, 2, 3, 4, 5, 6, 7, 8])

@patch('model_nano.tok.decode')
def test_verify_too_long(mock_decode):
    mock_decode.return_value = "one two three four five six seven eight nine"
    assert grpo.verify([1, 2, 3, 4, 5, 6, 7, 8, 9], True) == 0.0
    mock_decode.assert_called_once_with([1, 2, 3, 4, 5, 6, 7, 8, 9])
