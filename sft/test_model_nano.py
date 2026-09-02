import os
import sys
import pytest
import torch
from unittest.mock import patch, MagicMock

# Since model_nano.py relies on a local tokenizer.json which may not exist or be located
# in a different directory when running pytest, we mock the entire tokenizers module
mock_tokenizer_module = MagicMock()
mock_tokenizer_class = MagicMock()
mock_tokenizer_instance = MagicMock()
mock_tokenizer_instance.token_to_id.return_value = 1
mock_tokenizer_class.from_file.return_value = mock_tokenizer_instance
mock_tokenizer_module.Tokenizer = mock_tokenizer_class
sys.modules['tokenizers'] = mock_tokenizer_module

# Add the sft directory to path so that sft modules can import correctly if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_nano import GPT, load

def test_load_checkpoint(tmp_path):
    model = GPT()
    model.emb.weight.data.fill_(0.5)

    ckpt_path = tmp_path / "dummy.pt"
    torch.save(model.state_dict(), ckpt_path)

    with patch("model_nano.dev", "cpu"):
        loaded_model = load(str(ckpt_path))

    assert isinstance(loaded_model, GPT)
    assert not loaded_model.training
    assert torch.equal(loaded_model.emb.weight.data, model.emb.weight.data)
    assert next(loaded_model.parameters()).device.type == "cpu"

def test_load_checkpoint_missing_file():
    with patch("model_nano.dev", "cpu"):
        with pytest.raises(FileNotFoundError):
            load("non_existent_file.pt")

def test_load_checkpoint_invalid_file(tmp_path):
    invalid_path = tmp_path / "invalid.pt"
    invalid_path.write_text("not a pt file")
    with patch("model_nano.dev", "cpu"):
        with pytest.raises(Exception):
            load(str(invalid_path))
