"""Offline integration checks with tiny synthetic checkpoints, never research data."""
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from nanoscribe import qwen_inference


@pytest.fixture(autouse=True)
def cpu_loader(monkeypatch, tmp_path):
    from transformers import dynamic_module_utils

    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(qwen_inference, "_CACHE", {})


def save_tokenizer(path):
    from tokenizers import Tokenizer, models

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.save_pretrained(path)


@pytest.mark.parametrize("surface", ["tokenizer", "model"])
def test_custom_repository_python_is_not_executed(tmp_path, surface):
    marker = tmp_path / "executed"
    (tmp_path / "custom.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('synthetic repository code executed')\n"
    )
    if surface == "tokenizer":
        (tmp_path / "tokenizer_config.json").write_text(json.dumps({
            "tokenizer_class": "SyntheticTokenizer",
            "auto_map": {"AutoTokenizer": ["custom.SyntheticTokenizer", None]},
        }))
        legacy_loader = transformers.AutoTokenizer
    else:
        save_tokenizer(tmp_path)
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "synthetic_untrusted_model",
            "auto_map": {"AutoConfig": "custom.SyntheticConfig",
                         "AutoModelForCausalLM": "custom.SyntheticModel"},
        }))
        legacy_loader = transformers.AutoModelForCausalLM

    with pytest.raises(ValueError):
        qwen_inference._load_qwen(str(tmp_path))
    assert not marker.exists()
    assert str(tmp_path) not in qwen_inference._CACHE
    # Positive control proves this fixture actually reaches executable code
    # under the former policy, rather than merely being an invalid checkpoint.
    with pytest.raises(RuntimeError, match="synthetic repository code executed"):
        legacy_loader.from_pretrained(str(tmp_path), trust_remote_code=True, local_files_only=True)
    assert marker.exists()


def test_builtin_qwen_cpu_logits_tokens_and_cache_are_unchanged(tmp_path):
    torch.manual_seed(123)
    config = transformers.Qwen2Config(vocab_size=16, hidden_size=16,
        intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, max_position_embeddings=32,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    transformers.Qwen2ForCausalLM(config).save_pretrained(tmp_path)
    save_tokenizer(tmp_path)
    legacy = transformers.AutoModelForCausalLM.from_pretrained(
        tmp_path, torch_dtype=torch.float32, trust_remote_code=True).eval()
    loaded = qwen_inference._load_qwen(str(tmp_path))
    inputs = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        assert torch.equal(legacy(inputs).logits, loaded.model(inputs).logits)
        assert torch.equal(legacy.generate(inputs, max_new_tokens=3, do_sample=False),
                           loaded.model.generate(inputs, max_new_tokens=3, do_sample=False))
    assert loaded.tokenizer("hello")["input_ids"] == [1]
    assert loaded.device == "cpu"
    assert qwen_inference._load_qwen(str(tmp_path)) is loaded
