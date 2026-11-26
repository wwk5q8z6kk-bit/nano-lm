"""Cached decoding must make the same choice as Transformers greedy generation."""
import pytest


def test_equal_logits_follow_vocabulary_order_at_every_trie_branch(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from nanoscribe.compact import LocalCompactPredictor

    class Tokenizer:
        eos_token_id = 1

        def encode(self, text, add_special_tokens=False):
            return [ord(ch) for ch in text]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(int(token)) for token in ids if int(token) != self.eos_token_id)

        def apply_chat_template(self, messages, **kwargs):
            return "T"

        def __call__(self, text, return_tensors):
            return transformers.BatchEncoding({"input_ids": torch.tensor([self.encode(text)])})

    config = transformers.GPT2Config(vocab_size=128, n_layer=1, n_head=1, n_embd=8, n_positions=16,
                                     eos_token_id=1, pad_token_id=1, bos_token_id=0)
    model = transformers.GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    messages = [{"role": "user", "content": "Tie fixture"}]
    tokenizer = Tokenizer()
    for choices, expected in [(["U", "S"], "S"), (["S", "U"], "S"),
                              (["US", "UT", "ST", "SS"], "SS"), (["ST", "S"], "S")]:
        for cache in (False, True):
            predictor = LocalCompactPredictor(str(tmp_path), decoding="constrained", prefix_cache=cache)
            predictor._loaded = (tokenizer, model, "cpu")
            chosen, _ = predictor._choose(messages, choices)
            assert chosen == expected, (choices, cache)
