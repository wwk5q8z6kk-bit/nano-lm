"""Compare the published subset against an author-supplied original unified-loop model module."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

from run import ARMS, UnifiedBlockLM, digest


def compare(source_path):
    spec = importlib.util.spec_from_file_location("unified_source_for_check", source_path)
    source = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = source
    spec.loader.exec_module(source)
    results = {}
    torch.set_num_threads(1)
    for arm, (mechanism, registers, ffn) in ARMS.items():
        kwargs = dict(n_heads=4, hspa_registers=registers, ffn_hidden_override=ffn)
        models = []
        for cls in (source.UnifiedBlockLM, UnifiedBlockLM):
            torch.manual_seed(1000)
            models.append(cls(256, 128, mechanism, **kwargs).eval())
        original, public = models
        states = [model.state_dict() for model in models]
        assert states[0].keys() == states[1].keys()
        assert all(torch.equal(value, states[1][key]) for key, value in states[0].items())
        rng = torch.Generator().manual_seed(29)
        tokens = torch.randint(0, 256, (2, 49), generator=rng)
        outputs = [model(tokens[:, :-1], return_base_logits=True) for model in models]
        for full, reduced in zip(*outputs, strict=True):
            torch.testing.assert_close(full, reduced, atol=0, rtol=0)
        for mixed, base in outputs:
            loss = -mixed.gather(-1, tokens[:, 1:, None]).mean()
            loss += 0.25 * torch.nn.functional.cross_entropy(base.reshape(-1, 256), tokens[:, 1:].reshape(-1))
            loss.backward()
        for ((name, first), (other, second)) in zip(original.named_parameters(), public.named_parameters(), strict=True):
            assert name == other
            if first.grad is None:
                assert second.grad is None
            else:
                torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0)
        alternative = tokens[:, :-1].clone()
        alternative[:, 24:] = torch.randint(0, 256, (2, 24), generator=rng)
        alternative[1] = torch.randint(0, 256, (48,), generator=rng)
        with torch.no_grad():
            alternate_output = public(alternative)
            delta = (alternate_output[0, :24] - outputs[1][0][0, :24]).abs().max().item()
            assert delta < 1e-5, (arm, delta)
        results[arm] = {"same_initial_tensors": len(states[0]), "same_mixed_and_base_output": True,
                        "same_gradients": True, "unrelated_suffix_and_batch_row_max_abs_delta": delta}
    return {"source_sha256": digest(source_path), "public_sha256": digest(Path(__file__).with_name("model.py")), "arms": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    print(json.dumps(compare(ap.parse_args().source), indent=2))
