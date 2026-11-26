import torch

from .model import X9Config, X9LM
from .routing import cap_prefix, stock_prefix_end


def test_prefix_budget_matches_greedy_for_all_binary_sequences_through_fourteen():
    for length in range(1, 15):
        offered = ((torch.arange(2**length)[:, None] >> torch.arange(length)) % 2).bool()
        used = torch.zeros(len(offered), dtype=torch.long)
        expected = []
        for i in range(length):
            accepted = offered[:, i] & (used < max(1, (i + 1) * 3 // 10))
            expected.append(accepted)
            used += accepted
        assert torch.equal(cap_prefix(offered), torch.stack(expected, dim=1))


def test_serving_reference_replays_one_row_at_a_time_with_ties_and_shields():
    gen = torch.Generator().manual_seed(515007)
    scores = torch.randint(-5, 15, (16, 31), generator=gen)
    shield = torch.rand(16, 31, generator=gen) < 0.12
    eligible = (scores > 0) & ~shield & (torch.rand(16, 31, generator=gen) < 0.65)
    actual = stock_prefix_end(scores, eligible, shield)
    for row in range(len(scores)):
        for n in range(1, scores.shape[1] + 1):
            budget = max(1, int(n * .30))
            selected = eligible[row, n - 1]
            if eligible[row, :n].sum() > budget:
                bound = torch.topk(scores[row, :n], k=budget).values[-1]
                selected = (scores[row, n - 1] >= bound) & ~shield[row, n - 1]
            assert actual[row, n - 1] == selected


def test_model_keeps_logits_for_same_prefix_when_suffix_and_companion_change():
    torch.manual_seed(742)
    config = X9Config(vocab=64, d=32, n_head=4, n_kv=2, layout="DA", chunk=8, ffn=64, voc_lambda=-1)
    model = X9LM(config).eval()
    with torch.no_grad():
        model.refiner.ema_u_tau.fill_(0)
        model.refiner.o.weight.normal_(std=.06)
        model.refiner.dn.weight.normal_(std=.06)
        original = torch.randint(64, (2, 31))
        changed = torch.randint(64, original.shape)
        changed[0, :15] = original[0, :15]
        torch.testing.assert_close(model(changed)[0, :15], model(original)[0, :15], atol=1e-5, rtol=0)


def test_train_uses_previous_threshold_and_x8_retains_original_control():
    for mode in ("acex9", "x8"):
        config = X9Config(vocab=64, d=16, n_head=2, n_kv=1, layout="A", chunk=8, ffn=32, routing_mode=mode)
        model = X9LM(config).train()
        hidden = torch.ones(1, 4, 16)
        cache = torch.ones(1, 2, 4, 8)
        rope = torch.ones(4, 4), torch.zeros(4, 4)
        margin = torch.tensor([[.81, .86, .9, .99]])
        _, aux = model.refiner(hidden, hidden, cache, cache, rope, margin, margin, margin)
        assert model.refiner.ema_inited.item()
        threshold = .85 if mode == "acex9" else model.refiner.ema_u_tau.item()
        assert torch.equal(aux["hard_mask"], margin > threshold)
