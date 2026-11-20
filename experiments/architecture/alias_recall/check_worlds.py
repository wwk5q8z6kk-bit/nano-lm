"""Independent deterministic controls for the paired-world instrument."""
import torch

from baselines import ALIAS_OFFSET, BOS, SEP, SEEN_VALS, V0
from study import N_QUERIES, N_RECORDS, N_WORLDS, paired_worlds


def main():
    unique, exposed, answers = paired_worlds(N_WORLDS)
    sep_pos = 1 + 3 * N_RECORDS
    assert torch.equal(unique[:, :sep_pos + 1], exposed[:, :sep_pos + 1])
    assert torch.equal(unique[:, -1], exposed[:, -1])
    for clean, repeat, answer in zip(unique.tolist(), exposed.tolist(), answers.tolist()):
        assert clean[0] == BOS and clean[sep_pos] == SEP
        records = {clean[i]: clean[i + 1:i + 3] for i in range(1, sep_pos, 3)}
        assert len(records) == N_RECORDS
        key = clean[-1]
        assert records[key - ALIAS_OFFSET] == answer and min(answer) >= V0 + SEEN_VALS
        prefix_a, prefix_b = [], []
        for row, history in ((clean, prefix_a), (repeat, prefix_b)):
            for i in range(sep_pos + 1, len(row) - 1, 3):
                triple = row[i:i + 3]
                assert records[triple[0] - ALIAS_OFFSET] == triple[1:]
                history.append(triple)
            assert len(history) == N_QUERIES - 1
            assert len({triple[0] for triple in history}) == len(history)
        assert sum(a != b for a, b in zip(prefix_a, prefix_b)) == 1
        assert key not in {triple[0] for triple in prefix_a}
        assert [key, *answer] in prefix_b
    print(f"{len(answers)} paired worlds verified: prefix lookup 0/{len(answers)} vs {len(answers)}/{len(answers)}, source lookup perfect in both")


if __name__ == "__main__":
    main()
