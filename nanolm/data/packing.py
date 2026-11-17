"""Document-packed sequence stream with intra-document causal masking and position resets."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class PackedBatch:
    input_ids: Tensor     # [B, T]
    targets: Tensor       # [B, T] (-100 at cross-document transitions!)
    position_ids: Tensor  # [B, T] (resets to 0 at start of every packed document!)
    doc_ids: Tensor       # [B, T] (integer document segment ID for block-diagonal attention)


class PackedDocumentDataset:
    """Packs variable-length tokenized documents into fixed-length blocks with zero cross-doc leakage."""

    def __init__(
        self,
        documents: list[list[int]],
        seq_len: int,
        bos_id: int = 0,
        eos_id: int = 1,
    ) -> None:
        if seq_len < 4:
            raise ValueError("seq_len must be >= 4")
        self.seq_len = seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id

        flat_tokens: list[int] = []
        flat_pos: list[int] = []
        flat_doc: list[int] = []

        doc_counter = 0
        for raw_doc in documents:
            if not raw_doc:
                continue
            wrapped = [bos_id, *raw_doc, eos_id]
            flat_tokens.extend(wrapped)
            flat_pos.extend(range(len(wrapped)))
            flat_doc.extend([doc_counter] * len(wrapped))
            doc_counter += 1

        total_len = len(flat_tokens)
        usable_blocks = (total_len - 1) // seq_len
        if usable_blocks < 1:
            raise ValueError(f"Not enough tokens ({total_len}) to form one block of seq_len={seq_len}")

        keep = usable_blocks * seq_len + 1
        tok_arr = np.asarray(flat_tokens[:keep], dtype=np.int32)
        pos_arr = np.asarray(flat_pos[:keep], dtype=np.int32)
        doc_arr = np.asarray(flat_doc[:keep], dtype=np.int32)

        self.inputs = np.empty((usable_blocks, seq_len), dtype=np.int32)
        self.targets = np.empty((usable_blocks, seq_len), dtype=np.int32)
        self.positions = np.empty((usable_blocks, seq_len), dtype=np.int32)
        self.docs = np.empty((usable_blocks, seq_len), dtype=np.int32)

        for b in range(usable_blocks):
            lo = b * seq_len
            hi = lo + seq_len
            self.inputs[b] = tok_arr[lo:hi]
            tgt = tok_arr[lo + 1 : hi + 1].copy()
            # Never train to predict the next document's <bos> from the previous document's <eos>!
            cross_doc = doc_arr[lo:hi] != doc_arr[lo + 1 : hi + 1]
            tgt[cross_doc] = -100
            self.targets[b] = tgt
            self.positions[b] = pos_arr[lo:hi]
            self.docs[b] = doc_arr[lo:hi]

        self.num_blocks = usable_blocks
        self.total_tokens = usable_blocks * seq_len

    def sample_batch(
        self,
        batch_size: int,
        step: int,
        seed: int = 42,
        device: torch.device | str = "cpu",
    ) -> PackedBatch:
        """Deterministic, step-indexed sampling (resumable at any step without replay!)."""
        rng = np.random.Generator(np.random.PCG64(seed + step * 1_000_003))
        indices = rng.integers(0, self.num_blocks, size=batch_size)
        return PackedBatch(
            input_ids=torch.from_numpy(self.inputs[indices].astype(np.int64)).to(device),
            targets=torch.from_numpy(self.targets[indices].astype(np.int64)).to(device),
            position_ids=torch.from_numpy(self.positions[indices].astype(np.int64)).to(device),
            doc_ids=torch.from_numpy(self.docs[indices].astype(np.int64)).to(device),
        )
