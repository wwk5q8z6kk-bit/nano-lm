"""Static CUDA-Graph Stateful Decoder for Non-Abelian HSPA Language Models.

Captures the O(1)-state single-token step recurrence (``seq_len == 1``) of
non-transformer ``HSPAUnifiedLM`` architectures into a single replayable
``torch.cuda.CUDAGraph``, eliminating eager Python per-layer kernel launch
overhead during batch=1 autoregressive decoding.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor

from nanolm_holonomy.architecture import HSPAUnifiedLM


def _clone_states(states: list[Any]) -> list[Any]:
    out: list[Any] = []
    for st in states:
        if isinstance(st, tuple):
            out.append(tuple(t.clone() for t in st))
        else:
            out.append(st.clone())
    return out


def _copy_states_(dst: list[Any], src: list[Any]) -> None:
    for d_item, s_item in zip(dst, src):
        if isinstance(d_item, tuple):
            for d_t, s_t in zip(d_item, s_item):
                d_t.copy_(s_t)
        else:
            d_item.copy_(s_item)


class StaticGraphHSPADecoder:
    """Zero-allocation static-buffer CUDA Graph decoder for HSPA architectures."""

    def __init__(
        self,
        model: HSPAUnifiedLM,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
        use_autocast: bool = False,
        autocast_dtype: torch.dtype | None = None,
    ) -> None:
        if model.config.is_transformer:
            raise ValueError("StaticGraphHSPADecoder supports O(1)-state non-transformer HSPA families only")
        self.model = model
        self.model.eval()
        self.batch_size = int(batch_size)
        ref = next(model.parameters())
        self.device = torch.device(device) if device is not None else ref.device
        self.use_autocast = bool(use_autocast and self.device.type == "cuda")
        self.autocast_dtype = autocast_dtype or (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] >= 8
            else torch.float16
        )

        self.input_buf = torch.zeros((self.batch_size, 1), dtype=torch.long, device=self.device)
        with torch.no_grad(), self._ctx():
            lg0, st0 = self.model(
                self.input_buf,
                past_states=None,
                use_state=True,
                past_len=0,
                logits_to_keep=1,
            )
        self.state_bufs: list[Any] = _clone_states(st0)
        self.logits_buf: Tensor = lg0.clone()
        self._graph: Any = None

    def _ctx(self) -> Any:
        if self.use_autocast:
            return torch.amp.autocast(device_type="cuda", dtype=self.autocast_dtype)
        return nullcontext()

    def _single_step_into_buffers(self) -> None:
        with torch.no_grad(), self._ctx():
            lg, nxt = self.model(
                self.input_buf,
                past_states=self.state_bufs,
                use_state=True,
                past_len=0,
                logits_to_keep=1,
            )
        _copy_states_(self.state_bufs, nxt)
        self.logits_buf.copy_(lg)

    def capture_graph(self, *, warmup_steps: int = 3) -> bool:
        """Warmup and capture the static single-token step into a CUDA Graph."""
        if self.device.type != "cuda":
            self._graph = None
            return False

        saved_states = _clone_states(self.state_bufs)
        saved_logits = self.logits_buf.clone()
        try:
            capture_stream = torch.cuda.Stream(device=self.device)
            capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(capture_stream):
                for _ in range(max(1, warmup_steps)):
                    self._single_step_into_buffers()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    self._single_step_into_buffers()
            torch.cuda.current_stream(self.device).wait_stream(capture_stream)
            self._graph = graph
        finally:
            _copy_states_(self.state_bufs, saved_states)
            self.logits_buf.copy_(saved_logits)
        return True

    @torch.no_grad()
    def prefill(self, prompt_ids: Tensor, *, chunk_size: int = 512) -> Tensor:
        """Chunk-prefill ``prompt_ids`` into static state buffers and return first predicted token."""
        if prompt_ids.shape[0] != self.batch_size:
            raise ValueError(f"Expected batch_size={self.batch_size}, got {prompt_ids.shape[0]}")
        seq_len = int(prompt_ids.shape[1])
        st: list[Any] | None = None
        lg: Tensor | None = None
        with self._ctx():
            for cs in range(0, seq_len, chunk_size):
                ce = min(seq_len, cs + chunk_size)
                lg, st = self.model(
                    prompt_ids[:, cs:ce],
                    past_states=st,
                    use_state=True,
                    past_len=cs,
                    logits_to_keep=1,
                )
        assert st is not None and lg is not None
        _copy_states_(self.state_bufs, st)
        self.logits_buf.copy_(lg)
        return self.logits_buf[:, -1:, :].argmax(dim=-1).clone()

    @torch.no_grad()
    def step(self, token_ids: Tensor) -> Tensor:
        """Execute one O(1)-state decode step (replaying CUDA Graph when captured)."""
        if token_ids.shape != self.input_buf.shape:
            raise ValueError(f"Expected token_ids shape {tuple(self.input_buf.shape)}, got {tuple(token_ids.shape)}")
        self.input_buf.copy_(token_ids)
        if self._graph is not None:
            self._graph.replay()
        else:
            self._single_step_into_buffers()
        return self.logits_buf

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt_ids: Tensor,
        *,
        max_new_tokens: int,
        chunk_size: int = 512,
    ) -> Tensor:
        """Autoregressively decode ``max_new_tokens`` matching sequential greedy execution."""
        if max_new_tokens <= 0:
            return prompt_ids[:, :0]
        cur = self.prefill(prompt_ids, chunk_size=chunk_size)
        generated = [cur]
        for _ in range(max_new_tokens - 1):
            lg = self.step(cur)
            cur = lg[:, -1:, :].argmax(dim=-1).clone()
            generated.append(cur)
        return torch.cat(generated, dim=1)
