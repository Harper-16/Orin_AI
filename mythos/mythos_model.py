#!/usr/bin/env python3
"""Mythos: a recurrent-depth Transformer built on top of the scratch model.

This is a laptop-sized reconstruction of the "Mythos" hypothesis (a
Recurrent-Depth Transformer, or RDT). It keeps the same vocabulary,
tokenizer and chat plumbing as scratch/scratch_model.py, but replaces the plain
stack of blocks with three stages:

    Prelude           a few standard blocks, run once
    Recurrent Block   ONE weight-shared block, looped up to max_loop_iters
    Coda              a few standard blocks, run once

At every loop step the hidden state is refreshed with an LTI-stable
injection of the encoded input ``e``:

    h <- A * h + B * e + Block(h, e)

``A`` is constrained to a spectral radius below 1 so deep looping cannot
blow up, ``B`` re-injects the encoded input so the state cannot drift into
noise, a loop-index (depth) embedding marks which iteration is running, and
an optional Depth-Wise LoRA lets each iteration do slightly different work.
Optional adaptive computation time (ACT) halting lets easy positions stop
looping early.

Because the loop count is a runtime choice, one checkpoint can "think
longer" about harder questions instead of being trained deeper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scratch.scratch_model import ModelConfig, RMSNorm, SwiGLU, TransformerBlock, apply_rope


@dataclass
class MythosConfig(ModelConfig):
    """Configuration for the recurrent-depth Transformer.

    It extends :class:`scratch_model.ModelConfig`, so every field the shared
    blocks and the tokenizer already rely on (dim, heads, kv_heads,
    hidden_dim, rope_theta, dropout, vocab_size, max_seq_len) is reused.
    """

    prelude_layers: int = 2
    coda_layers: int = 2
    max_loop_iters: int = 6
    depth_lora_rank: int = 8
    use_halting: bool = True
    halting_threshold: float = 0.95
    ponder_weight: float = 0.01

    @classmethod
    def from_dict(cls, values: dict) -> "MythosConfig":
        accepted = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in accepted})

    def to_dict(self) -> dict:
        return asdict(self)


class DepthLinear(nn.Module):
    """A shared ``nn.Linear`` plus a small rank-r LoRA selected by loop index.

    The base projection is weight-shared across every loop iteration, but the
    low-rank correction is stored per depth. This is the "Depth-Wise LoRA"
    that keeps iteration 3 from being identical to iteration 12.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, max_depth: int):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.max_depth = max(1, max_depth)
        if rank > 0:
            self.lora_a = nn.Parameter(torch.empty(self.max_depth, rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(self.max_depth, out_features, rank))
            nn.init.normal_(self.lora_a, mean=0.0, std=0.02)
            self.scale = 1.0 / rank
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)
            self.scale = 0.0

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        out = self.base(x)
        if self.lora_a is not None:
            index = min(depth, self.max_depth - 1)
            a = self.lora_a[index]
            b = self.lora_b[index]
            out = out + (x @ a.t() @ b.t()) * self.scale
        return out


class RecurrentAttention(nn.Module):
    """Grouped-query causal attention whose projections are depth-aware."""

    def __init__(self, config: MythosConfig):
        super().__init__()
        rank, depth = config.depth_lora_rank, config.max_loop_iters
        self.head_dim = config.dim // config.heads
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.q_proj = DepthLinear(config.dim, config.heads * self.head_dim, rank, depth)
        self.k_proj = DepthLinear(config.dim, config.kv_heads * self.head_dim, rank, depth)
        self.v_proj = DepthLinear(config.dim, config.kv_heads * self.head_dim, rank, depth)
        self.o_proj = DepthLinear(config.dim, config.dim, rank, depth)
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x, depth).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x, depth).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x, depth).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_theta)
        if self.kv_heads != self.heads:
            repeat = self.heads // self.kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, time, -1)
        return self.o_proj(attended, depth)


class RecurrentFFN(nn.Module):
    """SwiGLU feed-forward whose projections are depth-aware."""

    def __init__(self, config: MythosConfig):
        super().__init__()
        rank, depth = config.depth_lora_rank, config.max_loop_iters
        self.gate_proj = DepthLinear(config.dim, config.hidden_dim, rank, depth)
        self.up_proj = DepthLinear(config.dim, config.hidden_dim, rank, depth)
        self.down_proj = DepthLinear(config.hidden_dim, config.dim, rank, depth)

    def forward(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(x, depth)) * self.up_proj(x, depth), depth
        )


class RecurrentBlock(nn.Module):
    """The single weight-shared block that is looped at every depth.

    It returns the *delta* to add to the recurrent state (rather than a full
    residual), so the caller can combine it with the LTI injection
    ``A*h + B*e``. It also owns the loop-index embedding that marks which
    iteration is running.
    """

    def __init__(self, config: MythosConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = RecurrentAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = RecurrentFFN(config)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.loop_embedding = nn.Parameter(torch.zeros(config.max_loop_iters, config.dim))

    def forward(self, h: torch.Tensor, depth: int) -> torch.Tensor:
        index = min(depth, self.loop_embedding.shape[0] - 1)
        x = h + self.loop_embedding[index]
        x = x + self.residual_dropout(self.attention(self.attention_norm(x), depth))
        x = x + self.residual_dropout(self.feed_forward(self.ffn_norm(x), depth))
        # Return only the update; the caller owns the recurrent residual.
        return x - h



class MythosTransformer(nn.Module):
    """Recurrent-depth Transformer: Prelude + looped Recurrent Block + Coda."""

    def __init__(self, config: MythosConfig):
        super().__init__()
        if config.dim % config.heads or config.heads % config.kv_heads:
            raise ValueError("dim must divide heads, and heads must divide evenly across kv_heads.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if config.max_loop_iters < 1:
            raise ValueError("max_loop_iters must be at least 1.")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.prelude = nn.ModuleList(TransformerBlock(config) for _ in range(config.prelude_layers))
        self.recurrent = RecurrentBlock(config)
        self.coda = nn.ModuleList(TransformerBlock(config) for _ in range(config.coda_layers))
        self.final_norm = RMSNorm(config.dim)
        self.halt_head = nn.Linear(config.dim, 1) if config.use_halting else None
        # LTI injection: h <- A*h + B*e. A is a diagonal matrix kept below
        # spectral radius 1 via a sigmoid, so deep loops stay stable.
        self.inject_a = nn.Parameter(torch.zeros(config.dim))
        self.inject_b = nn.Parameter(torch.zeros(config.dim))
        self.apply(self._init_weights)
        if self.halt_head is not None:
            nn.init.constant_(self.halt_head.bias, -1.0)
        # Shared input/output embedding keeps the laptop-sized model compact.
        self.lm_head = None

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def spectral_radius(self) -> float:
        """Largest |eigenvalue| of the diagonal injection matrix A (must be < 1)."""
        return float((torch.sigmoid(self.inject_a) * 0.99).abs().max().item())


    def _recur(self, e: torch.Tensor, n_loops: int):
        """Loop the shared block ``n_loops`` times over the encoded input ``e``."""
        batch, time, _ = e.shape
        device = e.device
        h = e
        use_halting = self.halt_head is not None
        if use_halting:
            halted = torch.zeros(batch, time, dtype=torch.bool, device=device)
            cumulative = torch.zeros(batch, time, device=device)
            step_count = torch.zeros(batch, time, device=device)
        a = torch.sigmoid(self.inject_a) * 0.99
        b = torch.sigmoid(self.inject_b)
        for depth in range(n_loops):
            delta = self.recurrent(h, depth)
            h_new = a * h + b * e + delta
            if use_halting:
                p_halt = torch.sigmoid(self.halt_head(h_new)).squeeze(-1)
                active = (~halted).to(h_new.dtype)
                step_count = step_count + active
                h = torch.where(halted.unsqueeze(-1), h, h_new)
                cumulative = cumulative + active * p_halt
                halted = halted | (cumulative >= self.config.halting_threshold)
            else:
                h = h_new
        ponder = torch.zeros((), device=device)
        if use_halting:
            # Expected number of loop steps taken, as a small compute penalty.
            ponder = step_count.mean() * self.config.ponder_weight
        return h, ponder

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, n_loops: int | None = None):
        if input_ids.shape[-1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence has {input_ids.shape[-1]} tokens; limit is {self.config.max_seq_len}."
            )
        n_loops = self.config.max_loop_iters if n_loops is None else int(n_loops)
        n_loops = max(1, min(n_loops, self.config.max_loop_iters))
        x = self.embedding_dropout(self.token_embeddings(input_ids))
        for block in self.prelude:
            x = block(x)
        encoded = x
        hidden, ponder = self._recur(encoded, n_loops)
        for block in self.coda:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embeddings.weight)
        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss = lm_loss + ponder
        return {"logits": logits, "loss": loss, "ponder": ponder}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        stop_token_ids: set[int] | None = None,
        n_loops: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits = self(context, n_loops=n_loops)["logits"][:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < values[:, [-1]]] = -float("inf")
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if stop_token_ids and next_token.item() in stop_token_ids:
                break
        return input_ids


def create_mythos(config: dict | MythosConfig | None = None) -> MythosTransformer:
    if config is None:
        parsed = MythosConfig()
    elif isinstance(config, MythosConfig):
        parsed = config
    else:
        parsed = MythosConfig.from_dict(config)
    return MythosTransformer(parsed)

