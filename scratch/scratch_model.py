#!/usr/bin/env python3
"""A small decoder-only Transformer language model initialized from random weights."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    dim: int = 384
    layers: int = 8
    heads: int = 6
    kv_heads: int = 2
    hidden_dim: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.1

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        accepted = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in accepted})

    def to_dict(self) -> dict:
        return asdict(self)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, theta: float):
    """Apply rotary position embeddings to [batch, heads, time, head_dim] tensors."""
    head_dim = q.shape[-1]
    positions = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim)
    )
    frequencies = torch.outer(positions, inv_freq)
    phases = torch.cat((frequencies, frequencies), dim=-1)[None, None, :, :]
    cos, sin = phases.cos().to(q.dtype), phases.sin().to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * scale.to(x.dtype)) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.head_dim = config.dim // config.heads
        self.heads = config.heads
        self.kv_heads = config.kv_heads
        self.q_proj = nn.Linear(config.dim, config.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.rope_theta = config.rope_theta
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape
        q = self.q_proj(x).view(batch, time, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, time, self.kv_heads, self.head_dim).transpose(1, 2)
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
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.dim)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim)
        self.feed_forward = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        return x + self.residual_dropout(self.feed_forward(self.ffn_norm(x)))


class ScratchTransformer(nn.Module):
    """GPT-style causal LM; every trainable weight starts from random initialization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.dim % config.heads or config.heads % config.kv_heads:
            raise ValueError("dim must divide heads, and heads must divide evenly across kv_heads.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.layers))
        self.final_norm = RMSNorm(config.dim)
        self.apply(self._init_weights)
        # Share input/output embeddings to keep the laptop-sized model compact.
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

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if input_ids.shape[-1] > self.config.max_seq_len:
            raise ValueError(
                f"Sequence has {input_ids.shape[-1]} tokens; limit is {self.config.max_seq_len}."
            )
        x = self.embedding_dropout(self.token_embeddings(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embeddings.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 40,
        stop_token_ids: set[int] | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits = self(context)["logits"][:, -1, :]
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


def create_model(config: dict | ModelConfig | None = None) -> ScratchTransformer:
    if config is None:
        parsed = ModelConfig()
    elif isinstance(config, ModelConfig):
        parsed = config
    else:
        parsed = ModelConfig.from_dict(config)
    return ScratchTransformer(parsed)
