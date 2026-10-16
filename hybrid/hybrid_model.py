#!/usr/bin/env python3
"""Hybrid: a Transformer + Mythos recurrent-depth model in one network.

This mixes the two architectures already in this project so they cooperate
instead of competing:

  * the plain Transformer path gives fast, unique-weight *parallel depth*
    (each stage has its own block, good at memorising patterns), and
  * the Mythos recurrent path gives weight-shared *recurrent depth* (one
    block looped, good at composing and at thinking longer on hard inputs).

Each :class:`HybridStage` runs both paths on the same hidden state and fuses
them with a learned per-channel gate, so the network itself decides, per
channel, how much to trust parallel memory vs. recurrent reasoning. The loop
count stays a runtime choice, so one checkpoint can think longer about
harder questions.

Layout:

    embeddings
      -> Prelude (standard blocks, once)
      -> [ HybridStage x stages:  standard block  ++gate++  looped block ]
      -> Coda (standard blocks, once)
      -> final norm -> tied logits
"""

from __future__ import annotations

import math
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

from scratch.scratch_model import ModelConfig, RMSNorm, TransformerBlock
from mythos.mythos_model import RecurrentBlock


@dataclass
class HybridConfig(ModelConfig):
    """Configuration for the dual-path Transformer + recurrent-depth hybrid."""

    prelude_layers: int = 2
    coda_layers: int = 2
    stages: int = 2
    max_loop_iters: int = 4
    depth_lora_rank: int = 8
    use_halting: bool = True
    halting_threshold: float = 0.95
    ponder_weight: float = 0.01
    # Initial weight given to the standard (parallel) path when fusing.
    standard_weight: float = 0.5

    @classmethod
    def from_dict(cls, values: dict) -> "HybridConfig":
        accepted = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in accepted})

    def to_dict(self) -> dict:
        return asdict(self)


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-4), 1.0 - 1e-4)
    return math.log(probability / (1.0 - probability))


class HybridStage(nn.Module):
    """One hybrid stage: a standard block and a looped recurrent block, fused.

    The standard path is a unique :class:`TransformerBlock` (parallel depth).
    The recurrent path is a shared :class:`RecurrentBlock` looped up to
    ``max_loop_iters`` times with LTI-stable input injection. A learned
    per-channel gate decides how much of each path to keep.
    """

    def __init__(self, config: HybridConfig):
        super().__init__()
        self.standard = TransformerBlock(config)
        self.recurrent = RecurrentBlock(config)
        # Gate logits -> sigmoid -> weight of the standard path per channel.
        self.fuse = nn.Linear(config.dim, config.dim)
        nn.init.normal_(self.fuse.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.fuse.bias, _logit(config.standard_weight))
        # LTI injection for the recurrent path: h <- A*h + B*e.
        self.inject_a = nn.Parameter(torch.zeros(config.dim))
        self.inject_b = nn.Parameter(torch.zeros(config.dim))
        self.halt_head = nn.Linear(config.dim, 1) if config.use_halting else None
        if self.halt_head is not None:
            nn.init.zeros_(self.halt_head.weight)
            nn.init.constant_(self.halt_head.bias, -1.0)
        self.halting_threshold = config.halting_threshold
        self.ponder_weight = config.ponder_weight

    def spectral_radius(self) -> float:
        return float((torch.sigmoid(self.inject_a) * 0.99).abs().max().item())

    def _recur(self, e: torch.Tensor, n_loops: int):
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
                halted = halted | (cumulative >= self.halting_threshold)
            else:
                h = h_new
        ponder = torch.zeros((), device=device)
        if use_halting:
            ponder = step_count.mean() * self.ponder_weight
        return h, ponder

    def forward(self, x: torch.Tensor, n_loops: int):
        standard_out = self.standard(x)
        recurrent_out, ponder = self._recur(x, n_loops)
        gate = torch.sigmoid(self.fuse(x))
        return gate * standard_out + (1.0 - gate) * recurrent_out, ponder


class HybridTransformer(nn.Module):
    """Transformer + recurrent-depth hybrid: Prelude -> hybrid stages -> Coda."""

    def __init__(self, config: HybridConfig):
        super().__init__()
        if config.dim % config.heads or config.heads % config.kv_heads:
            raise ValueError("dim must divide heads, and heads must divide evenly across kv_heads.")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if config.stages < 1 or config.max_loop_iters < 1:
            raise ValueError("stages and max_loop_iters must be at least 1.")
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.prelude = nn.ModuleList(TransformerBlock(config) for _ in range(config.prelude_layers))
        self.stages = nn.ModuleList(HybridStage(config) for _ in range(config.stages))
        self.coda = nn.ModuleList(TransformerBlock(config) for _ in range(config.coda_layers))
        self.final_norm = RMSNorm(config.dim)
        self.apply(self._init_weights)
        # Keep the fusion gates and halting heads at their intended start values.
        for stage in self.stages:
            nn.init.normal_(stage.fuse.weight, mean=0.0, std=0.02)
            nn.init.constant_(stage.fuse.bias, _logit(config.standard_weight))
            if stage.halt_head is not None:
                nn.init.zeros_(stage.halt_head.weight)
                nn.init.constant_(stage.halt_head.bias, -1.0)
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
        """Largest spectral radius across every stage's injection matrix."""
        return max(stage.spectral_radius() for stage in self.stages)

    def effective_depth(self, n_loops: int) -> int:
        """Total block evaluations for a given loop count."""
        return (
            self.config.prelude_layers
            + self.config.stages * (1 + n_loops)
            + self.config.coda_layers
        )

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
        ponder = torch.zeros((), device=input_ids.device)
        for stage in self.stages:
            x, stage_ponder = stage(x, n_loops)
            ponder = ponder + stage_ponder
        for block in self.coda:
            x = block(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.token_embeddings.weight)
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


def create_hybrid(config: dict | HybridConfig | None = None) -> HybridTransformer:
    if config is None:
        parsed = HybridConfig()
    elif isinstance(config, HybridConfig):
        parsed = config
    else:
        parsed = HybridConfig.from_dict(config)
    return HybridTransformer(parsed)

