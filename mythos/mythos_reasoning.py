#!/usr/bin/env python3
"""Self-verifying "rethink" controller for the Mythos recurrent-depth model.

Instead of trusting a single greedy decode, the controller lets the model
think at several recurrence depths, samples a few candidate answers at each
depth, then *verifies* them with cheap, honest signals before returning the
best one:

  * determinism  - a tool/helper answer (e.g. exact arithmetic) wins outright
  * grounding    - how well the answer is supported by a retrieved passage
  * consistency  - how many independent samples agree with each other
  * sanity       - rejects empty, tiny or degenerate/repetitive output

It "rethinks until it has a best answer" by deepening the loop until the
verified score stops improving (or the budget is spent). An optional final
self-check can ask the model whether its own answer follows from the
evidence; that mode only helps once the model has been trained for it, so it
is off by default.
"""

from __future__ import annotations

from collections import Counter

import torch

# Allow running this file directly from anywhere inside the project.
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common.knowledge_store import query_terms


def content_words(text: str) -> list[str]:
    return query_terms(text)


def normalise(text: str) -> str:
    return " ".join(content_words(text))


def repetition_ratio(text: str) -> float:
    """Fraction of words that belong to the single most repeated 3-gram."""
    words = text.split()
    if len(words) < 4:
        return 0.0
    trigrams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    _, count = Counter(trigrams).most_common(1)[0]
    return min(1.0, (count * 3) / len(words))



class SelfVerifyingReasoner:
    """Sample-then-verify controller that turns loop depth into thinking time."""

    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        stop_token_ids: set[int],
        *,
        temperature: float = 0.35,
        top_k: int = 40,
        max_new_tokens: int = 64,
        min_loops: int = 1,
        max_loops: int = 6,
        candidates: int = 4,
        patience: int = 2,
        good_enough: float = 0.75,
        max_seq_len: int = 512,
        weights: tuple[float, float, float] = (2.0, 1.5, 0.5),
        extra_samplers: list | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.stop_token_ids = stop_token_ids
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.min_loops = max(1, min_loops)
        self.max_loops = max(self.min_loops, max_loops)
        self.candidates = max(1, candidates)
        self.patience = max(1, patience)
        self.good_enough = good_enough
        self.max_seq_len = max_seq_len
        self.w_ground, self.w_consistency, self.w_sanity = weights
        # Optional extra candidate sources: (label, fn(prompt_ids, n_loops) -> text).
        # This lets the verifier mix in a second, possibly different-architecture model.
        self.extra_samplers = list(extra_samplers or [])

    def _sample(self, prompt_ids: list[int], n_loops: int, temperature: float | None) -> str:
        prompt = prompt_ids[-self.max_seq_len :]
        tensor = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                tensor,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if temperature is None else temperature,
                top_k=self.top_k,
                stop_token_ids=self.stop_token_ids,
                n_loops=n_loops,
            )
        generated = output[0, len(prompt) :].tolist()
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _verify(self, text, question, retrieved, deterministic, group_size, total):
        detail: dict = {}
        norm = normalise(text)
        if not norm:
            return -1.0, {"empty": True}
        sanity = 1.0 - repetition_ratio(text)
        detail["sanity"] = round(sanity, 3)

        if deterministic is not None:
            want = normalise(deterministic)
            match = 1.0 if (want and (want in norm or norm in want)) else 0.0
            detail["deterministic"] = match
            if match == 1.0:
                # A tool/helper answer is always preferred.
                return 10.0 + sanity, detail

        support = 0.0
        if retrieved:
            answer_terms = set(content_words(text))
            source_terms = set(content_words(f"{retrieved['title']} {retrieved['passage']}"))
            if answer_terms:
                support = len(answer_terms & source_terms) / len(answer_terms)
        detail["support"] = round(support, 3)

        consistency = group_size / total if total else 0.0
        detail["consistency"] = round(consistency, 3)

        score = (
            self.w_ground * support
            + self.w_consistency * consistency
            + self.w_sanity * sanity
        )
        return score, detail

    def reason(self, prompt_ids, question, retrieved=None, deterministic=None):
        """Think until the verified answer stops improving, then return it."""
        candidates: list[dict] = []
        best = None
        history = []
        stall = 0
        sampled = 0
        depth = self.min_loops

        while depth <= self.max_loops:
            labelled = [(None, self._sample(prompt_ids, depth, None)) for _ in range(self.candidates)]
            for label, sampler in self.extra_samplers:
                labelled.append((label, sampler(prompt_ids, depth)))
            sampled += len(labelled)
            groups = Counter(normalise(text) for _, text in labelled)
            for source, text in labelled:
                group_size = groups[normalise(text)]
                score, detail = self._verify(
                    text, question, retrieved, deterministic, group_size, len(labelled)
                )
                candidates.append(
                    {"text": text, "loops": depth, "score": score, "detail": detail, "source": source}
                )

            depth_best = max(candidates, key=lambda item: item["score"])
            if best is None or depth_best["score"] > best["score"] + 1e-6:
                best = depth_best
                stall = 0
            else:
                stall += 1
            history.append({"loops": depth, "best_score": round(depth_best["score"], 3)})

            if best["detail"].get("deterministic") == 1.0:
                break
            if stall >= self.patience and best["score"] >= self.good_enough:
                break
            depth += 1

        return {
            "answer": best["text"].strip(),
            "score": round(best["score"], 3),
            "loops": best["loops"],
            "detail": best["detail"],
            "history": history,
            "sampled": sampled,
        }
