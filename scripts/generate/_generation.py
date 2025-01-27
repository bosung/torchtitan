# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
from torchtitan.logging import logger

def multinomial_sample_one(
    probs: torch.Tensor, rng: Optional[torch.Generator] = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(
    self,
    logits,
    need_probs: bool,
    temperature: float = 0,
    top_k: Optional[int] = None,
):
    logits = logits[0, -1]
    logger.debug("Logits: %s", logits)
    if temperature == 0 and not need_probs:
        _, idx_next = torch.topk(logits, k=1, dim=-1)
        return (idx_next, None)
    probs = logits_to_probs(logits, temperature, top_k)
    #idx_next = self.multinomial_sample_one_no_sync(probs)
    idx_next = multinomial_sample_one(probs)
    return idx_next, probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    logits = model(x)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    for i in range(max_new_tokens):
        logger.info(f" -- {i}: {generated_tokens.shape}")
        next_token = generate_next_token(
            model,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens
