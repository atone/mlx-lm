# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class DepthGPTArgs(BaseModelArgs):
    block_size: int = 8
    vocab_size: int = 2049
    n_layer: int = 6
    n_head: int = 16
    n_embd: int = 1024
    dropout: float = 0.0
    bias: bool = False
    main_hidden_size: int = 1536
    pad_token_id: int = 2048
    layer_norm_epsilon: float = 1e-6


class Attention(nn.Module):
    def __init__(self, args: DepthGPTArgs):
        super().__init__()
        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = args.n_head
        self.head_dim = args.n_embd // args.n_head

        self.scale = self.head_dim**-0.5

        self.c_attn = nn.Linear(args.n_embd, 3 * args.n_embd, bias=args.bias)
        self.c_proj = nn.Linear(args.n_embd, args.n_embd, bias=args.bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: DepthGPTArgs):
        super().__init__()
        self.intermediate_size = int(8 * args.n_embd / 3)

        self.gate_proj = nn.Linear(args.n_embd, self.intermediate_size, bias=args.bias)
        self.up_proj = nn.Linear(args.n_embd, self.intermediate_size, bias=args.bias)
        self.down_proj = nn.Linear(self.intermediate_size, args.n_embd, bias=args.bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class BlockCMLP(nn.Module):
    def __init__(self, args: DepthGPTArgs):
        super().__init__()
        self.epsilon = args.layer_norm_epsilon

        self.attn = Attention(args)
        self.mlps = [MLP(args) for _ in range(args.block_size)]
        self.ln_1 = nn.RMSNorm(args.n_embd, eps=self.epsilon)
        self.ln_2 = nn.RMSNorm(args.n_embd, eps=self.epsilon)

    def __call__(
        self,
        x: mx.array,
        step: int,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.attn(self.ln_1(x), mask, cache)
        h = x + r
        r = self.mlps[step](self.ln_2(h))
        out = h + r
        return out


class DepthGPT(nn.Module):
    def __init__(self, args: DepthGPTArgs):
        super().__init__()
        self.n_layer = args.n_layer
        self.epsilon = args.layer_norm_epsilon
        self.block_size = args.block_size

        self.linear_in = nn.Linear(args.main_hidden_size, args.n_embd * args.block_size, bias=False)
        self.transformer = dict(
            wtes = [nn.Embedding(args.vocab_size, args.n_embd) for _ in range(args.block_size)],
            wpe = nn.Embedding(args.block_size, args.n_embd),
            h = [BlockCMLP(args) for _ in range(args.n_layer)],
            ln_f = nn.RMSNorm(args.n_embd, eps=self.epsilon),
        )
        self.lm_heads = [nn.Linear(args.n_embd, args.vocab_size, bias=False) for _ in range(args.block_size)]

    def __call__(
        self,
        main_hidden_states: mx.array,
        audio_token_ids: mx.array,
        step: int,
        cache=None,
    ):
        assert main_hidden_states.shape[0] == audio_token_ids.shape[0], "Batch size must match"
        assert audio_token_ids.shape[1] == 1, "Only single step forward is supported"
        B, L = audio_token_ids.shape
        position_ids = mx.arange(L) + mx.array([step])

        x = self.transformer["wtes"][step](audio_token_ids) # (B, 1, n_embd)
        x += self.transformer["wpe"](position_ids) # (B, 1, n_embd)

        main_hidden = self.linear_in(main_hidden_states).reshape(B, self.block_size, -1)[:, step : step + L, :]
        x += main_hidden

        if cache is None:
            cache = [None] * self.n_layer

        mask = create_attention_mask(x, cache[0])

        for layer, cache in zip(self.transformer["h"], cache):
            x = layer(x, step, mask, cache)

        x = self.transformer["ln_f"](x) # (B, 1, n_embd)
        logits = self.lm_heads[step](x) # (B, 1, vocab_size)

        return logits
