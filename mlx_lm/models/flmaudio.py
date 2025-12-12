# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import KVCache, RotatingKVCache
from .rope_utils import initialize_rope
from .depth_gpt import DepthGPT, DepthGPTArgs


@dataclass
class TokenInfo(BaseModelArgs):
    text_wait_token_id: int
    aud_pad_token_id: int
    aud_emp_token_id: int


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    aud_channel: int
    aud_vocab_size: int
    aud_depthgpt: DepthGPTArgs
    mm_token_info: TokenInfo
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    attention_bias: bool = False
    disable_att_o_bias: bool = True
    mlp_bias: bool = False
    rope_theta: float = 10000
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    layer_types: Optional[List[str]] = None
    sliding_window: Optional[int] = None
    use_mup: bool = False
    mup_scale_factor: float = 1.0
    output_mult: float = 1.0
    input_mult: float = 1.0


    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers

        if isinstance(self.aud_depthgpt, dict):
            self.aud_depthgpt = DepthGPTArgs.from_dict(self.aud_depthgpt)

        if isinstance(self.mm_token_info, dict):
            self.mm_token_info = TokenInfo.from_dict(self.mm_token_info)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        self.head_dim = head_dim = args.head_dim or args.hidden_size // n_heads

        self.scale = head_dim**-0.5
        if hasattr(args, "attention_bias"):
            attention_bias = args.attention_bias
        else:
            attention_bias = False

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=attention_bias and not args.disable_att_o_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            args.rope_traditional,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        hidden_dim = args.intermediate_size
        if hasattr(args, "mlp_bias"):
            mlp_bias = args.mlp_bias
        else:
            mlp_bias = False

        self.gate_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=mlp_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=mlp_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, use_sliding: bool = False):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.use_sliding = use_sliding
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class MultiModalEmbedding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.use_mup = args.use_mup
        self.input_mult = args.input_mult
        self.aud_channel = args.aud_channel

        self.text_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.aud_listen_embeddings = [
            nn.Embedding(args.aud_vocab_size, args.hidden_size)
            for _ in range(args.aud_channel)
        ]
        self.aud_speak_embeddings = [
            nn.Embedding(args.aud_vocab_size, args.hidden_size)
            for _ in range(args.aud_channel)
        ]

    def __call__(self, text_ids: mx.array, speak_ids: mx.array, listen_ids: mx.array) -> mx.array:
        assert text_ids is not None
        embeddings = self.text_embeddings(text_ids)

        for aud_chn_idx in range(self.aud_channel):
            aud_speak_emb = self.aud_speak_embeddings[aud_chn_idx](speak_ids[..., aud_chn_idx])
            aud_listen_emb = self.aud_listen_embeddings[aud_chn_idx](listen_ids[..., aud_chn_idx])
            embeddings += aud_speak_emb + aud_listen_emb

        if self.use_mup:
            embeddings = embeddings * self.input_mult
        return embeddings


class FLMAudioModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        assert self.vocab_size > 0
        self.embed_tokens = MultiModalEmbedding(args)
        self.layers = [
            TransformerBlock(args=args, use_sliding=layer_type == "sliding_attention")
            for layer_type in self.layer_types
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.fa_idx = self.layer_types.index("full_attention")
        self.swa_idx = None
        for e, l in enumerate(self.layers):
            if l.use_sliding:
                self.swa_idx = e
                break

    def __call__(
        self,
        input_ids: mx.array,
        speak_ids: mx.array,
        listen_ids: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(input_ids, speak_ids, listen_ids)

        if cache is None:
            cache = [None] * len(self.layers)

        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        if self.swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[self.swa_idx], window_size=self.sliding_window
            )

        for layer, cache in zip(self.layers, cache):
            mask = swa_mask if layer.use_sliding else fa_mask
            h = layer(h, mask, cache=cache)

        return self.norm(h)


@dataclass
class FLMAudioOutput:
    logits: mx.array
    hidden_states: mx.array


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.audio_args = DepthGPTArgs(
            block_size=args.aud_channel,
            vocab_size=args.aud_vocab_size,
            n_layer=args.aud_depthgpt.n_layer,
            n_head=args.aud_depthgpt.n_head,
            n_embd=args.aud_depthgpt.n_embd,
            dropout=args.aud_depthgpt.dropout,
            bias=args.aud_depthgpt.bias,
            main_hidden_size=args.hidden_size,
            pad_token_id=args.mm_token_info.aud_emp_token_id,
        )
        self.use_mup = args.use_mup
        self.output_mult = args.output_mult / args.mup_scale_factor if self.use_mup else 1.0
        self.model_type = args.model_type
        self.text_wait_token_id = args.mm_token_info.text_wait_token_id
        self.model = FLMAudioModel(args)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self.audio = DepthGPT(self.audio_args)

    def __call__(
        self,
        input_ids: mx.array,
        speak_ids: mx.array,
        listen_ids: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        hidden_states = self.model(input_ids, speak_ids, listen_ids, cache, input_embeddings)
        out = self.lm_head(hidden_states)

        if self.use_mup:
            out = out * self.output_mult

        return FLMAudioOutput(logits=out, hidden_states=hidden_states)

    def sanitize(self, weights):
        def transform_key(key: str) -> str:
            if key.startswith("aud_output_layers."):
                return "audio." + key[len("aud_output_layers."):]
            return key
        # Remove unused precomputed rotary freqs
        weights = {
            transform_key(k): v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        return weights

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.model.sliding_window)
                if layer.use_sliding
                else KVCache()
            )
            for layer in self.layers
        ]

    def _get_initial_token(self) -> mx.array:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        text_special = self.args.mm_token_info.text_wait_token_id
        audio_special = self.args.mm_token_info.aud_pad_token_id
        token = mx.array(
            [text_special] + [audio_special] * (2 * self.args.aud_channel),
            dtype=mx.int32,
        )  # [K]
        return token
