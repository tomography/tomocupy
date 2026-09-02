# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE_DINOV2 file in the /third_party_licenses directory of this source tree.

# The VisionTransformerAggregator is based on the implementation from https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py.

# Inspired by:
# https://github.com/naver-ai/rope-vit/blob/main/models/vit_rope.py
# https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py

# Modifications Copyright 2026 tomocupy authors

import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Callable, Type

from einops import rearrange
from tomocupy.ai.rope2 import RotaryEmbeddingCat


logger = logging.getLogger(__name__)

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output

class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: Tensor, is_causal: bool = False, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop.p if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(nn.Module):
    """Transformer block with pre-normalization."""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
            mlp_layer: Type[nn.Module] = Mlp,
            attn_layer: Type[nn.Module] = AttentionRope,
            rope = None,
    ) -> None:
        """Initialize Block.

        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            qk_norm: If True, apply normalization to query and key.
            proj_bias: If True, add bias to output projection.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            init_values: Initial values for layer scale.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            mlp_layer: MLP layer.
            attn_layer: Attention layer type (class or string).
        """
        super().__init__()
        

        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            rope=rope,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            is_causal: bool = False,
            pos = None,
    ) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), is_causal=is_causal, pos=pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class VisionTransformerAggregator(nn.Module):
    def __init__(
            self,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = True,
            proj_bias: bool = True,
            init_values: float = 0.01,
            class_token: bool = True,
            reg_tokens: int = 4,
            block_fn: nn.Module = Block,
            rope_freq: float = 100.
        ):
        super().__init__()

        assert embed_dim % num_heads == 0
        if rope_freq > 0:
            self.rope = RotaryEmbeddingCat(dim=embed_dim//num_heads,temperature=rope_freq)
        else:
            self.rope = None

        self.use_reentrant = False
        self.num_prefix_tokens = 1 + reg_tokens if class_token else reg_tokens
        self.positions_cached = {}
        self.global_blocks = nn.ModuleList(
                    [
                        block_fn(
                            dim=embed_dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            proj_bias=proj_bias,
                            init_values=init_values,
                            qk_norm=qk_norm,
                            rope=self.rope,
                        )
                        for _ in range(depth)
                    ]
                )
        self.range_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if class_token else None
        self.register_token = nn.Parameter(torch.randn(1, reg_tokens, embed_dim)) if reg_tokens else None
        self.init_weights()

    def init_weights(self):
        if self.range_token is not None:
            nn.init.normal_(self.range_token, std=1e-6)
        if self.register_token is not None:
            nn.init.normal_(self.register_token, std=1e-6)

    def _pos_embed(self, x, x_shape_):
        B, R, S, H, W, C = x_shape_
        to_cat = []
        if self.range_token is not None:
            to_cat.append(self.range_token.expand((B*R), *self.range_token.shape[1:]))
        if self.register_token is not None:
            to_cat.append(self.register_token.expand((B*R), *self.register_token.shape[1:]))
        x = torch.cat(to_cat + [x], dim=1)

        positions = None
        if self.rope is not None:
            if (H, W) not in self.positions_cached.keys():
                ys = torch.arange(H, device=x.device)
                xs = torch.arange(W, device=x.device)
                positions = torch.cartesian_prod(ys, xs)
                self.positions_cached[H, W] = positions
            else:
                positions = self.positions_cached[H, W]
            positions = rearrange(positions.view(1, H * W, 2).expand(B * R * S, -1, -1).clone(),'(b s) n c -> b (s n) c',s=S)
            if self.num_prefix_tokens > 0:
                positions = F.pad(positions+1,(0,0,self.num_prefix_tokens,0),mode='constant',value=0.)
        return x, positions


    def forward(self, x: Tensor) -> Tuple[List[Tensor], int]:
        x_shape_ = x.shape
        x = rearrange(x,'b r s h w c -> (b r) (s h w) c')
        x, positions = self._pos_embed(x, x_shape_)

        for blk in self.global_blocks:
            if self.training:
                x = checkpoint(blk, x, positions, use_reentrant=self.use_reentrant)
            else:
                x = blk(x, pos=positions)
        x = rearrange(x, '(b r) n c -> b r n c',r=x_shape_[1])
        return x, self.num_prefix_tokens