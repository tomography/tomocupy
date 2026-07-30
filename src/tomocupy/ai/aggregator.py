# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any, Callable

from einops import rearrange
from tomocupy.ai.rope import RotaryPositionEmbedding2D, PositionGetter


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

class Attention(nn.Module):
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
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: Union[float, torch.Tensor] = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_add_residual_stochastic_depth(
    x: torch.Tensor, residual_func: Callable[[torch.Tensor], torch.Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> torch.Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor, pos=None) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path1(ffn_residual_func(x))  # FIXME: drop_path2
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x

class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # self.frame_blocks = nn.ModuleList(
        #     [
        #         block_fn(
        #             dim=embed_dim,
        #             num_heads=num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=qkv_bias,
        #             proj_bias=proj_bias,
        #             ffn_bias=ffn_bias,
        #             init_values=init_values,
        #             qk_norm=qk_norm,
        #             rope=self.rope,
        #         )
        #         for _ in range(depth)
        #     ]
        # )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        

        # Validate that depth is divisible by aa_block_size
        # if self.depth % self.aa_block_size != 0:
        #     raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        # self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        # self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.range_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        # nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.range_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        

        self.use_reentrant = False # hardcoded to False

    def forward(self, patch_tokens: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            patch_tokens (torch.Tensor): Input tokens with shape [B, R, S, H, W, C], in range [0, 1].
                B: batch size, R: number of windows, S: number of try reconstructions, H: number of tokens in the height dimension, W: number of tokens in the width dimension, C: token dimension.

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, R, S, H, W, C = patch_tokens.shape
        patch_tokens = rearrange(patch_tokens,'b r s h w c -> (b r) (s h w) c')
        

        # Expand camera and register tokens to match batch size and sequence length
        # camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        range_token = self.range_token.expand((B*R), *self.range_token.shape[1:])
        register_token = self.register_token.expand((B*R), *self.register_token.shape[1:]) #slice_expand_and_flatten(self.register_token, B, S)

        # Concatenate special tokens with patch tokens
        tokens = torch.cat([range_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * R * S, H, W, device=patch_tokens.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * R, self.patch_start_idx, 2).to(patch_tokens.device).to(pos.dtype)
            pos = torch.cat([pos_special, rearrange(pos,'(b s) n c -> b (s n) c',s=S)], dim=1)

        # update P because we added special tokens
        # _, P, C = tokens.shape

        # global_idx = 0
        
        for global_idx in range(self.depth):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
        tokens = rearrange(tokens, '(b r) n c -> b r n c',r=R)
        return tokens, self.patch_start_idx

    # def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
    #     """
    #     Process frame attention blocks. We keep tokens in shape (B*S, P, C).
    #     """
    #     # If needed, reshape tokens or positions:
    #     if tokens.shape != (B * S, P, C):
    #         tokens = tokens.view(B, S, P, C).view(B * S, P, C)

    #     if pos is not None and pos.shape != (B * S, P, 2):
    #         pos = pos.view(B, S, P, 2).view(B * S, P, 2)

    #     intermediates = []

    #     # by default, self.aa_block_size=1, which processes one block at a time
    #     for _ in range(self.aa_block_size):
    #         if self.training:
    #             tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
    #         else:
    #             tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
    #         frame_idx += 1
    #         intermediates.append(tokens.view(B, S, P, C))

    #     return tokens, frame_idx, intermediates

    # def _process_global_attention(self, tokens, global_idx, pos=None):
    #     """
    #     Process global attention blocks. We keep tokens in shape (B, S*P, C).
    #     """
    #     # if tokens.shape != (B, S * P, C):
    #     #     tokens = tokens.view(B, S, P, C).view(B, S * P, C)

    #     # if pos is not None and pos.shape != (B, S * P, 2):
    #     #     pos = pos.view(B, S, P, 2).view(B, S * P, 2)

    #     # intermediates = []

    #     # by default, self.aa_block_size=1, which processes one block at a time
    #     # for _ in range(self.aa_block_size):
    #     if self.training:
    #         tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
    #     else:
    #         tokens = self.global_blocks[global_idx](tokens, pos=pos)
    #     global_idx += 1
    #         # intermediates.append(tokens.view(B, S, P, C))

    #     return tokens, global_idx #, intermediates


