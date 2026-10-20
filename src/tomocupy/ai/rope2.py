""" Sin-cos, fourier, rotary position embedding modules and functions

Hacked together by / Copyright 2022 Ross Wightman

Inspired by:
        https://github.com/meta-llama/codellama/blob/main/llama/model.py
        https://github.com/naver-ai/rope-vit
        https://github.com/facebookresearch/vggt/blob/main/vggt/layers/rope.py

Modifications Copyright 2026 tomocupy authors

"""

import math
import torch
from typing import List, Tuple, Optional, Union
from torch import nn as nn
import torch.nn.functional as F
from einops import rearrange

def rot(x):
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x1  x0 -x3  x2 -x5  x4]
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def rope_rotate_half(x: torch.Tensor) -> torch.Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rot_embed_cat(
        x: torch.Tensor,
        emb: torch.Tensor,
        pos: torch.Tensor,
        half: bool = True
) -> torch.Tensor:
    sin_emb, cos_emb = emb.chunk(2, -1)
    cos = F.embedding(pos, cos_emb)[:, None, :, :]
    sin = F.embedding(pos, sin_emb)[:, None, :, :]
    # x: [..., D], eg [x0, x1, x2, x3, x4, x5]
    if half:
        # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
        # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2
        # rope_rotate_half(x), eg [-x3, -x4, -x5, x0, x1, x2]
        return x * cos + rope_rotate_half(x) * sin
    else:
        # sin: [..., D], eg [sin0, sin0, sin1, sin1, sin2, sin2]
        # cos: [..., D], eg [cos0, cos0, cos1, cos1, cos2, cos2]
        # rot(x), eg [-x1, x0, -x3, x2, -x5, x4]
        return x * cos + rot(x) * sin

def pixel_freq_bands(dim, base_frequency = 100, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
    exponents = torch.arange(0, dim, 2, device=device, dtype=dtype).float() / dim
    bands = 1.0 / (base_frequency**exponents)
    return bands

def freq_bands(
        num_bands: int,
        temperature: float = 10000.,
        step: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.int64,
) -> torch.Tensor:
    exp = torch.arange(0, num_bands, step, dtype=dtype, device=device).to(torch.float32) / num_bands
    bands = 1. / (temperature ** exp)
    return bands

def swap_shape_xy(seq: List[int]) -> List[int]:
    if len(seq) < 2:
        return seq
    return [seq[1], seq[0]] + list(seq[2:])

def build_fourier_pos_embed(
        feat_shape: List[int],
        bands: torch.Tensor,
        dtype: torch.dtype,
        include_grid: bool = False,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
) -> List[torch.Tensor]:
    """

    Args:
        feat_shape: Feature shape for embedding.
        bands: Pre-calculated frequency bands.
        include_grid: Include the spatial grid in output.
        in_pixels: Output in pixel freq.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.

    Returns:

    """
    assert bands is not None

    if grid_indexing == 'xy':
        feat_shape = swap_shape_xy(feat_shape)
        if ref_feat_shape is not None:
            ref_feat_shape = swap_shape_xy(ref_feat_shape)

    if in_pixels:
        if len(feat_shape) == 1:
            t = [torch.arange(feat_shape[0], device=bands.device, dtype=bands.dtype)]
        else:
            t = [
                torch.linspace(-1., 1., steps=s, device=bands.device, dtype=bands.dtype)
                for s in feat_shape
            ]
    else:
        t = [
            torch.arange(s, device=bands.device, dtype=torch.int64).to(bands.dtype) + grid_offset
            for s in feat_shape
        ]

    if ref_feat_shape is not None:
        # eva's scheme for resizing rope embeddings (ref shape = pretrain)
        t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    if len(t) == 1:
        grid = t[0].unsqueeze(-1)
    else:
        grid = torch.stack(torch.meshgrid(t, indexing=grid_indexing), dim=-1)
        grid = grid.unsqueeze(-1)
    pos = grid * bands
    pos = pos.to(dtype)
    pos = torch.cat((pos, pos), dim=-1)
    pos_sin, pos_cos = pos.sin().to(dtype=dtype), pos.cos().to(dtype=dtype)
    out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
    return out

def build_rotary_pos_embed(
        feat_shape: List[int],
        bands: torch.Tensor,
        dtype: torch.dtype,
        in_pixels: bool = True,
        ref_feat_shape: Optional[List[int]] = None,
        grid_offset: float = 0.,
        grid_indexing: str = 'ij',
):
    """

    Args:
        feat_shape: Spatial shape of the target tensor for embedding.
        bands: pre-generated frequency bands
        in_pixels: Pixel vs language (inv freq) mode.
        ref_feat_shape: Reference feature shape for resize / fine-tune.
        grid_offset: Constant offset to add to grid for non-pixel freq.
        grid_indexing: Indexing mode for meshgrid ('ij' or 'xy')
        dtype: Output dtype.

    Returns:

    """
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands,
        dtype,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
    )
    if len(feat_shape) == 1:
        return sin_emb, cos_emb
    else:
        num_spatial_dim = 1
        # this would be much nicer as a .numel() call to torch.Size(), but torchscript sucks
        for x in feat_shape:
            num_spatial_dim *= x
        sin_emb = sin_emb.reshape(num_spatial_dim, -1).repeat(1, 2)
        cos_emb = cos_emb.reshape(num_spatial_dim, -1).repeat(1, 2)
        return sin_emb, cos_emb

class RotaryEmbeddingCat(nn.Module):
    """ Rotary position embedding w/ concatenatd sin & cos

    The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    """

    def __init__(
            self,
            dim: int,
            temperature: float = 100,
            in_pixels: bool = True,
            ref_feat_shape: Optional[List[int]] = None,
            grid_offset: float = 0.,
            grid_indexing: str = 'ij',
    ):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing
        self.pos_embed_cached = {}
        

    def get_embed(self, bands:torch.Tensor, dtype:torch.dtype, shape: Optional[List[int]] = None):
        if shape is not None:
            # rebuild embeddings from cached bands every call, use if target shape changes
            embeds = build_rotary_pos_embed(
                shape,
                bands,
                dtype,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
            return torch.cat(embeds, -1)
        else:
            assert False, "get_embed() requires pre-computed pos embed or valid shape w/ pre-computed bands"

    def forward(self, x:torch.Tensor, positions:torch.Tensor):
        # assuming channel-first tensor where spatial dim are >= 2
        # x of shape: (b*r), m, (1+4+(s*h*w)), c
        assert x.size(-1) % 2 == 0, "Feature dimension must be even"
        max_position = int(positions.max()) + 1
        if (max_position,x.device,x.dtype) not in self.pos_embed_cached.keys():

            if self.in_pixels:
                bands = pixel_freq_bands(
                    self.dim // 2,
                    base_frequency=self.temperature,
                    device = x.device,
                    dtype = x.dtype,
                )
            else:
                bands = freq_bands(
                    self.dim // 4,
                    temperature=self.temperature,
                    step=1,
                    device = x.device,
                    dtype = x.dtype,
                )

            pos_embed = self.get_embed(bands,x.dtype,[max_position])
            self.pos_embed_cached[(max_position,x.device,x.dtype)] = pos_embed
        else:
            pos_embed = self.pos_embed_cached[(max_position,x.device,x.dtype)]
        x_v, x_h = x.chunk(2, dim=-1)
        x_v = apply_rot_embed_cat(x_v, pos_embed, positions[...,0])
        x_h = apply_rot_embed_cat(x_h, pos_embed, positions[...,1])

        return torch.cat((x_v, x_h), dim=-1)