"""Minimal stand-in for `utils3d_moge`, the git-pinned helper MoGe depends on.

Only the handful of geometry helpers MoGe calls are implemented, because the
real package is a GitHub-only dependency and GitHub is unreachable from
repo-scoped sandboxes. Each function matches the upstream signature and
semantics used at MoGe's call sites.
"""
from __future__ import annotations

import numpy as np
import torch


class _PT:
    @staticmethod
    def intrinsics_from_focal_center(fx, fy, cx, cy):
        fx, fy, cx, cy = (torch.as_tensor(v) for v in (fx, fy, cx, cy))
        shape = torch.broadcast_shapes(fx.shape, fy.shape, cx.shape, cy.shape)
        fx, fy, cx, cy = (v.expand(shape) for v in (fx, fy, cx, cy))
        zero = torch.zeros_like(fx)
        one = torch.ones_like(fx)
        return torch.stack([fx, zero, cx, zero, fy, cy, zero, zero, one],
                           dim=-1).reshape(*shape, 3, 3)

    @staticmethod
    def uv_map(size, device=None, dtype=None):
        """Pixel-centre UV coordinates in [0,1], shape (H, W, 2)."""
        h, w = size
        v = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        u = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([uu, vv], dim=-1)

    @staticmethod
    def pixel_coord_map(size, device=None, dtype=None):
        """Pixel-centre coordinates in pixels, shape (H, W, 2)."""
        h, w = size
        v = torch.arange(h, device=device, dtype=dtype) + 0.5
        u = torch.arange(w, device=device, dtype=dtype) + 0.5
        vv, uu = torch.meshgrid(v, u, indexing="ij")
        return torch.stack([uu, vv], dim=-1)

    @staticmethod
    def sliding_window(x, window_size, stride=1, dim=(-2, -1)):
        if isinstance(window_size, int):
            window_size = (window_size,) * len(dim)
        if isinstance(stride, int):
            stride = (stride,) * len(dim)
        out = x
        for d, ws, st in zip(dim, window_size, stride):
            out = out.unfold(d, ws, st)
        return out

    @staticmethod
    def depth_map_to_point_map(depth, intrinsics):
        """Back-project a depth map with normalised intrinsics (cx,cy,fx,fy in [0,1])."""
        *batch, h, w = depth.shape
        uv = _PT.uv_map((h, w), device=depth.device, dtype=depth.dtype)
        inv = torch.inverse(intrinsics)
        ones = torch.ones_like(uv[..., :1])
        hom = torch.cat([uv, ones], dim=-1)
        dirs = torch.einsum("...ij,hwj->...hwi", inv, hom)
        return dirs * depth[..., None]


class _NP:
    @staticmethod
    def uv_map(size, dtype=np.float32):
        h, w = size
        v = (np.arange(h, dtype=dtype) + 0.5) / h
        u = (np.arange(w, dtype=dtype) + 0.5) / w
        uu, vv = np.meshgrid(u, v)
        return np.stack([uu, vv], axis=-1)

    @staticmethod
    def pixel_coord_map(size, dtype=np.float32):
        h, w = size
        uu, vv = np.meshgrid(np.arange(w, dtype=dtype) + 0.5,
                             np.arange(h, dtype=dtype) + 0.5)
        return np.stack([uu, vv], axis=-1)

    @staticmethod
    def sliding_window(x, window_size, stride=1, dim=(-2, -1)):
        t = _PT.sliding_window(torch.as_tensor(x), window_size, stride, dim)
        return t.numpy()

    @staticmethod
    def masked_nearest_resize(*args, **kwargs):
        raise NotImplementedError("not needed on the MoGe inference path")


pt = _PT()
np_ = _NP()
globals()["np"] = _NP()  # MoGe calls utils3d.np.*
