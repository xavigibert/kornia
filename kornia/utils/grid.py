# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from kornia.core import Tensor, stack
from kornia.utils._compat import torch_meshgrid

# --------- Memoization helpers for kernels and meshgrids ---------

# Cache for commonly-used window center kernels, window grid kernels, and meshgrids
# as (h,w,device,dtype) for proper per-device computation and fp32/fp16 support

_center_kernel2d_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
_window_grid_kernel2d_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}
_meshgrid_cache: Dict[Tuple[int, int, int, str, str], torch.Tensor] = {}


def create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, H, W, 2)`.

    Example:
        >>> create_meshgrid(2, 2)
        tensor([[[[-1., -1.],
                  [ 1., -1.]],
        <BLANKLINE>
                 [[-1.,  1.],
                  [ 1.,  1.]]]])

        >>> create_meshgrid(2, 2, normalized_coordinates=False)
        tensor([[[[0., 0.],
                  [1., 0.]],
        <BLANKLINE>
                 [[0., 1.],
                  [1., 1.]]]])

    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    # Fix TracerWarning
    # Note: normalize_pixel_coordinates still gots TracerWarning since new width and height
    #       tensors will be generated.
    # Below is the code using normalize_pixel_coordinates:
    # base_grid: torch.Tensor = torch.stack(torch.meshgrid([xs, ys]), dim=2)
    # if normalized_coordinates:
    #     base_grid = K.geometry.normalize_pixel_coordinates(base_grid, height, width)
    # return torch.unsqueeze(base_grid.transpose(0, 1), dim=0)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    # TODO: torchscript doesn't like `torch_version_ge`
    # if torch_version_ge(1, 13, 0):
    #     x, y = torch_meshgrid([xs, ys], indexing="xy")
    #     return stack([x, y], -1).unsqueeze(0)  # 1xHxWx2
    # TODO: remove after we drop support of old versions
    base_grid: Tensor = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    return base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2


def create_meshgrid3d(
    depth: int,
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Generate a coordinate grid for an image.

    When the flag ``normalized_coordinates`` is set to True, the grid is
    normalized to be in the range :math:`[-1,1]` to be consistent with the pytorch
    function :py:func:`torch.nn.functional.grid_sample`.

    Args:
        depth: the image depth (channels).
        height: the image height (rows).
        width: the image width (cols).
        normalized_coordinates: whether to normalize
          coordinates in the range :math:`[-1,1]` in order to be consistent with the
          PyTorch function :py:func:`torch.nn.functional.grid_sample`.
        device: the device on which the grid will be generated.
        dtype: the data type of the generated grid.

    Return:
        grid tensor with shape :math:`(1, D, H, W, 3)`.

    """
    xs: Tensor = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys: Tensor = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    zs: Tensor = torch.linspace(0, depth - 1, depth, device=device, dtype=dtype)
    # Fix TracerWarning
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
        zs = (zs / (depth - 1) - 0.5) * 2
    # generate grid by stacking coordinates
    base_grid = stack(torch_meshgrid([zs, xs, ys], indexing="ij"), dim=-1)  # DxWxHx3
    return base_grid.permute(0, 2, 1, 3).unsqueeze(0)  # 1xDxHxWx3


def _get_device_id(device):
    # Returns a stable id for cache key: CPU=0, CUDA=unique index
    if device.type == "cpu":
        return 0
    return torch.cuda.device(device) if hasattr(torch.cuda, "device") else device.index


def _fast_get_window_grid_kernel2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Optimized and cached version
    cache_key = (h, w, _get_device_id(device), str(dtype))
    out = _window_grid_kernel2d_cache.get(cache_key)
    if out is not None:
        return out
    # Direct grid and normalization in correct dtype/device
    window_grid2d = _fast_create_meshgrid(h, w, False, device, dtype)
    window_grid2d = _fast_normalize_pixel_coordinates(window_grid2d, h, w)
    # Equivalent to permute(3, 0, 1, 2)
    conv_kernel = window_grid2d.permute(3, 0, 1, 2).contiguous()
    _window_grid_kernel2d_cache[cache_key] = conv_kernel
    return conv_kernel


def _fast_get_center_kernel2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Optimized and cached version
    cache_key = (h, w, _get_device_id(device), str(dtype))
    out = _center_kernel2d_cache.get(cache_key)
    if out is not None:
        return out
    # Allocate directly in correct dtype/device
    center_kernel = torch.zeros(2, 2, h, w, device=device, dtype=dtype)
    # Center patch (same logic)
    if h % 2 != 0:
        h_i1 = h // 2
        h_i2 = (h // 2) + 1
    else:
        h_i1 = (h // 2) - 1
        h_i2 = (h // 2) + 1
    if w % 2 != 0:
        w_i1 = w // 2
        w_i2 = (w // 2) + 1
    else:
        w_i1 = (w // 2) - 1
        w_i2 = (w // 2) + 1
    val = 1.0 / float((h_i2 - h_i1) * (w_i2 - w_i1))
    center_kernel[(0, 1), (0, 1), h_i1:h_i2, w_i1:w_i2] = val
    _center_kernel2d_cache[cache_key] = center_kernel
    return center_kernel


def _fast_create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    # Memoized version with fast construction & cache
    cache_key = (height, width, int(normalized_coordinates), str(device), str(dtype))
    out = _meshgrid_cache.get(cache_key)
    if out is not None:
        return out

    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        if width > 1:
            xs = (xs / (width - 1) - 0.5) * 2
        else:
            xs.fill_(0)
        if height > 1:
            ys = (ys / (height - 1) - 0.5) * 2
        else:
            ys.fill_(0)
    base_grid = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)
    grid = base_grid.permute(1, 0, 2).unsqueeze(0).contiguous()
    _meshgrid_cache[cache_key] = grid
    return grid


def _fast_normalize_pixel_coordinates(pixel_coordinates: Tensor, height: int, width: int, eps: float = 1e-8) -> Tensor:
    # Optimized: batch creation of normalization factor
    if pixel_coordinates.shape[-1] != 2:
        raise ValueError(f"Input pixel_coordinates must be of shape (*, 2). Got {pixel_coordinates.shape}")

    # Direct torch.tensor without additional function call
    hw = torch.tensor([width, height], device=pixel_coordinates.device, dtype=pixel_coordinates.dtype)
    factor = 2.0 / (hw - 1).clamp(min=eps)
    return factor * pixel_coordinates - 1
