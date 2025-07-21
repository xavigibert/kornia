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

# The following is a fast LRU cache for (H, W, device, dtype) -> meshgrid tensor.
# For general usage, 64 is plenty; bump if running on a broad set of shapes.
_MESHGRID_CACHE: Dict[Tuple[int, int, torch.device, torch.dtype], Tensor] = {}


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
    key = (
        height,
        width,
        device if device is not None else torch.device("cpu"),
        dtype if dtype is not None else torch.get_default_dtype(),
        normalized_coordinates,
    )
    # Only cache for normalized_coordinates == False (most common for depth_to_3d), else use old way
    if not normalized_coordinates:
        cached = _MESHGRID_CACHE.get(key, None)
        if cached is not None:
            return cached
    xs = torch.linspace(0, width - 1, width, device=device, dtype=dtype)
    ys = torch.linspace(0, height - 1, height, device=device, dtype=dtype)
    if normalized_coordinates:
        xs = (xs / (width - 1) - 0.5) * 2
        ys = (ys / (height - 1) - 0.5) * 2
    base_grid = stack(torch_meshgrid([xs, ys], indexing="ij"), dim=-1)  # WxHx2
    out = base_grid.permute(1, 0, 2).unsqueeze(0)  # 1xHxWx2
    if not normalized_coordinates:
        if len(_MESHGRID_CACHE) > 64:
            _MESHGRID_CACHE.clear()
        _MESHGRID_CACHE[key] = out
    return out


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
