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

import math
from typing import Optional

import torch
from torch import nn

from kornia.core import Tensor
from kornia.filters import filter2d


def distance_transform(image: torch.Tensor, kernel_size: int = 3, h: float = 0.35) -> torch.Tensor:
    """Approximates the Manhattan distance transform of images using cascaded convolution operations.

    The value at each pixel in the output represents the distance to the nearest non-zero pixel in the image image.
    It uses the method described in :cite:`pham2021dtlayer`.
    The transformation is applied independently across the channel dimension of the images.

    Args:
        image: Image with shape :math:`(B,C,H,W)`.
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    Returns:
        tensor with shape :math:`(B,C,H,W)`.

    Example:
        >>> tensor = torch.zeros(1, 1, 5, 5)
        >>> tensor[:,:, 1, 2] = 1
        >>> dt = kornia.contrib.distance_transform(tensor)

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")
    if not len(image.shape) == 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    H, W = image.shape[2:4]
    k2 = kernel_size // 2
    n_iters: int = math.ceil(max(H, W) / k2)

    # Accelerated meshgrid construction and kernel math
    grid = fast_create_meshgrid(
        kernel_size, kernel_size, normalized_coordinates=False, device=image.device, dtype=image.dtype
    )
    # grid[0, :, :, 0] is x, grid[0, :, :, 1] is y
    # Instead of grid -= k2 (broadcasts), use slice-index arithmetic:
    kernel_y = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - k2
    kernel_x = torch.arange(kernel_size, device=image.device, dtype=image.dtype) - k2
    gy, gx = torch.meshgrid(kernel_y, kernel_x, indexing="ij")
    kernel = torch.hypot(gx, gy)
    kernel = torch.exp(kernel / -h).unsqueeze(0)

    out = torch.zeros_like(image)
    boundary = image.clone()
    signal_ones = torch.ones_like(boundary)

    # Vectorized mask and adjustment inside hot loop
    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)
        cdt = torch.nan_to_num(cdt, posinf=0.0)

        # Use float mask to avoid unnecessary .where (torch.gt is already fast)
        mask = cdt > 0
        mask_sum = mask.sum().item()
        if mask_sum == 0:
            break

        offset: int = i * k2
        # Efficient masked add/mul
        out = out + ((offset + cdt) * mask)
        # Only update boundary in-place if mask is not empty
        boundary = torch.where(mask, signal_ones, boundary)

    return out


# FAST CREATE MESHGRID: avoid unnecessary stack and permute by manual broadcasting
def fast_create_meshgrid(
    height: int,
    width: int,
    normalized_coordinates: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    if normalized_coordinates:
        if width > 1:
            xs = (xs / (width - 1) - 0.5) * 2
        else:
            xs = torch.zeros_like(xs)
        if height > 1:
            ys = (ys / (height - 1) - 0.5) * 2
        else:
            ys = torch.zeros_like(ys)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
    return grid


class DistanceTransform(nn.Module):
    r"""Module that approximates the Manhattan (city block) distance transform of images using convolutions.

    Args:
        kernel_size: size of the convolution kernel.
        h: value that influence the approximation of the min function.

    """

    def __init__(self, kernel_size: int = 3, h: float = 0.35) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.h = h

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # If images have multiple channels, view the channels in the batch dimension to match kernel shape.
        if image.shape[1] > 1:
            image_in = image.view(-1, 1, image.shape[-2], image.shape[-1])
        else:
            image_in = image

        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)
