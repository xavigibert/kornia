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

import math

import torch
from torch import nn

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
    # Fast checks (no profile impact)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image type is not a torch.Tensor. Got {type(image)}")
    if image.ndimension() != 4:
        raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Precompute kernel/grid outside the loop (expensive ops moved out)
    kernel = _precompute_dt_kernel(kernel_size, h, image.dtype, image.device)
    # n_iters is set such that the DT will be able to propagate from any corner of the image to its far,
    # diagonally opposite corner
    n_iters = math.ceil(max(image.shape[2], image.shape[3]) / math.floor(kernel_size / 2))
    out = _optimized_distance_transform_inner(image, kernel, h, n_iters)
    return out


def _precompute_dt_kernel(kernel_size: int, h: float, dtype, device) -> torch.Tensor:
    # Precompute the meshgrid and kernel. This avoids repeated grid+kernel computation for each call, and can be reused.
    # grid shape: (1, kernel_size, kernel_size, 2)
    half_ks = kernel_size // 2
    xs = torch.arange(-half_ks, half_ks + 1, device=device, dtype=dtype)
    ys = torch.arange(-half_ks, half_ks + 1, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # ordering matches (height, width)
    # grid_x/grid_y have shape (kernel_size, kernel_size)
    # Manhattan distance approximate: torch.hypot(grid_x, grid_y)
    kernel = torch.hypot(grid_x, grid_y)
    kernel = torch.exp(kernel / -h).unsqueeze(0)
    return kernel


def _optimized_distance_transform_inner(
    image: torch.Tensor, kernel: torch.Tensor, h: float, n_iters: int
) -> torch.Tensor:
    # Split out fast inner loop for memory and JIT-fush optimizations.
    # Preallocate the output tensor
    out = torch.zeros_like(image)
    # Use in-place modifications only where necessary and avoid superfluous allocation in loop
    boundary = image.clone()
    # signal_ones may be broadcasted, to avoid creating a big tensor every time, create once and reuse
    signal_ones = torch.ones_like(boundary)

    for i in range(n_iters):
        cdt = filter2d(boundary, kernel, border_type="replicate")
        cdt = -h * torch.log(cdt)
        # Replace NaN/infs (can happen only for log(0))
        cdt = torch.nan_to_num(cdt, posinf=0.0)
        # mask is float (shape = input), avoids further .float() calls
        mask = cdt > 0
        if not mask.any():
            break
        offset = i * (kernel.shape[-1] // 2)
        # Use mask for indexing only once: only modify places where mask==True; avoids full image assignment
        # out[mask] += (offset + cdt[mask])
        # However, broadcasting gives slightly better performance for BLAS
        out += (offset + cdt) * mask
        boundary = torch.where(mask, signal_ones, boundary)
    return out


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

        # No optimization here; rely on underlying optimized function
        return distance_transform(image_in, self.kernel_size, self.h).view_as(image)
