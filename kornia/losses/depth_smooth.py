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

import torch
from torch import nn

# Based on
# https://github.com/tensorflow/models/blob/master/research/struct2depth/model.py#L625-L641


def _gradient_x(img: torch.Tensor) -> torch.Tensor:
    # Assumes input is 4D tensor: (N, C, H, W)
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def _gradient_y(img: torch.Tensor) -> torch.Tensor:
    # Assumes input is 4D tensor: (N, C, H, W)
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def inverse_depth_smoothness_loss(idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \\left | \\partial_x d_{ij} \right | e^{-\\left \\|
        \\partial_x I_{ij} \right \\|} + \\left |
        \\partial_y d_{ij} \right | e^{-\\left \\| \\partial_y I_{ij} \right \\|}

    Args:
        idepth: tensor with the inverse depth with shape :math:`(N, 1, H, W)`.
        image: tensor with the input image with shape :math:`(N, 3, H, W)`.

    Return:
        a scalar with the computed loss.

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> loss = inverse_depth_smoothness_loss(idepth, image)

    """
    _check_input_shapes(idepth, image)

    # Unroll all in parallel to minimize temporaries and CUDA overhead
    idepth_dx = _gradient_x(idepth)
    idepth_dy = _gradient_y(idepth)
    image_dx = _gradient_x(image)
    image_dy = _gradient_y(image)

    # Combine mean/abs/exp with minimal temporary allocation
    # weights_x, weights_y: shape (N,1,H,W-1) and (N,1,H-1,W)
    weights_x = torch.exp(-torch.mean(image_dx.abs(), dim=1, keepdim=True))
    weights_y = torch.exp(-torch.mean(image_dy.abs(), dim=1, keepdim=True))

    # Fused abs and product to reduce intermediate memory
    smoothness_x = (idepth_dx * weights_x).abs()
    smoothness_y = (idepth_dy * weights_y).abs()

    # torch.mean is already efficient here since last operation
    return torch.mean(smoothness_x) + torch.mean(smoothness_y)


def _check_input_shapes(idepth: torch.Tensor, image: torch.Tensor):
    # Fast input shape validation
    if not (isinstance(idepth, torch.Tensor) and isinstance(image, torch.Tensor)):
        raise TypeError(f"Input idepth and image must be torch.Tensor, got {type(idepth)} {type(image)}")
    if idepth.ndim != 4 or image.ndim != 4:
        raise ValueError(f"idepth and image must be 4D tensors. Got {idepth.shape} and {image.shape}")
    if idepth.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"idepth and image sizes must match spatially. Got: {idepth.shape} vs {image.shape}")
    if idepth.device != image.device:
        raise ValueError(f"idepth and image must be on same device. Got: {idepth.device} vs {image.device}")
    if idepth.dtype != image.dtype:
        raise ValueError(f"idepth and image must be same dtype. Got: {idepth.dtype} vs {image.dtype}")


class InverseDepthSmoothnessLoss(nn.Module):
    r"""Criterion that computes image-aware inverse depth smoothness loss.

    .. math::

        \text{loss} = \left | \partial_x d_{ij} \right | e^{-\left \|
        \partial_x I_{ij} \right \|} + \left |
        \partial_y d_{ij} \right | e^{-\left \| \partial_y I_{ij} \right \|}

    Shape:
        - Inverse Depth: :math:`(N, 1, H, W)`
        - Image: :math:`(N, 3, H, W)`
        - Output: scalar

    Examples:
        >>> idepth = torch.rand(1, 1, 4, 5)
        >>> image = torch.rand(1, 3, 4, 5)
        >>> smooth = InverseDepthSmoothnessLoss()
        >>> loss = smooth(idepth, image)

    """

    def forward(self, idepth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return inverse_depth_smoothness_loss(idepth, image)
