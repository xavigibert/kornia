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

from typing import Dict, Tuple, Union

import torch
from torch.distributions import Uniform

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling
from kornia.core import Tensor, tensor

__all__ = ["PerspectiveGenerator"]


class PerspectiveGenerator(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: the degree of distortion, ranged from 0 to 1.
        sampling_method: ``'basic'`` | ``'area_preserving'``. Default: ``'basic'``
            If ``'basic'``, samples by translating the image corners randomly inwards.
            If ``'area_preserving'``, samples by randomly translating the image corners in any direction.
            Preserves area on average. See https://arxiv.org/abs/2104.03308 for further details.

    Returns:
        A dict of parameters to be passed for transformation.
            - start_points (Tensor): element-wise perspective source areas with a shape of (B, 4, 2).
            - end_points (Tensor): element-wise perspective target areas with a shape of (B, 4, 2).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, distortion_scale: Union[Tensor, float] = 0.5, sampling_method: str = "basic") -> None:
        super().__init__()
        if sampling_method not in ("basic", "area_preserving"):
            raise NotImplementedError(f"Sampling method {sampling_method} not yet implemented.")
        self.distortion_scale = distortion_scale
        self.sampling_method = sampling_method

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = torch.as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}.")
        self.rand_val_sampler = Uniform(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        # Fast-path: use local variables and minimize device/dtype extraction
        batch_size = batch_shape[0]
        height = batch_shape[-2]
        width = batch_shape[-1]

        # Early parameter checking
        if not (isinstance(batch_size, int) and batch_size >= 0):
            raise AssertionError(f"`batch_size` shall be a positive integer. Got {batch_size}.")
        if same_on_batch is not None and not isinstance(same_on_batch, bool):
            raise AssertionError(f"`same_on_batch` shall be boolean. Got {same_on_batch}.")
        if not (isinstance(height, int) and height > 0 and isinstance(width, int) and width > 0):
            raise AssertionError(f"'height' and 'width' must be integers. Got {height}, {width}.")

        # Use fast path for device/dtype extraction
        d_scale = self.distortion_scale
        if isinstance(d_scale, Tensor):
            device = d_scale.device
            dtype = d_scale.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.get_default_dtype()
        _device = device
        _dtype = dtype

        base_corners = torch.tensor(
            [[0.0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            device=_device,
            dtype=_dtype,
        )
        start_points = base_corners.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute fx, fy directly without extra indirection
        _dscale = (
            d_scale
            if isinstance(d_scale, float)
            else d_scale.item()
            if isinstance(d_scale, Tensor) and d_scale.numel() == 1
            else d_scale
        )
        fx = _dscale * width / 2
        fy = _dscale * height / 2

        # Construct factor tensor more efficiently
        factor = torch.tensor([[fx, fy]], device=_device, dtype=_dtype)  # shape (1,2)
        factor = factor.unsqueeze(1)  # shape (1,1,2), broadcastable to (B,4,2)

        # Generate random values once, already on target device/dtype
        rand_shape = (batch_size, 4, 2)
        rand_val = _adapted_rsampling(rand_shape, self.rand_val_sampler, same_on_batch)
        if rand_val.device != _device or rand_val.dtype != _dtype:
            rand_val = rand_val.to(device=_device, dtype=_dtype)

        if self.sampling_method == "basic":
            # Use constant pts_norm tensor, allocate once
            pts_norm = torch.tensor(
                [[[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]], device=_device, dtype=_dtype
            )
            offset = factor * rand_val * pts_norm
        elif self.sampling_method == "area_preserving":
            offset = 2 * factor * (rand_val - 0.5)
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} not yet implemented.")

        end_points = start_points + offset

        return {"start_points": start_points, "end_points": end_points}
