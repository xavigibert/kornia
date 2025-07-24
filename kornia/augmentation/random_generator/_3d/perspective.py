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
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor, as_tensor, tensor
from kornia.utils.helpers import _extract_device_dtype


class PerspectiveGenerator3D(RandomGeneratorBase):
    r"""Get parameters for ``perspective`` for a random perspective transform.

    Args:
        distortion_scale: controls the degree of distortion and ranges from 0 to 1.

    Returns:
        A dict of parameters to be passed for transformation.
            - src (Tensor): perspective source bounding boxes with a shape of (B, 8, 3).
            - dst (Tensor): perspective target bounding boxes with a shape (B, 8, 3).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, distortion_scale: Union[Tensor, float] = 0.5) -> None:
        super().__init__()
        self.distortion_scale = distortion_scale

    def __repr__(self) -> str:
        repr = f"distortion_scale={self.distortion_scale}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self._distortion_scale = as_tensor(self.distortion_scale, device=device, dtype=dtype)
        if not (self._distortion_scale.dim() == 0 and 0 <= self._distortion_scale <= 1):
            raise AssertionError(f"'distortion_scale' must be a scalar within [0, 1]. Got {self._distortion_scale}")
        self.rand_sampler = Uniform(
            tensor(0, device=device, dtype=dtype), tensor(1, device=device, dtype=dtype), validate_args=False
        )

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, Tensor]:
        batch_size = batch_shape[0]
        depth = batch_shape[-3]
        height = batch_shape[-2]
        width = batch_shape[-1]

        _common_param_check(batch_size, same_on_batch)
        _device, _dtype = _extract_device_dtype([self.distortion_scale])

        # Precompute start_points as numpy array for less time constructing the tensor
        # This avoids running several python ops (width-1, etc.) for every batch
        # All points should be float for torch
        pts = [
            [0.0, 0.0, 0.0],
            [width - 1, 0.0, 0.0],
            [width - 1, height - 1, 0.0],
            [0.0, height - 1, 0.0],
            [0.0, 0.0, depth - 1],
            [width - 1, 0.0, depth - 1],
            [width - 1, height - 1, depth - 1],
            [0.0, height - 1, depth - 1],
        ]
        # Only do single construction then expand
        start_points_ = torch.tensor([pts], device=_device, dtype=_dtype).expand(batch_size, -1, -1)

        # generate random offset not larger than half of the image
        # Use locals so Python doesn't look up fields every time
        dscale = self.distortion_scale
        dtype = _dtype
        device = _device
        # Use torch.as_tensor for optimal broadcast, and single torch ops to minimize python overhead
        fx = dscale * (width / 2)
        fy = dscale * (height / 2)
        fz = dscale * (depth / 2)
        factor = torch.as_tensor([fx, fy, fz], dtype=dtype, device=device).view(1, 1, 3)
        # Instead of stack-view-to, we can use as_tensor directly and broadcast with the next op

        # Save allocation/transfer: do adapted_rsampling directly on the device and dtype
        rand_shape = start_points_.shape
        # If random sampler can take device/dtype we can set in distribution directly; if not, keep this
        rand_val = _adapted_rsampling(rand_shape, self.rand_sampler, same_on_batch)
        if rand_val.device != device or rand_val.dtype != dtype:
            rand_val = rand_val.to(device=device, dtype=dtype)

        # pts_norm is always the same: construct once, float for torch, and broadcast
        pts_norm_const = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                ]
            ],
            device=device,
            dtype=dtype,
        )
        # For full broadcast, we expand to shape (batch_size, 8, 3)
        pts_norm = pts_norm_const.expand(batch_size, -1, -1)

        # Optimize math: elementwise faster, broadcast compatible
        # (batch, 8, 3) + (1, 1, 3) * (batch, 8, 3) * (batch, 8, 3)
        # Use inplace add for memory and speed, and fused multiply
        end_points = start_points_ + factor * rand_val * pts_norm

        return {"start_points": start_points_, "end_points": end_points}
