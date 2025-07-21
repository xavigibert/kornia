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

from typing import Dict, Optional, Tuple, Union

import torch
from torch.distributions import Bernoulli

from kornia.augmentation.random_generator.base import RandomGeneratorBase, UniformDistribution
from kornia.augmentation.utils import _joint_range_check
from kornia.core import Tensor

__all__ = ["MixupGenerator"]


class MixupGenerator(RandomGeneratorBase):
    r"""Generate mixup indexes and lambdas for a batch of inputs.

    Args:
        lambda_val (torch.Tensor, optional): min-max strength for mixup images, ranged from [0., 1.].
            If None, it will be set to tensor([0., 1.]), which means no restrictions.

    Returns:
        A dict of parameters to be passed for transformation.
            - mix_pairs (torch.Tensor): element-wise probabilities with a shape of (B,).
            - mixup_lambdas (torch.Tensor): element-wise probabilities with a shape of (B,).

    Note:
        The generated random numbers are not reproducible across different devices and dtypes. By default,
        the parameters will be generated on CPU in float32. This can be changed by calling
        ``self.set_rng_device_and_dtype(device="cuda", dtype=torch.float64)``.

    """

    def __init__(self, lambda_val: Optional[Union[torch.Tensor, Tuple[float, float]]] = None, p: float = 1.0) -> None:
        super().__init__()
        self.lambda_val = lambda_val
        self.p = p

    def __repr__(self) -> str:
        repr = f"lambda_val={self.lambda_val}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.lambda_val is None:
            lambda_val = torch.tensor([0.0, 1.0], device=device, dtype=dtype)
        else:
            lambda_val = torch.as_tensor(self.lambda_val, device=device, dtype=dtype)

        _joint_range_check(lambda_val, "lambda_val", bounds=(0, 1))
        self.lambda_sampler = UniformDistribution(lambda_val[0], lambda_val[1], validate_args=False)
        self.prob_sampler = Bernoulli(torch.tensor(float(self.p), device=device, dtype=dtype))

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        # Inline all external util calls for lower call overhead and faster dispatch.
        batch_size = batch_shape[0]

        # Fast param check - inlined version to avoid extra stack/call/lookup
        if not (isinstance(batch_size, int) and batch_size >= 0):
            raise AssertionError(f"`batch_size` shall be a positive integer. Got {batch_size}.")
        if same_on_batch is not None and not isinstance(same_on_batch, bool):
            raise AssertionError(f"`same_on_batch` shall be boolean. Got {same_on_batch}.")

        # Fast device, dtype extract
        lambda_val = self.lambda_val
        device, dtype = None, None
        if lambda_val is not None and isinstance(lambda_val, Tensor):
            device = lambda_val.device
            dtype = lambda_val.dtype

        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.get_default_dtype()

        # Sample batch_probs efficiently without function indirection
        with torch.no_grad():
            # _adapted_sampling inlined here for speed
            dist = self.prob_sampler
            shape = (batch_size,)
            if same_on_batch:
                batch_probs = dist.sample(torch.Size((1,))).repeat(batch_size)
            else:
                batch_probs = dist.sample(torch.Size(shape))

        # Use torch.randperm directly for better speed and less object conversion
        # Use previously determined device and dtype
        mixup_pairs = torch.randperm(batch_size, device=device).long()

        # Sample lambdas efficiently (using inlined _adapted_rsampling for speed) and avoid copying when possible
        dist = self.lambda_sampler
        if same_on_batch:
            rsample = dist.rsample(torch.Size((1,)))
            mixup_lambdas = rsample.repeat(batch_size)
        else:
            mixup_lambdas = dist.rsample(torch.Size((batch_size,)))
        mixup_lambdas = mixup_lambdas * batch_probs

        # Construct dict while minimizing tensor copying/casting
        # .to for dtype-safe output
        return {
            "mixup_pairs": mixup_pairs.to(device=device, dtype=torch.long),
            "mixup_lambdas": mixup_lambdas.to(device=device, dtype=dtype),
        }
