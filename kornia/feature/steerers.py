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

import numpy as np
import torch

from kornia.core import Module, Tensor


class DiscreteSteerer(Module):
    """Module for discrete rotation steerers.

    A steerer rotates keypoint descriptions in latent space as if they were obtained from rotated images.

    Args:
        generator: [N, N] tensor where N is the descriptor dimension.

    Example:
        >>> desc = torch.randn(512, 128)
        >>> generator = torch.randn(128, 128)
        >>> steerer = DiscreteSteerer(generator)
        >>> # steer 3 times:
        >>> steered_desc = steerer.steer_descriptions(desc, steerer_power=3, normalize=True)

    """

    def __init__(self, generator: Tensor) -> None:
        super().__init__()
        self.generator = torch.nn.Parameter(generator)

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.generator)

    def steer_descriptions(
        self,
        descriptions: Tensor,
        steerer_power: int = 1,
        normalize: bool = False,
    ) -> Tensor:
        for _ in range(steerer_power):
            descriptions = self.forward(descriptions)
        if normalize:
            descriptions = torch.nn.functional.normalize(descriptions, dim=-1)
        return descriptions

    @classmethod
    def create_dedode_default(
        cls,
        generator_type: str = "C4",
        steerer_order: int = 8,
    ) -> Module:
        """Create a steerer for pretrained DeDoDe descriptors int the "C-setting"
            from the paper https://arxiv.org/abs/2312.02152, where descriptors were
            trained for fixed steerers.

        Args:
            generator_type: The type of steerer generator.
                One of 'C4', 'SO2', default is 'C4'.
                These can be used with the DeDoDe descriptors in Kornia
                with C4 or SO2 in the name respectively (so called C-setting steerers).
            steerer_order: The discretisation order for SO2-steerers (NOT used for C4-steerers).

        Returns:
            The pretrained model.

        """
        descriptor_dim = 256
        if generator_type == "C4":
            # Optimized: Create single 4x4 permutation block, repeat it, and block-diag in one fast op
            # 4x4 cyclic permutation (same as before)
            base_block = torch.tensor(
                [[0.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]],
                dtype=torch.float32,
            )
            num_blocks = descriptor_dim // 4
            generator = torch.block_diag(*([base_block] * num_blocks))
            return cls(generator).eval()
        elif generator_type == "SO2":
            # Optimized: Prebuild the blocks as a contiguous tensor
            block_count = descriptor_dim // 14
            zeros_dim = descriptor_dim - 12 * block_count
            lie_blocks = []
            if zeros_dim > 0:
                lie_blocks.append(torch.zeros((zeros_dim, zeros_dim), dtype=torch.float32))
            # Use numpy for fast tiny matrix creation and stacking
            for j in range(1, 7):
                block = np.array([[0.0, j], [-j, 0]], dtype=np.float32)
                lie_blocks += [torch.from_numpy(block)] * block_count  # Each block_count
            # Optimized block_diag pass: only call once
            lie_generator = torch.block_diag(*lie_blocks)
            generator = torch.matrix_exp((2 * 3.14159 / steerer_order) * lie_generator)
            return cls(generator).eval()
        else:
            raise ValueError
