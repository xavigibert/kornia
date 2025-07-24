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

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR
from kornia.losses._utils import mask_ignore_pixels
from kornia.utils.one_hot import one_hot

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
# https://github.com/Lightning-AI/metrics/blob/v0.11.3/src/torchmetrics/functional/classification/dice.py#L66-L207


def dice_loss(
    pred: Tensor,
    target: Tensor,
    average: str = "micro",
    eps: float = 1e-8,
    weight: Optional[Tensor] = None,
    ignore_index: Optional[int] = -100,
) -> Tensor:
    """Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X \\cap Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        pred: logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        target: labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C-1`.
        average:
            Reduction applied in multi-class scenario:
            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Calculate the loss for each class separately and average the metrics across classes.
        eps: Scalar to enforce numerical stabiliy.
        weight: weights for classes with shape :math:`(num\\_of\\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Return:
        One-element tensor of the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(pred, target)
        >>> output.backward()

    """
    KORNIA_CHECK_IS_TENSOR(pred)
    if pred.ndim != 4:
        raise ValueError(f"Invalid pred shape, we expect BxNxHxW. Got: {pred.shape}")
    if pred.shape[-2:] != target.shape[-2:]:
        raise ValueError(f"pred and target shapes must be the same. Got: {pred.shape} and {target.shape}")
    if pred.device != target.device:
        raise ValueError(f"pred and target must be in the same device. Got: {pred.device} and {target.device}")
    num_of_classes = pred.shape[1]
    possible_average = {"micro", "macro"}
    KORNIA_CHECK(average in possible_average, f"The `average` has to be one of {possible_average}. Got: {average}")

    # Use in-place or native functional softmax for increased speed
    pred_soft = F.softmax(pred, dim=1)

    # Optimized mask_ignore_pixels: Only go further if ignore_index will actually cause entries to be ignored.
    target, target_mask = mask_ignore_pixels(target, ignore_index)

    # Efficient one_hot computation
    target_one_hot: Tensor = one_hot(target, num_classes=num_of_classes, device=pred.device, dtype=pred.dtype)

    # Only mask if really needed
    if target_mask is not None:
        # Avoid in-place since .unsqueeze_ may have no storage on bool tensors (depends on PyTorch)
        target_mask = target_mask.unsqueeze(1)
        target_one_hot = target_one_hot * target_mask
        pred_soft = pred_soft * target_mask

    # Validate or set class weights efficiently
    if weight is not None:
        KORNIA_CHECK_IS_TENSOR(weight, "weight must be Tensor or None.")
        KORNIA_CHECK(
            weight.shape[0] == num_of_classes and weight.numel() == num_of_classes,
            f"weight shape must be (num_of_classes,): ({num_of_classes},), got {weight.shape}",
        )
        KORNIA_CHECK(
            weight.device == pred.device,
            f"weight and pred must be in the same device. Got: {weight.device} and {pred.device}",
        )
        use_weight = True
        weight_ = weight
    else:
        use_weight = False
        weight_ = pred.new_ones(num_of_classes)

    # Set reduction dimensions
    dims: tuple[int, ...] = (2, 3)
    if average == "micro":
        dims = (1, 2, 3)
        # Pre-multiply to save one expansion op for each
        pred_soft = pred_soft * weight_.view(-1, 1, 1) if use_weight else pred_soft
        target_one_hot = target_one_hot * weight_.view(-1, 1, 1) if use_weight else target_one_hot

    intersection = torch.sum(pred_soft * target_one_hot, dims)
    cardinality = torch.sum(pred_soft + target_one_hot, dims)
    dice_score = 2.0 * intersection / (cardinality + eps)
    dice_loss_tensor = -dice_score + 1.0
    # reduce the loss across samples (and classes in case of `macro` averaging)
    if average == "macro":
        dice_loss_tensor = (dice_loss_tensor * weight_).sum(-1) / weight_.sum()
    return dice_loss_tensor.mean()


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Args:
        average:
            Reduction applied in multi-class scenario:
            - ``'micro'`` [default]: Calculate the loss across all classes.
            - ``'macro'``: Calculate the loss for each class separately and average the metrics across classes.
        eps: Scalar to enforce numerical stabiliy.
        weight: weights for classes with shape :math:`(num\_of\_classes,)`.
        ignore_index: labels with this value are ignored in the loss computation.

    Shape:
        - Pred: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C-1`.

    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(pred, target)
        >>> output.backward()

    """

    def __init__(
        self,
        average: str = "micro",
        eps: float = 1e-8,
        weight: Optional[Tensor] = None,
        ignore_index: Optional[int] = -100,
    ) -> None:
        super().__init__()
        self.average = average
        self.eps = eps
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return dice_loss(pred, target, self.average, self.eps, self.weight, self.ignore_index)
