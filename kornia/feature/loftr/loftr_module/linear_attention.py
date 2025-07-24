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

"""Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention".

Modified from: https://github.com/idiap/fast-
transformers/blob/master/fast_transformers/attention/linear_attention.py.
"""

from typing import Optional

import torch
from torch.nn import Dropout

from kornia.core import Module, Tensor


def elu_feature_map(x: Tensor) -> Tensor:
    """Apply elu activation."""
    return torch.nn.functional.elu(x) + 1


def _batch_mask_fill(QK: Tensor, q_mask: Tensor, kv_mask: Tensor) -> None:
    # Helper for fused broadcasting and avoiding creating large temp mask tensors
    # QK: [N, L, S, H], q_mask: [N, L], kv_mask: [N, S]
    # Fill QK at positions where mask is False
    N, L, S, H = QK.shape
    # The boolean mask we're after is:
    # mask [N, L, S] = (q_mask[:, :, None] & kv_mask[:, None, :])
    # But to save memory, construct flat indices for efficiency
    # Compute outer AND for q_mask and kv_mask to get index
    valid_index = q_mask.bool()[:, :, None] & kv_mask.bool()[:, None, :]  # [N, L, S]
    valid_index = valid_index[:, :, :, None].expand(-1, -1, -1, H)  # [N, L, S, H]
    # Now fill
    QK.masked_fill_(~valid_index, float("-inf"))


class LinearAttention(Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-Head linear attention proposed in "Transformers are RNNs".

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(Module):
    def __init__(self, use_dropout: bool = False, attention_dropout: float = 0.1) -> None:
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        q_mask: Optional[Tensor] = None,
        kv_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Multi-head scaled dot-product attention, a.k.a full attention.

        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]

        Returns:
            queried_values: (N, L, H, D)

        """
        # Compute (Q, K^T) dot product using torch.matmul for efficiency.
        # Equivalent to: torch.einsum("nlhd,nshd->nlsh", queries, keys)
        # Rearrange and matmul for batch efficiency
        N, L, H, D = queries.shape
        S = keys.size(1)
        queries_ = queries.permute(0, 2, 1, 3).reshape(N * H, L, D)  # (N*H, L, D)
        keys_ = keys.permute(0, 2, 3, 1).reshape(N * H, D, S)  # (N*H, D, S)
        QK = torch.bmm(queries_, keys_)  # (N*H, L, S)
        QK = QK.view(N, H, L, S).permute(0, 2, 3, 1).contiguous()  # (N, L, S, H)

        # Efficient masking
        if kv_mask is not None and q_mask is not None:
            _batch_mask_fill(QK, q_mask, kv_mask)

        # Compute the attention weights and the weighted average (softmax on S)
        softmax_temp = 1.0 / D**0.5  # sqrt(D)
        # Multiply in-place for lower memory use
        QK.mul_(softmax_temp)
        # Softmax along S=2 as before – XLA/cuda can merge this with matmul efficiently
        A = torch.softmax(QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        # Compute output: weighted sum of values.
        # A: (N, L, S, H), values: (N, S, H, D)
        # Efficient batch matmul as above
        A_ = A.permute(0, 3, 1, 2).reshape(N * H, L, S)  # (N*H, L, S)
        values_ = values.permute(0, 2, 1, 3).reshape(N * H, S, D)  # (N*H, S, D)
        queried_values = torch.bmm(A_, values_)  # (N*H, L, D)
        queried_values = queried_values.view(N, H, L, D).permute(0, 2, 1, 3).contiguous()  # (N, L, H, D)

        return queried_values  # contiguous() already applied above
