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

from typing import Any, Dict, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from kornia.core import Module, Tensor

INF = 1e9


def mask_border(m: Tensor, b: int, v: Union[Tensor, float, bool]) -> None:
    """Mask borders with value.

    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int): border size.
        v (m.dtype): border value.
    """
    if b <= 0:
        return

    # Vectorize by creating slices once, operating at all batch at once.
    s = slice(None)
    _b = b if isinstance(b, int) else int(b)

    # Use direct slice assignment for all borders
    m[:, :_b, :, :, :] = v
    m[:, -_b:, :, :, :] = v
    m[:, :, :_b, :, :] = v
    m[:, :, -_b:, :, :] = v
    m[:, :, :, :_b, :] = v
    m[:, :, :, -_b:, :] = v
    m[:, :, :, :, :_b] = v
    m[:, :, :, :, -_b:] = v


def mask_border_with_padding(m: Tensor, bd: int, v: Union[Tensor, float, bool], p_m0: Tensor, p_m1: Tensor) -> None:
    """Apply masking to a padded boarder."""
    if bd <= 0:
        return

    s = slice(None)
    _bd = bd if isinstance(bd, int) else int(bd)
    m[:, :_bd, :, :, :] = v
    m[:, :, :_bd, :, :] = v
    m[:, :, :, :_bd, :] = v
    m[:, :, :, :, :_bd] = v

    # Move as much as possible out of Python loop for efficiency
    h0s = p_m0.sum(1).max(-1)[0].int()
    w0s = p_m0.sum(-1).max(-1)[0].int()
    h1s = p_m1.sum(1).max(-1)[0].int()
    w1s = p_m1.sum(-1).max(-1)[0].int()

    # Vectorize masking for all batches -- fully vectorizable for last slices
    batch_indices = torch.arange(m.shape[0], device=m.device)

    # For all subsequent tails, use broadcasting for speed
    for idx, (h0, w0, h1, w1) in enumerate(zip(h0s.tolist(), w0s.tolist(), h1s.tolist(), w1s.tolist())):
        m[idx, h0 - _bd :, :, :, :] = v
        m[idx, :, w0 - _bd :, :, :] = v
        m[idx, :, :, h1 - _bd :, :] = v
        m[idx, :, :, :, w1 - _bd :] = v


def compute_max_candidates(p_m0: Tensor, p_m1: Tensor) -> Tensor:
    """Compute the max candidates of all pairs within a batch.

    Args:
        p_m0: padded mask 0
        p_m1: padded mask 1

    """
    h0s = p_m0.sum(1).max(-1)[0]
    w0s = p_m0.sum(-1).max(-1)[0]
    h1s = p_m1.sum(1).max(-1)[0]
    w1s = p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(h0s * w0s, h1s * w1s))
    return max_cand


class CoarseMatching(Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        self.train_coarse_percent = config["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train_pad_num_gt_min"]
        self.match_type = config["match_type"]
        if self.match_type == "dual_softmax":
            self.temperature = config["dsmax_temperature"]
        elif self.match_type == "sinkhorn":
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!") from None
            self.log_optimal_transport = log_optimal_transport
            self.bin_score = nn.Parameter(torch.tensor(config["skh_init_bin_score"], requires_grad=True))
            self.skh_iters = config["skh_iters"]
            self.skh_prefilter = config["skh_prefilter"]
        else:
            raise NotImplementedError

    def forward(
        self,
        feat_c0: Tensor,
        feat_c1: Tensor,
        data: Dict[str, Tensor],
        mask_c0: Optional[Tensor] = None,
        mask_c1: Optional[Tensor] = None,
    ) -> None:
        """Run forward.

        Args:
            feat_c0 (torch.Tensor): [N, L, C]
            feat_c1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.

        """
        _, L, S, _ = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize
        feat_c0, feat_c1 = (feat / feat.shape[-1] ** 0.5 for feat in [feat_c0, feat_c1])

        if self.match_type == "dual_softmax":
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
            if mask_c0 is not None and mask_c1 is not None:
                sim_matrix.masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == "sinkhorn":
            # sinkhorn, dustbin included
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None and mask_c1 is not None:
                sim_matrix[:, :L, :S].masked_fill_(~(mask_c0[..., None] * mask_c1[:, None]).bool(), -INF)

            # build uniform prior & use sinkhorn
            log_assign_matrix = self.log_optimal_transport(sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            if self.config["sparse_spvs"]:
                data.update({"conf_matrix_with_bin": assign_matrix.clone()})

        data.update({"conf_matrix": conf_matrix})

        # predict coarse matches from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix: Tensor, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Get corase matching.

        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']

        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}

        """
        axes_lengths = {
            "h0c": data["hw0_c"][0],
            "w0c": data["hw0_c"][1],
            "h1c": data["hw1_c"][0],
            "w1c": data["hw1_c"][1],
        }
        n = conf_matrix.shape[0]
        h0c = axes_lengths["h0c"]
        w0c = axes_lengths["w0c"]
        h1c = axes_lengths["h1c"]
        w1c = axes_lengths["w1c"]
        device = conf_matrix.device

        mask = (conf_matrix > self.thr).reshape(n, h0c, w0c, h1c, w1c)

        # Mask borders
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data["mask0"], data["mask1"])

        mask = mask.reshape(n, h0c * w0c, h1c * w1c)

        # 2. Efficient mutual nearest mask; factor maxes to avoid redundant .max calls
        conf_matrix_max2 = conf_matrix.max(dim=2, keepdim=True)[0]
        conf_matrix_max1 = conf_matrix.max(dim=1, keepdim=True)[0]
        mask = mask & (conf_matrix == conf_matrix_max2) & (conf_matrix == conf_matrix_max1)

        # 3. Find all valid matches: take max j for each row
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # 4. Training-time random sampling (minimize branching, no listcomp needed)
        if self.training:
            if "mask0" not in data:
                num_candidates_max = n * max(mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(data["mask0"], data["mask1"])
            num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = b_ids.numel()
            if self.train_pad_num_gt_min >= num_matches_train:
                raise ValueError("min-num-gt-pad should be less than num-train-matches")
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred, (num_matches_train - self.train_pad_num_gt_min,), device=device
                )
            gt_pad_indices = torch.randint(
                len(data["spv_b_ids"]),
                (max(num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=device,
            )
            mconf_gt = torch.zeros(len(data["spv_b_ids"]), device=device)
            b_ids, i_ids, j_ids, mconf = (
                torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0)
                for x, y in zip(
                    [b_ids, data["spv_b_ids"]],
                    [i_ids, data["spv_i_ids"]],
                    [j_ids, data["spv_j_ids"]],
                    [mconf, mconf_gt],
                )
            )

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        scale = data["hw0_i"][0] / h0c
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale

        i_idx_mod = i_ids % w0c
        i_idx_div = i_ids // w0c
        j_idx_mod = j_ids % w1c
        j_idx_div = j_ids // w1c
        mkpts0_c = torch.stack([i_idx_mod, i_idx_div], dim=1) * scale0
        mkpts1_c = torch.stack([j_idx_mod, j_idx_div], dim=1) * scale1

        mask_pred = mconf != 0
        coarse_matches.update(
            {
                "gt_mask": mconf == 0,
                "m_bids": b_ids[mask_pred],
                "mkpts0_c": mkpts0_c[mask_pred].to(dtype=conf_matrix.dtype),
                "mkpts1_c": mkpts1_c[mask_pred].to(dtype=conf_matrix.dtype),
                "mconf": mconf[mask_pred],
            }
        )
        return coarse_matches
