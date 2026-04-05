# -*- coding: utf-8 -*-
# @Time    : 7/19/23
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : loss.py

__all__ = [
    "LossBase",
    "LabelSmoothingLoss",
    "label_smoothing_loss_cfg"
]

import logging
from abc import abstractmethod

import torch
import torch.nn.functional as F
from hydra_zen import builds
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class LossBase(nn.Module):
    @abstractmethod
    def forward(self, inputs, outputs) -> Tensor:
        """Compute loss."""


class LabelSmoothingLoss(LossBase):
    def __init__(self, label_smoothing=0.1, target_vocab_size=49408, ignore_index=0):
        assert 0.0 < label_smoothing <= 1.0

        super().__init__()

        self.tgt_vocab_size = target_vocab_size
        self.ignore_index = ignore_index

        self.log_softmax = nn.LogSoftmax(dim=-1)

        smoothing_value = label_smoothing / (self.tgt_vocab_size - 1)  # count for the ground-truth word
        one_hot = torch.full((self.tgt_vocab_size,), smoothing_value)
        # one_hot[self.ignore_index] = 0
        self.register_buffer("one_hot", one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, target, output):
        output = output["prediction_scores"]
        output = output.view(-1, self.tgt_vocab_size)
        target = target['input_labels'].reshape(-1).long()
        valid_indices = target != self.ignore_index  # ignore examples with target value -1
        target = target[valid_indices]
        output = self.log_softmax(output[valid_indices])

        model_prob = self.one_hot.repeat(target.size(0), 1).to(target.device)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        return F.kl_div(output, model_prob, reduction="sum")

class MicroCapKendallLoss(LossBase):
    def __init__(self, label_smoothing=0.1, target_vocab_size=49408, ignore_index=0, clamp_val=1.6):
        super().__init__()
        
        # 1. The Captioning Loss (Cross Entropy with Label Smoothing)
        self.ce_loss_fn = LabelSmoothingLoss(label_smoothing, target_vocab_size, ignore_index)
        
        # 2. Kendall Learnable Parameters (Log-Variance for stability)
        self.clamp_val = clamp_val
        self.log_var_ce = nn.Parameter(torch.zeros(1))
        self.log_var_mse = nn.Parameter(torch.zeros(1))
        self.log_var_cos = nn.Parameter(torch.zeros(1))

    def forward(self, target, output) -> Tensor:
        # 1. Calculate standard Captioning CE Loss
        loss_ce = self.ce_loss_fn(target, output)
        
        # 2. If Teacher data is present (Training Mode), calculate distillation losses
        if "teacher_predicted" in output and "teacher_target" in output:
            z_pred = output["teacher_predicted"]
            z_target = output["teacher_target"]
            
            # L_mse: Spatial magnitude alignment
            loss_mse = F.mse_loss(z_pred, z_target, reduction="mean")
            
            # L_cos: Semantic directional alignment
            cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1)
            loss_cos = (1.0 - cos_sim).mean()
            
            # Clamp the sigmas to prevent degenerate collapse
            log_s_ce = self.log_var_ce.clamp(-self.clamp_val, self.clamp_val)
            log_s_mse = self.log_var_mse.clamp(-self.clamp_val, self.clamp_val)
            log_s_cos = self.log_var_cos.clamp(-self.clamp_val, self.clamp_val)
            
            # Convert to precision (1 / sigma^2)
            # Classification: 1 / sigma^2
            # Regression: 1 / (2 * sigma^2)
            w_ce = torch.exp(-2 * log_s_ce) * loss_ce + log_s_ce
            w_mse = 0.5 * torch.exp(-2 * log_s_mse) * loss_mse + log_s_mse
            w_cos = 0.5 * torch.exp(-2 * log_s_cos) * loss_cos + log_s_cos
            
            total_loss = w_ce + w_mse + w_cos
            
            # Optional: Print/Log the weights to see Kendall working in real-time
            if torch.rand(1).item() < 0.01: # print ~1% of the time
                logger.debug(f"Kendall Weights | CE: {torch.exp(-2 * log_s_ce).item():.4f} | MSE: {0.5 * torch.exp(-2 * log_s_mse).item():.4f} | COS: {0.5 * torch.exp(-2 * log_s_cos).item():.4f}")
                
            return total_loss
        else:
            # Inference / Validation fallback (No Teacher)
            return loss_ce

# Register it for Hydra configs
microcap_kendall_loss_cfg = builds(MicroCapKendallLoss, populate_full_signature=True)
# Build configs for organizing modules with hydra
label_smoothing_loss_cfg = builds(LabelSmoothingLoss, populate_full_signature=True)
