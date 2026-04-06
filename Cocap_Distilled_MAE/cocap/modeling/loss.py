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

class PhaseAwareLoss(LossBase):
    def __init__(
        self, 
        label_smoothing: float = 0.1, 
        target_vocab_size: int = 50257, # Make sure this matches your GPT-2 vocab size!
        ignore_index: int = -100,       
        lambda_mse: float = 1.0,        
        lambda_cos: float = 1.0,        
        lambda_reg: float = 0.1         
    ):
        super().__init__()
        
        # We reuse your excellent LabelSmoothingLoss for the text generation
        self.ce_loss_fn = LabelSmoothingLoss(label_smoothing, target_vocab_size, ignore_index)
        
        # Explicit control weights
        self.lambda_mse = lambda_mse
        self.lambda_cos = lambda_cos
        self.lambda_reg = lambda_reg

    def forward(self, target, output, phase: int) -> Tensor:
        # PHASE 1: Distillation (Visual Only)
        if phase == 1:
            z_pred = output["teacher_predicted"]
            z_target = output["teacher_target"]
            
            loss_mse = F.mse_loss(z_pred, z_target, reduction="mean")
            loss_cos = (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()
            
            return (self.lambda_mse * loss_mse) + (self.lambda_cos * loss_cos)

        # PHASE 2: Alignment (Text Only)
        elif phase == 2:
            return self.ce_loss_fn(target, output)

        # PHASE 3: SFT (Text + Visual Regularization)
        elif phase == 3:
            loss_ce = self.ce_loss_fn(target, output)
            
            z_pred = output["teacher_predicted"]
            z_target = output["teacher_target"]
            
            loss_mse = F.mse_loss(z_pred, z_target, reduction="mean")
            loss_cos = (1.0 - F.cosine_similarity(z_pred, z_target, dim=-1)).mean()
            
            loss_distill = (self.lambda_mse * loss_mse) + (self.lambda_cos * loss_cos)
            
            # CE is the boss, distillation is just a 10% regularizer (lambda_reg)
            return loss_ce + (self.lambda_reg * loss_distill)
        
        else:
            raise ValueError(f"Unknown training phase: {phase}")

# Register the new loss for Hydra
phase_aware_loss_cfg = builds(PhaseAwareLoss, populate_full_signature=True)
label_smoothing_loss_cfg = builds(LabelSmoothingLoss, populate_full_signature=True)