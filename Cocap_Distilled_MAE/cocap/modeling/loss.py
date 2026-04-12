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
        target_labels = target['input_labels']

        # Some model paths may include visual-prefix logits; keep only text-token logits.
        if output.dim() == 3 and target_labels.dim() == 2 and output.size(1) != target_labels.size(1):
            output = output[:, -target_labels.size(1):, :]

        output = output.reshape(-1, self.tgt_vocab_size)
        target = target_labels.reshape(-1).long()
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
        target_vocab_size: int = 50257, 
        ignore_index: int = -100,       
        l_mse_scale_factor: float = 5.0, # Brings MSE up to Cosine scale
        lambda_delta: float = 0.5,       # Phase 1 delta weight
        rho_min_tokens: float = 4.0,     # Phase 2 min tokens per GOP
        lambda_coverage: float = 0.5,    # Phase 2 coverage weight
        lambda_reg: float = 0.05         # Phase 3 distillation regularizer weight
    ):
        super().__init__()
        
        self.ce_loss_fn = LabelSmoothingLoss(label_smoothing, target_vocab_size, ignore_index)
        
        self.l_mse_scale = l_mse_scale_factor
        self.lambda_delta = lambda_delta
        self.rho_min_tokens = rho_min_tokens
        self.lambda_coverage = lambda_coverage
        self.lambda_reg = lambda_reg

        # Filled on each forward pass for logging only (no gradient usage).
        self.latest_loss_components = {}

    def forward(self, target, output, phase: int) -> Tensor:
        """
        Expects `output` dictionary from the model to contain specific tensors based on phase.
        G = number of GOPs (usually 8)
        """
        
        # ==========================================================
        # PHASE 1: Motion Distillation Pretraining (GOP-Wise)
        # ==========================================================
        if phase == 1:
            if "predicted_spatial" not in output and "teacher_predicted" in output and "teacher_target" in output:
                z_pred = output["teacher_predicted"].float()  # [B*G, 196, 768]
                z_tgt = output["teacher_target"]              # [B*G, 196, 768]

                if z_tgt is None:
                    zero = torch.zeros((), device=z_pred.device, dtype=z_pred.dtype)
                    self.latest_loss_components = {
                        "phase1/delta_mse_loss": zero.detach(),
                        "phase1/delta_cosine_loss": zero.detach(),
                        "phase1/regenerative_loss": zero.detach(),
                        "phase1/total_loss": zero.detach(),
                    }
                    return zero

                z_tgt = z_tgt.float()

                valid_mask = None
                if isinstance(target, dict):
                    v = target.get("video", None)
                    if isinstance(v, dict) and "input_mask_gop" in v:
                        gop_mask = v["input_mask_gop"]
                        if torch.is_tensor(gop_mask) and gop_mask.dtype == torch.bool:
                            valid_mask = (~gop_mask).to(dtype=z_pred.dtype)
                        else:
                            valid_mask = 1 - gop_mask.to(dtype=z_pred.dtype)
                        valid_mask = valid_mask.reshape(-1).to(z_pred.device)

                if valid_mask is None or valid_mask.numel() != z_pred.shape[0]:
                    valid_mask = torch.ones((z_pred.shape[0],), device=z_pred.device, dtype=z_pred.dtype)

                denom = valid_mask.sum().clamp_min(1.0)

                mse_per = F.mse_loss(z_pred, z_tgt, reduction='none').mean(dim=(-2, -1))
                l_recon = (mse_per * valid_mask).sum() / denom

                cos_per = 1.0 - F.cosine_similarity(z_pred.mean(dim=1), z_tgt.mean(dim=1), dim=-1)
                l_delta_cos = (cos_per * valid_mask).sum() / denom

                l_delta_mse = l_recon
                l_delta = l_delta_cos + (self.l_mse_scale * l_delta_mse)
                total_loss = l_recon + (self.lambda_delta * l_delta)

                self.latest_loss_components = {
                    "phase1/delta_mse_loss": l_delta_mse.detach(),
                    "phase1/delta_cosine_loss": l_delta_cos.detach(),
                    "phase1/regenerative_loss": l_recon.detach(),
                    "phase1/total_loss": total_loss.detach(),
                }
                return total_loss

            predicted_spatial = output["predicted_spatial"] # [B, G, 196, 768]
            p_spatial = output["p_spatial"]                 # [B, G, 196, 768]
            i_spatial = output["i_spatial"]                 # [B, G, 196, 768]
            motion_tokens = output["motion_tokens"]         # [B, G, 8, 768]
            valid_mask = output["valid_mask"]               # [B, G]
            
            B, G = valid_mask.shape
            
            l_recon_per_gop = []
            l_delta_per_gop = []
            l_delta_mse_per_gop = []
            l_delta_cos_per_gop = []

            for g in range(G):
                pred_g = predicted_spatial[:, g]        
                target_g = p_spatial[:, g]        
                valid_g = valid_mask[:, g].float()  
                n_valid = valid_g.sum() + 1e-6

                # --- 1. Reconstruction Loss (MSE) ---
                diff_g = (pred_g - target_g).pow(2).mean(dim=(-2, -1)) # [B]
                loss_recon_g = (diff_g * valid_g).sum() / n_valid
                l_recon_per_gop.append(loss_recon_g)

                # --- 2. Delta Distillation (Hybrid) ---
                m_mean_g = motion_tokens[:, g].mean(dim=1)             # [B, 768]
                dt_g = (target_g - i_spatial[:, g]).mean(dim=1)        # [B, 768]

                cos_g = F.cosine_similarity(m_mean_g, dt_g, dim=-1)
                l_cos_g = ((1.0 - cos_g) * valid_g).sum() / n_valid
                l_delta_cos_per_gop.append(l_cos_g)

                mse_g = F.mse_loss(m_mean_g, dt_g, reduction='none').mean(dim=-1) # [B]
                l_mse_g = (mse_g * valid_g).sum() / n_valid
                l_delta_mse_per_gop.append(l_mse_g)

                l_delta_g = l_cos_g + (self.l_mse_scale * l_mse_g)
                l_delta_per_gop.append(l_delta_g)

            l_recon = torch.stack(l_recon_per_gop).mean()
            l_delta = torch.stack(l_delta_per_gop).mean()
            l_delta_mse = torch.stack(l_delta_mse_per_gop).mean()
            l_delta_cos = torch.stack(l_delta_cos_per_gop).mean()

            total_loss = l_recon + (self.lambda_delta * l_delta)

            # For plotting/logging only. Total-loss math is unchanged.
            self.latest_loss_components = {
                "phase1/delta_mse_loss": l_delta_mse.detach(),
                "phase1/delta_cosine_loss": l_delta_cos.detach(),
                "phase1/regenerative_loss": l_recon.detach(),
                "phase1/total_loss": total_loss.detach(),
            }

            return total_loss

        # ==========================================================
        # PHASE 2: SpatialTokenSelector Warmup (MG-DTR)
        # ==========================================================
        elif phase == 2:
            # Assumes model computes per-GOP patch means and counts internally and passes them
            patches_g_mean = output["patches_g_mean"] # [B, G, 768]
            cls_tokens = output["cls_tokens"]         # [B, G, 768]
            coverage_counts = output["coverage_counts"] # [B, G]
            valid_mask = output.get("valid_mask", torch.ones_like(coverage_counts))
            valid_mask = valid_mask.float()
            denom = valid_mask.sum().clamp_min(1.0)
            
            # --- 1. Alignment Loss (Cosine) ---
            cos_loss = 1.0 - F.cosine_similarity(patches_g_mean, cls_tokens, dim=-1) # [B, G]
            l_align = (cos_loss * valid_mask).sum() / denom

            # --- 2. Coverage Penalty ---
            l_coverage = (F.relu(self.rho_min_tokens - coverage_counts.float()) * valid_mask).sum() / denom

            total_loss = l_align + (self.lambda_coverage * l_coverage)
            self.latest_loss_components = {
                "phase2/align_loss": l_align.detach(),
                "phase2/coverage_loss": l_coverage.detach(),
                "phase2/total_loss": total_loss.detach(),
            }
            return total_loss

        # ==========================================================
        # PHASE 3: Full Captioning SFT
        # ==========================================================
        elif phase == 3:
            # --- 1. Cross Entropy (Language) ---
            l_ce = self.ce_loss_fn(target, output)

            p_spatial = output.get("p_spatial", None)
            i_spatial = output.get("i_spatial", None)
            motion_tokens = output.get("motion_tokens", None)
            valid_mask = output.get("valid_mask", None)

            # If phase3 distillation tensors are unavailable, use CE-only training.
            if p_spatial is None or i_spatial is None or motion_tokens is None or valid_mask is None:
                total_loss = l_ce
                zero = torch.zeros((), device=l_ce.device, dtype=l_ce.dtype)
                self.latest_loss_components = {
                    "phase3/caption_ce_loss": l_ce.detach(),
                    "phase3/distill_loss": zero.detach(),
                    "phase3/total_loss": total_loss.detach(),
                }
                return total_loss
            
            # --- 2. Distillation Regularizer (GOP-Wise) ---
            p_spatial = p_spatial                             # [B, G, 196, 768]
            i_spatial = i_spatial                             # [B, G, 196, 768]
            motion_tokens = motion_tokens                     # [B, G, 8, 768]
            valid_mask = valid_mask                           # [B, G]
            
            B, G = valid_mask.shape
            l_distill_per_gop = []

            for g in range(G):
                m_mean_g = motion_tokens[:, g].mean(dim=1)             # [B, 768]
                dt_g = (p_spatial[:, g] - i_spatial[:, g]).mean(dim=1) # [B, 768]
                valid_g = valid_mask[:, g].float()
                n_valid = valid_g.sum() + 1e-6

                cos_g = F.cosine_similarity(m_mean_g, dt_g, dim=-1)
                l_cos_g = ((1.0 - cos_g) * valid_g).sum() / n_valid

                mse_g = F.mse_loss(m_mean_g, dt_g, reduction='none').mean(dim=-1)
                l_mse_g = (mse_g * valid_g).sum() / n_valid

                l_distill_per_gop.append(l_cos_g + (self.l_mse_scale * l_mse_g))
            
            l_distill = torch.stack(l_distill_per_gop).mean()

            total_loss = l_ce + (self.lambda_reg * l_distill)
            self.latest_loss_components = {
                "phase3/caption_ce_loss": l_ce.detach(),
                "phase3/distill_loss": l_distill.detach(),
                "phase3/total_loss": total_loss.detach(),
            }
            return total_loss
        
        else:
            raise ValueError(f"Unknown training phase: {phase}")

# Register for Hydra
phase_aware_loss_cfg = builds(PhaseAwareLoss, populate_full_signature=True)
label_smoothing_loss_cfg = builds(LabelSmoothingLoss, populate_full_signature=True)