# -*- coding: utf-8 -*-
# @Time    : 6/16/25
# @Author  : Yaojie Shen (Updated for Hav-CoCap)
# @Project : CoCap
# @File    : lm_cocap.py

import copy
import logging
import os
from collections import defaultdict

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from hydra_zen import builds
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
from transformers import GPT2Tokenizer

# Import the new Hav-CoCap Architecture
from cocap.modules.hav_cocap_gpt2 import HavCoCapGPT2

# Import the new Phase Aware Loss
from .loss import LossBase, phase_aware_loss_cfg
from .eval_captioning import evaluate
from ..utils.json import save_json
from ..utils.train_utils import gather_object_multiple_gpu, get_timestamp

logger = logging.getLogger(__name__)


class CoCapLM(pl.LightningModule):
    """Hav-CoCap Lightning Module with Phase-Aware Multitask Optimization"""

    def __init__(
            self,
            cocap_model: nn.Module,   # Now specifically takes the HavCoCapGPT2
            loss: LossBase,
            phase: int = 2,           
            mask_ratio: float = 0.75,
            lr: float = 1e-4,
            clip_lr: float = 1e-6,    
            warmup_ratio: float = 0.05,
            lr_decay_gamma: float = 0.95,
            weight_decay: float = 0.01
    ):
        super().__init__()
        self.model = cocap_model
        
        # --- STRICT OFFLINE TOKENIZER LOADING ---
        # Resolve the absolute path to your offline folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        offline_gpt2_path = os.path.join(current_dir, "../../model_zoo/gpt2_model")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            offline_gpt2_path, 
            local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # ----------------------------------------
        
        self.mask_ratio = mask_ratio
        self.loss = loss
        self.phase = phase
        
        # 1. Lock the internal model to the exact phase specified in the config
        if hasattr(self.model, 'set_training_phase'):
            self.model.set_training_phase(self.phase)
            
        self.lr = lr
        self.clip_lr = clip_lr
        self.warmup_ratio = warmup_ratio
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay

        self.batch_res = None

    @property
    def total_steps(self):
        return self.trainer.estimated_stepping_batches

    @property
    def epoch_steps(self):
        return self.trainer.estimated_stepping_batches // max(1, self.trainer.max_epochs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Professional Optimizer Setup using Native PyTorch AdamW:
        1. Only optimizes parameters where requires_grad == True
        2. Safely excludes LayerNorms and Biases from Weight Decay
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue # Skip frozen parameters completely
                
            # No weight decay for biases, LayerNorms, or Embeddings
            if param.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name or "ln_" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Loss parameters (if any learnable params exist in the loss module)
        loss_params = [p for p in self.loss.parameters() if p.requires_grad]
        if loss_params:
            no_decay_params.extend(loss_params)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, eps=1e-8)

        # Step-based warmup, epoch-based decay
        def lr_lambda(current_step):
            warmup_steps = max(1, int(self.warmup_ratio * self.total_steps))
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            else:
                return self.lr_decay_gamma ** ((current_step - warmup_steps) // max(1, self.epoch_steps))

        warmup_decay_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_decay_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_decay"
            }
        }

    def training_step(self, batch, batch_idx):
        iframes = batch["video"]["iframe"] if "video" in batch else batch["iframe"]
        
        # --- THE T+1 SHIFT HACK ---
        # Shift frames left by 1 to get t+1. Duplicate the last frame to maintain G=8.
        next_iframes = torch.cat([iframes[:, 1:], iframes[:, -1:]], dim=1)
        
        # Pass the shifted frames to the model
        outputs = self.model(
            iframes=iframes, 
            motion_vectors=batch["video"]["motion_vector"] if "video" in batch else batch["motion"], 
            input_ids=batch["input_ids"], 
            attention_mask=batch["input_mask"], 
            labels=batch["input_labels"]
        )
        
        loss = self.loss(target=batch, output=outputs, phase=self.phase)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["input_labels"].size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.batch_res = {"version": "VERSION 1.0",
                          "results": defaultdict(list),
                          "external_data": {"used": "true", "details": "ay"}}

        # Print one-time validation diagnostics (rank 0 only in distributed)
        if not dist.is_initialized() or dist.get_rank() == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            non_trainable_params = total_params - trainable_params

            val_dl = self.trainer.val_dataloaders
            if isinstance(val_dl, list):
                val_dl = val_dl[0]
            val_dataset = getattr(val_dl, "dataset", None)

            val_samples = len(val_dataset) if val_dataset is not None else -1
            val_batch_size = getattr(val_dl, "batch_size", "unknown")
            unfold_sentences = getattr(val_dataset, "unfold_sentences", "unknown") if val_dataset is not None else "unknown"

            self.print(
                "\n===== Validation Diagnostics =====\n"
                f"Phase Active: Phase {self.phase}\n"
                f"params(total/trainable/non-trainable): {total_params}/{trainable_params}/{non_trainable_params}\n"
                f"val_dataset_samples: {val_samples}\n"
                f"val_batch_size: {val_batch_size}\n"
                f"val_unfold_sentences: {unfold_sentences}\n"
                "=================================\n"
            )

    def validation_step(self, batch, batch_idx):
        iframes = batch["video"]["iframe"] if "video" in batch else batch["iframe"]
        motion_vectors = batch["video"]["motion_vector"] if "video" in batch else batch["motion"]
        
        # Generate text directly using the GPT-2 Generate function
        gen_sentences = []
        for i in range(iframes.size(0)):
            # Pass single video slices to generate
            sentence = self.model.generate(
                iframes=iframes[i:i+1], 
                motion_vectors=motion_vectors[i:i+1], 
                tokenizer=self.tokenizer,
                max_new_tokens=30
            )
            gen_sentences.append(sentence)

        # Append to results for CIDEr/BLEU evaluation
        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(gen_sentences, batch['metadata'][1])):
            cur_data = {
                "sentence": cur_gen_sen,
                "gt_sentence": cur_meta
            }
            self.batch_res["results"][batch['metadata'][0][example_idx].split("video")[-1]].append(cur_data)

    def on_validation_epoch_end(self) -> None:
        json_res = copy.deepcopy(self.batch_res)
        if dist.is_initialized():
            all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
            json_res['results'] = {k: v for k, v in all_results}
            logger.debug("Caption test length: %s", len(json_res["results"].items()))

        # save result to log for debug
        if not dist.is_initialized() or dist.get_rank() == 0:
            res_filepath = os.path.join(self.trainer.default_root_dir,
                                        "caption_greedy_pred_validation_{}.json".format(get_timestamp()))
            os.makedirs(os.path.dirname(res_filepath), exist_ok=True)
            save_json(json_res, res_filepath, save_pretty=True)

        if not dist.is_initialized() or dist.get_rank() == 0:
            val_dl = self.trainer.val_dataloaders
            if isinstance(val_dl, list):
                val_dl = val_dl[0]
            json_ref = val_dl.dataset.json_ref
            metrics = evaluate(json_res, json_ref)
            self.print(f"\n================ Validation Metrics ================\n{metrics}\n==================================================")
            self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        if dist.is_initialized():
            dist.barrier()


# =====================================================================
# HYDRA CONFIGURATION BUILDERS
# =====================================================================

# 1. Build the Hav-CoCap GPT-2 Architecture
hav_cocap_cfg = builds(
    HavCoCapGPT2,
    gpt2_model_path="./model_zoo/gpt2_model", # MUST point to your local offline folder!
    populate_full_signature=True
)

# 2. Build the full Lightning Module
cocap_lm_cfg = builds(
    CoCapLM,
    cocap_model=hav_cocap_cfg, 
    loss=phase_aware_loss_cfg, 
    populate_full_signature=True
)