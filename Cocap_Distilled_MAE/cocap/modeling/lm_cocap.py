# -*- coding: utf-8 -*-
# @Time    : 6/16/25
# @Author  : Yaojie Shen / Asok BK (Updated for Hav-CoCap)
# @Project : CoCap
# @File    : lm_cocap.py

import copy
import logging
import math
import os
from collections import defaultdict
from typing import Any

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
            cocap_model: nn.Module,   
            loss: LossBase,
            phase: int = 2,           
            use_preextracted_features: bool = True,
            init_weights_ckpt: str = "",
            init_motion_ckpt: str = "",
            init_mgdtr_ckpt: str = "",
            mask_ratio: float = 0.75,
            lr: float = 1e-4,         # Used for Phase 1 & 2
            clip_lr: float = 1e-6,    # Fallback/reference
            warmup_ratio: float = 0.05,
            lr_decay_gamma: float = 0.95, # Unused now (replaced by Cosine Decay)
                weight_decay: float = 0.01,
                phase_hparams: Any = None,
    ):
        super().__init__()
        self.model = cocap_model
        
        # --- STRICT OFFLINE TOKENIZER LOADING ---
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
        self.use_preextracted_features = bool(use_preextracted_features)
        self.init_weights_ckpt = init_weights_ckpt
        self.init_motion_ckpt = init_motion_ckpt
        self.init_mgdtr_ckpt = init_mgdtr_ckpt
        self._did_init_from_ckpt = False
        self._best_train_loss = None
        
        # Lock the internal model to the exact phase specified in the config
        if hasattr(self.model, 'set_training_phase'):
            self.model.set_training_phase(self.phase)

        # Phase-wise memory optimization
        if hasattr(self.model, "unload_gpt2") and self.phase in [1, 2]:
            self.model.unload_gpt2()
        # Keep CLIP loaded for all phases. It is required when raw MP4 path is used
        # (fallback CLIP feature extraction from I/P frames).
            
        self.lr = lr
        self.clip_lr = clip_lr
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.phase_hparams = phase_hparams or {}

        self.batch_res = None
        self._logged_lr_groups_once = False

    def _module_param_stats(self, module: nn.Module):
        if module is None:
            return (0, 0)
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    def _log_phase_module_status(self) -> None:
        m = self.model
        rows = []

        clip = getattr(m, "clip", None)
        motion = getattr(m, "motion_encoder", None)
        student = getattr(motion, "student", None) if motion is not None else None
        decoder = getattr(motion, "decoder", None) if motion is not None else None
        mgdtr = getattr(m, "mgdtr", getattr(m, "selector", None))
        proj_spatial = getattr(m, "proj_spatial", None)
        motion_to_gpt = getattr(m, "motion_to_gpt", None)
        gpt2 = getattr(m, "gpt2", None)

        for name, mod in [
            ("clip", clip),
            ("motion_encoder", motion),
            ("motion_student", student),
            ("distillation_decoder", decoder),
            ("mgdtr", mgdtr),
            ("proj_spatial", proj_spatial),
            ("motion_to_gpt", motion_to_gpt),
            ("gpt2", gpt2),
        ]:
            total, trainable = self._module_param_stats(mod)
            status = "unloaded" if mod is None else ("trainable" if trainable > 0 else "frozen")
            rows.append(f"{name}={status} ({trainable}/{total})")

        logger.info("Phase %s module status: %s", self.phase, " | ".join(rows))

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Checkpoint compatibility shim.

        - Old checkpoints stored MGDTR under `model.selector.*`.
        - New model uses `model.mgdtr.*` and keeps `selector` as a property alias.
        - New model also has mandatory `model.motion_to_gpt.*` which old checkpoints may miss.
        """
        state = checkpoint.get("state_dict", None)
        if not isinstance(state, dict):
            return

        remapped = {}
        for k, v in state.items():
            if k.startswith("model.selector."):
                new_k = "model.mgdtr." + k[len("model.selector."):]
                # Prefer explicit mgdtr keys if both exist.
                if new_k not in state:
                    remapped[new_k] = v
                continue
            remapped[k] = v

        # Backfill new mandatory projection layer when loading old checkpoints.
        cur = self.state_dict()
        for req_key in ("model.motion_to_gpt.weight", "model.motion_to_gpt.bias"):
            if req_key not in remapped and req_key in cur:
                remapped[req_key] = cur[req_key].detach().clone()

        checkpoint["state_dict"] = remapped

    def on_fit_start(self) -> None:
        if self.phase == 3 and self.use_preextracted_features:
            raise ValueError(
                "Phase 3 is configured to use raw MP4 flow only. "
                "Please run with ++model.use_preextracted_features=false"
            )

        self._apply_default_phase_handoff_ckpts()

        # Robustify user-provided init paths with phase-aware fallback.
        self._resolve_missing_init_ckpts()

        if self.phase == 3 and not self.init_weights_ckpt:
            if not self.init_motion_ckpt or not self.init_mgdtr_ckpt:
                raise ValueError(
                    "Phase 3 requires either `init_weights_ckpt` (full model) OR both "
                    "`init_motion_ckpt` and `init_mgdtr_ckpt`."
                )

        if self._did_init_from_ckpt or not self.init_weights_ckpt:
            # Still allow module-specific init even without full-model init.
            self._maybe_load_submodule_ckpt(
                ckpt_path=self.init_motion_ckpt,
                module=getattr(self.model, "motion_encoder", None),
                lightning_prefixes=["model.motion_encoder.", "module.model.motion_encoder."],
                module_name="motion_encoder",
                keep_prefixes=["student."] if self.phase in (2, 3) else None,
            )
            self._maybe_load_submodule_ckpt(
                ckpt_path=self.init_mgdtr_ckpt,
                module=getattr(self.model, "mgdtr", getattr(self.model, "selector", None)),
                lightning_prefixes=[
                    "model.mgdtr.",
                    "module.model.mgdtr.",
                    "model.selector.",
                    "module.model.selector.",
                ],
                module_name="mgdtr_selector",
            )
            self._log_phase_module_status()
            return

        if not os.path.isfile(self.init_weights_ckpt):
            raise FileNotFoundError(f"init_weights_ckpt not found: {self.init_weights_ckpt}")

        ckpt = torch.load(self.init_weights_ckpt, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model_state = {}

        for k, v in state.items():
            if k.startswith("model."):
                model_state[k[len("model."):]] = v
            elif k.startswith("module.model."):
                model_state[k[len("module.model."):]] = v

        if len(model_state) == 0:
            model_state = state

        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        logger.info(
            "Initialized model weights from %s (missing=%d, unexpected=%d)",
            self.init_weights_ckpt,
            len(missing),
            len(unexpected),
        )

        # Optional targeted re-load on top (if user passes dedicated module ckpts).
        self._maybe_load_submodule_ckpt(
            ckpt_path=self.init_motion_ckpt,
            module=getattr(self.model, "motion_encoder", None),
            lightning_prefixes=["model.motion_encoder.", "module.model.motion_encoder."],
            module_name="motion_encoder",
            keep_prefixes=["student."] if self.phase in (2, 3) else None,
        )
        self._maybe_load_submodule_ckpt(
            ckpt_path=self.init_mgdtr_ckpt,
            module=getattr(self.model, "mgdtr", getattr(self.model, "selector", None)),
            lightning_prefixes=[
                "model.mgdtr.",
                "module.model.mgdtr.",
                "model.selector.",
                "module.model.selector.",
            ],
            module_name="mgdtr_selector",
        )
        self._did_init_from_ckpt = True
        self._log_phase_module_status()

    def _resolve_missing_init_ckpts(self) -> None:
        """Resolve invalid/missing init ckpt paths using phase-aware defaults.

        This prevents hard failure when users pass stale paths (e.g., phase3 motion from phase2 modules).
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        def _exists(p: str) -> bool:
            return bool(p) and os.path.isfile(p)

        if self.phase == 2:
            if self.init_motion_ckpt and not _exists(self.init_motion_ckpt):
                logger.warning("init_motion_ckpt not found (%s). Falling back to phase1 motion module.", self.init_motion_ckpt)
                self.init_motion_ckpt = ""
            if not self.init_motion_ckpt:
                p = os.path.join(project_root, "logs", "vatex_captioning", "phase1", "checkpoints", "modules", "motion_encoder_best.pt")
                if _exists(p):
                    self.init_motion_ckpt = p
                    logger.info("Resolved phase2 init_motion_ckpt=%s", p)

        if self.phase == 3:
            if self.init_motion_ckpt and not _exists(self.init_motion_ckpt):
                logger.warning("phase3 init_motion_ckpt not found (%s). Falling back to phase1 motion module.", self.init_motion_ckpt)
                self.init_motion_ckpt = ""
            if self.init_mgdtr_ckpt and not _exists(self.init_mgdtr_ckpt):
                logger.warning("phase3 init_mgdtr_ckpt not found (%s). Falling back to phase2 mgdtr module.", self.init_mgdtr_ckpt)
                self.init_mgdtr_ckpt = ""

            if not self.init_motion_ckpt:
                p_motion = os.path.join(project_root, "logs", "vatex_captioning", "phase1", "checkpoints", "modules", "motion_encoder_best.pt")
                if _exists(p_motion):
                    self.init_motion_ckpt = p_motion
                    logger.info("Resolved phase3 init_motion_ckpt=%s", p_motion)

            if not self.init_mgdtr_ckpt:
                p_mgdtr = os.path.join(project_root, "logs", "vatex_captioning", "phase2", "checkpoints", "modules", "mgdtr_selector_best.pt")
                if _exists(p_mgdtr):
                    self.init_mgdtr_ckpt = p_mgdtr
                    logger.info("Resolved phase3 init_mgdtr_ckpt=%s", p_mgdtr)

    def _apply_default_phase_handoff_ckpts(self) -> None:
        """Default phase handoff:
        - phase2: if no manual init is set, initialize motion encoder from phase1 module .pt
        - phase3: if no manual init is set, initialize motion + mgdtr from phase2 module .pt
          (fallback to phase2 last.ckpt only if module files are missing)
        """
        if self.init_weights_ckpt or self.init_motion_ckpt or self.init_mgdtr_ckpt:
            return

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cur_root = (self.trainer.default_root_dir or "").replace("\\", "/")
        if "/phase2" in cur_root and self.phase == 2:
            p_motion = os.path.join(cur_root.replace("/phase2", "/phase1"), "checkpoints", "modules", "motion_encoder_best.pt")
            if os.path.isfile(p_motion):
                self.init_motion_ckpt = p_motion
                logger.info("Auto phase2 handoff: init_motion_ckpt=%s", p_motion)
            else:
                default_motion = os.path.join(project_root, "logs", "vatex_captioning", "phase1", "checkpoints", "modules", "motion_encoder_best.pt")
                if os.path.isfile(default_motion):
                    self.init_motion_ckpt = default_motion
                    logger.info("Auto phase2 handoff (project-root fallback): init_motion_ckpt=%s", default_motion)
                    return
                p = os.path.join(cur_root.replace("/phase2", "/phase1"), "checkpoints", "best", "last.ckpt")
                if os.path.isfile(p):
                    self.init_weights_ckpt = p
                    logger.info("Auto phase2 handoff fallback: init_weights_ckpt=%s", p)

        if "/phase3" in cur_root and self.phase == 3:
            p2_root = cur_root.replace("/phase3", "/phase2")
            p1_root = cur_root.replace("/phase3", "/phase1")
            p_motion = os.path.join(p1_root, "checkpoints", "modules", "motion_encoder_best.pt")
            p_mgdtr = os.path.join(p2_root, "checkpoints", "modules", "mgdtr_selector_best.pt")
            if os.path.isfile(p_motion):
                self.init_motion_ckpt = p_motion
                logger.info("Auto phase3 handoff: init_motion_ckpt=%s", p_motion)
            if os.path.isfile(p_mgdtr):
                self.init_mgdtr_ckpt = p_mgdtr
                logger.info("Auto phase3 handoff: init_mgdtr_ckpt=%s", p_mgdtr)

            if not self.init_motion_ckpt:
                default_motion = os.path.join(project_root, "logs", "vatex_captioning", "phase1", "checkpoints", "modules", "motion_encoder_best.pt")
                if os.path.isfile(default_motion):
                    self.init_motion_ckpt = default_motion
                    logger.info("Auto phase3 handoff (project-root fallback): init_motion_ckpt=%s", default_motion)
            if not self.init_mgdtr_ckpt:
                default_mgdtr = os.path.join(project_root, "logs", "vatex_captioning", "phase2", "checkpoints", "modules", "mgdtr_selector_best.pt")
                if os.path.isfile(default_mgdtr):
                    self.init_mgdtr_ckpt = default_mgdtr
                    logger.info("Auto phase3 handoff (project-root fallback): init_mgdtr_ckpt=%s", default_mgdtr)

            if not self.init_motion_ckpt or not self.init_mgdtr_ckpt:
                p = os.path.join(p2_root, "checkpoints", "best", "last.ckpt")
                if os.path.isfile(p):
                    self.init_weights_ckpt = p
                    logger.info("Auto phase3 handoff fallback: init_weights_ckpt=%s", p)

    @staticmethod
    def _maybe_load_submodule_ckpt(
        ckpt_path: str,
        module: nn.Module,
        lightning_prefixes,
        module_name: str,
        keep_prefixes=None,
    ) -> None:
        if not ckpt_path:
            return
        if module is None:
            logger.warning("Requested %s init from %s, but module is None", module_name, ckpt_path)
            return
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"{module_name} checkpoint not found: {ckpt_path}")

        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = raw.get("state_dict", raw)

        filtered = {}
        for k, v in state.items():
            for prefix in lightning_prefixes:
                if k.startswith(prefix):
                    filtered[k[len(prefix):]] = v
                    break

        if len(filtered) == 0:
            filtered = state

        if keep_prefixes:
            filtered = {
                k: v for k, v in filtered.items()
                if any(k.startswith(pref) for pref in keep_prefixes)
            }

        missing, unexpected = module.load_state_dict(filtered, strict=False)
        logger.info(
            "Initialized %s from %s (loaded_keys=%d, missing=%d, unexpected=%d)",
            module_name,
            ckpt_path,
            len(filtered),
            len(missing),
            len(unexpected),
        )

    def on_train_start(self) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        max_epochs = getattr(self.trainer, "max_epochs", None)
        if hasattr(self.trainer, "fit_loop") and hasattr(self.trainer.fit_loop, "max_epochs"):
            max_epochs = self.trainer.fit_loop.max_epochs
        patience = None
        for cb in self.trainer.callbacks:
            if cb.__class__.__name__ == "EarlyStopping" and getattr(cb, "monitor", "") == "train_loss":
                patience = getattr(cb, "patience", None)
                break
        logger.info("Training policy: phase=%s | max_epochs=%s | early_stopping(train_loss).patience=%s", self.phase, max_epochs, patience)

    def on_train_epoch_start(self) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        if self._logged_lr_groups_once:
            return
        if not self.trainer.optimizers:
            return
        opt = self.trainer.optimizers[0]
        cur_lrs = [pg.get("lr", None) for pg in opt.param_groups]
        base_lrs = [pg.get("initial_lr", pg.get("lr", None)) for pg in opt.param_groups]
        cur_fmt = ", ".join([f"{v:.3e}" if isinstance(v, float) else str(v) for v in cur_lrs])
        base_fmt = ", ".join([f"{v:.3e}" if isinstance(v, float) else str(v) for v in base_lrs])
        logger.info("LR groups (current): [%s] | (base): [%s]", cur_fmt, base_fmt)
        self._logged_lr_groups_once = True

    @property
    def total_steps(self):
        return self.trainer.estimated_stepping_batches

    @property
    def epoch_steps(self):
        return self.trainer.estimated_stepping_batches // max(1, self.trainer.max_epochs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Phase-Aware Optimizer Setup using Native PyTorch AdamW:
        - Phase 1 & 2: Standard unified learning rate.
        - Phase 3: Differential learning rates to protect pre-trained physics.
        - Uses Cosine Decay with Warmup.
        """
        optimizer_grouped_parameters = []

        # Pull phase profile from config (accepts dict or DictConfig-like objects).
        phase_key = f"phase{self.phase}"
        phase_cfg = None
        if self.phase_hparams:
            if hasattr(self.phase_hparams, "get"):
                phase_cfg = self.phase_hparams.get(phase_key, None)
            if phase_cfg is None and hasattr(self.phase_hparams, "__getitem__"):
                try:
                    phase_cfg = self.phase_hparams[phase_key]
                except Exception:
                    phase_cfg = None

        learning_rates = None
        if phase_cfg is not None and hasattr(phase_cfg, "get"):
            learning_rates = phase_cfg.get("learning_rates", None)
        
        # =======================================================
        # PHASE 1 & 2: Standard Unified Learning Rate
        # =======================================================
        if self.phase in [1, 2]:
            phase_lr = self.lr
            if learning_rates is not None and hasattr(learning_rates, "get"):
                if self.phase == 1:
                    phase_lr = float(learning_rates.get("motion_student", self.lr))
                elif self.phase == 2:
                    phase_lr = float(learning_rates.get("mgdtr_selector", self.lr))

            decay_params = []
            no_decay_params = []
            
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue 
                
                # No weight decay for biases, LayerNorms, or Embeddings
                if param.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name or "ln_" in name or "embedding" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            # Check if loss module has learnable parameters
            loss_params = [p for p in self.loss.parameters() if p.requires_grad]
            if loss_params:
                no_decay_params.extend(loss_params)

            optimizer_grouped_parameters = [
                {"params": decay_params, "weight_decay": self.weight_decay, "lr": phase_lr},
                {"params": no_decay_params, "weight_decay": 0.0, "lr": phase_lr}
            ]

        # =======================================================
        # PHASE 3: Differential Learning Rates
        # =======================================================
        elif self.phase == 3:
            # Defaults are preserved, but can be overridden from model.phase_hparams.phase3.learning_rates.
            motion_backbone_lr = 1e-6
            motion_transformer_lr = 5e-6
            proj_student_to_mgdtr_lr = 1e-5
            mgdtr_lr = 1e-5
            proj_spatial_to_gpt_lr = 5e-5
            proj_student_to_gpt_lr = 5e-5
            if learning_rates is not None and hasattr(learning_rates, "get"):
                motion_backbone_lr = float(learning_rates.get("motion_student_backbone", motion_backbone_lr))
                motion_transformer_lr = float(learning_rates.get("motion_student_transformer", motion_transformer_lr))
                proj_student_to_mgdtr_lr = float(learning_rates.get("proj_student_to_mgdtr", proj_student_to_mgdtr_lr))
                mgdtr_lr = float(learning_rates.get("mgdtr_selector", mgdtr_lr))
                proj_spatial_to_gpt_lr = float(learning_rates.get("proj_spatial_to_gpt", proj_spatial_to_gpt_lr))
                proj_student_to_gpt_lr = float(learning_rates.get("proj_student_to_gpt", proj_student_to_gpt_lr))

            param_groups = {
                "motion_backbone_decay": {"params": [], "lr": motion_backbone_lr, "weight_decay": self.weight_decay},
                "motion_backbone_no_decay": {"params": [], "lr": motion_backbone_lr, "weight_decay": 0.0},
                "motion_transformer_decay": {"params": [], "lr": motion_transformer_lr, "weight_decay": self.weight_decay},
                "motion_transformer_no_decay": {"params": [], "lr": motion_transformer_lr, "weight_decay": 0.0},
                "mgdtr_proj_decay": {"params": [], "lr": proj_student_to_mgdtr_lr, "weight_decay": self.weight_decay},
                "mgdtr_proj_no_decay": {"params": [], "lr": proj_student_to_mgdtr_lr, "weight_decay": 0.0},
                "mgdtr_decay": {"params": [], "lr": mgdtr_lr, "weight_decay": self.weight_decay},
                "mgdtr_no_decay": {"params": [], "lr": mgdtr_lr, "weight_decay": 0.0},
                "proj_spatial_decay": {"params": [], "lr": proj_spatial_to_gpt_lr, "weight_decay": self.weight_decay},
                "proj_spatial_no_decay": {"params": [], "lr": proj_spatial_to_gpt_lr, "weight_decay": 0.0},
                "proj_motion_to_gpt_decay": {"params": [], "lr": proj_student_to_gpt_lr, "weight_decay": self.weight_decay},
                "proj_motion_to_gpt_no_decay": {"params": [], "lr": proj_student_to_gpt_lr, "weight_decay": 0.0},
            }

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                    
                is_no_decay = (param.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name or "ln_" in name)
                is_mgdtr = ("mgdtr" in name) or ("selector" in name) or ("router" in name)
                is_mgdtr_proj = ("mgdtr.motion_proj" in name) or ("mgdtr.motion_proj_768" in name)
                is_motion_backbone = (
                    "motion_encoder.student.stem" in name
                )
                is_motion_transformer = (
                    "motion_encoder.student.transformer" in name
                    or "motion_encoder.student.cross_attn" in name
                    or "motion_encoder.student.ln_" in name
                    or "motion_encoder.student.motion_queries" in name
                )
                is_proj_spatial = ("proj_spatial" in name)
                is_proj_motion_to_gpt = ("motion_to_gpt" in name)

                if is_motion_backbone:
                    bucket = "motion_backbone_no_decay" if is_no_decay else "motion_backbone_decay"
                elif is_motion_transformer:
                    bucket = "motion_transformer_no_decay" if is_no_decay else "motion_transformer_decay"
                elif is_mgdtr_proj:
                    bucket = "mgdtr_proj_no_decay" if is_no_decay else "mgdtr_proj_decay"
                elif is_mgdtr:
                    bucket = "mgdtr_no_decay" if is_no_decay else "mgdtr_decay"
                elif is_proj_spatial:
                    bucket = "proj_spatial_no_decay" if is_no_decay else "proj_spatial_decay"
                elif is_proj_motion_to_gpt:
                    bucket = "proj_motion_to_gpt_no_decay" if is_no_decay else "proj_motion_to_gpt_decay"
                else:
                    bucket = "proj_spatial_no_decay" if is_no_decay else "proj_spatial_decay"

                param_groups[bucket]["params"].append(param)
            
            # Pack into expected format, ignoring empty groups
            optimizer_grouped_parameters = [v for k, v in param_groups.items() if len(v["params"]) > 0]

        # Initialize Optimizer
        optimizer = AdamW(optimizer_grouped_parameters, eps=1e-8)

        # =======================================================
        # SCHEDULER: Warmup + Cosine Decay
        # =======================================================
        def lr_lambda(current_step):
            warmup_steps = max(1, int(self.warmup_ratio * self.total_steps))
            if current_step < warmup_steps:
                return float(current_step) / float(warmup_steps)
            else:
                progress = float(current_step - warmup_steps) / float(max(1, self.total_steps - warmup_steps))
                # Cosine decay from 1.0 down to 0.0
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine_decay"
            }
        }

    def training_step(self, batch, batch_idx):
        video = batch.get("video", batch)

        clip_i_cls = video.get("clip_i_cls", None)
        clip_i_spatial = video.get("clip_i_spatial", None)
        clip_p_spatial = video.get("clip_p_spatial", None)
        motion_vectors = video.get("motion_vectors", video.get("motion_vector", None))
        input_mask_mv = video.get("input_mask_mv", None)
        input_mask_gop = video.get("input_mask_gop", None)

        # Backward-compatible fallback for raw-video batches.
        if clip_i_cls is None or clip_i_spatial is None:
            if self.use_preextracted_features:
                raise RuntimeError(
                    "use_preextracted_features=True but batch is missing clip_i_cls/clip_i_spatial. "
                    "Please ensure dataset/vatex/video_preextracted_pt contains required .pt files."
                )
            iframes = video["iframe"]
            if self.model.clip is None:
                raise RuntimeError("CLIP model is not loaded; cannot compute CLIP features from raw frames.")

            B, G = iframes.shape[:2]
            with torch.no_grad():
                i_out = self.model.clip.encode_image(
                    iframes.view(B * G, 3, 224, 224), output_all_features=True
                )[1]

            clip_i_cls = i_out[:, 0, :].view(B, G, 768)
            clip_i_spatial = i_out[:, 1:, :].view(B, G, 196, 768)

            # Phase 1 requires P-frame CLIP spatial targets for distillation.
            if self.phase == 1:
                last_p_frames = video.get("last_p_frame", None)
                if last_p_frames is None:
                    raise RuntimeError("Phase 1 requires `last_p_frame` when using raw-video CLIP fallback.")
                with torch.no_grad():
                    p_out = self.model.clip.encode_image(
                        last_p_frames.view(B * G, 3, 224, 224), output_all_features=True
                    )[1]
                clip_p_spatial = p_out[:, 1:, :].view(B, G, 196, 768)

        if motion_vectors is None:
            raise RuntimeError("Missing `motion_vectors`/`motion_vector` in batch video dict.")

        if input_mask_mv is None:
            B, G, M = motion_vectors.shape[:3]
            input_mask_mv = torch.zeros((B, G, M), dtype=torch.long, device=motion_vectors.device)

        if input_mask_gop is None:
            B, G = motion_vectors.shape[:2]
            input_mask_gop = torch.zeros((B, G), dtype=torch.long, device=motion_vectors.device)

        target_delta = None
        if clip_p_spatial is not None and clip_i_spatial is not None:
            target_delta = clip_p_spatial - clip_i_spatial

        outputs = self.model(
            clip_i_cls=clip_i_cls,
            clip_i_spatial=clip_i_spatial,
            motion_vectors=motion_vectors,
            input_ids=batch["input_ids"],
            attention_mask=batch["input_mask"],
            labels=batch["input_labels"],
            input_mask_mv=input_mask_mv,
            input_mask_gop=input_mask_gop,
            target_delta=target_delta,
        )
        
        loss = self.loss(target=batch, output=outputs, phase=self.phase)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,
                 batch_size=batch["input_labels"].size(0), sync_dist=True)

        # Log component losses (especially phase-1 4-loss breakdown) for plotting.
        if hasattr(self.loss, "latest_loss_components") and isinstance(self.loss.latest_loss_components, dict):
            for k, v in self.loss.latest_loss_components.items():
                if torch.is_tensor(v):
                    self.log(
                        k,
                        v,
                        on_step=True,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        batch_size=batch["input_labels"].size(0),
                        sync_dist=True,
                    )
        return loss

    def on_train_epoch_end(self) -> None:
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        module_dir = os.path.join(self.trainer.default_root_dir, "checkpoints", "modules")
        phase_dir = os.path.join(module_dir, f"phase{self.phase}")
        os.makedirs(module_dir, exist_ok=True)
        os.makedirs(phase_dir, exist_ok=True)

        cur_epoch = int(self.current_epoch)
        cur_step = int(self.global_step)

        # Phase-aware module saving policy:
        # - phase1: motion encoder only
        # - phase2: MGDTR/selector only
        # - phase3: motion encoder + MGDTR/selector
        save_motion = self.phase in (1, 3)
        save_mgdtr = self.phase in (2, 3)

        if save_motion and hasattr(self.model, "motion_encoder"):
            motion_payload = {
                "phase": int(self.phase),
                "epoch": cur_epoch,
                "step": cur_step,
                "state_dict": self.model.motion_encoder.state_dict(),
            }
            torch.save(
                motion_payload,
                os.path.join(module_dir, "motion_encoder_last.pt"),
            )
            torch.save(
                motion_payload,
                os.path.join(phase_dir, f"motion_encoder-epoch{cur_epoch:03d}-step{cur_step}.pt"),
            )

        # MGDTR-equivalent in current Hav-CoCap wrapper is selector.
        if save_mgdtr and hasattr(self.model, "selector") and self.model.selector is not None:
            mgdtr_payload = {
                "phase": int(self.phase),
                "epoch": cur_epoch,
                "step": cur_step,
                "state_dict": self.model.selector.state_dict(),
            }
            torch.save(
                mgdtr_payload,
                os.path.join(module_dir, "mgdtr_selector_last.pt"),
            )
            torch.save(
                mgdtr_payload,
                os.path.join(phase_dir, f"mgdtr_selector-epoch{cur_epoch:03d}-step{cur_step}.pt"),
            )

        train_loss = self.trainer.callback_metrics.get("train_loss")
        if train_loss is not None:
            cur = float(train_loss.detach().cpu()) if torch.is_tensor(train_loss) else float(train_loss)
            if self._best_train_loss is None or cur < self._best_train_loss:
                self._best_train_loss = cur
                if save_motion and hasattr(self.model, "motion_encoder"):
                    motion_best_payload = {
                        "phase": int(self.phase),
                        "epoch": cur_epoch,
                        "step": cur_step,
                        "best_train_loss": self._best_train_loss,
                        "state_dict": self.model.motion_encoder.state_dict(),
                    }
                    torch.save(
                        motion_best_payload,
                        os.path.join(module_dir, "motion_encoder_best.pt"),
                    )
                    torch.save(
                        motion_best_payload,
                        os.path.join(phase_dir, f"motion_encoder-best-epoch{cur_epoch:03d}-step{cur_step}.pt"),
                    )

                if save_mgdtr and hasattr(self.model, "selector") and self.model.selector is not None:
                    mgdtr_best_payload = {
                        "phase": int(self.phase),
                        "epoch": cur_epoch,
                        "step": cur_step,
                        "best_train_loss": self._best_train_loss,
                        "state_dict": self.model.selector.state_dict(),
                    }
                    torch.save(
                        mgdtr_best_payload,
                        os.path.join(module_dir, "mgdtr_selector_best.pt"),
                    )
                    torch.save(
                        mgdtr_best_payload,
                        os.path.join(phase_dir, f"mgdtr_selector-best-epoch{cur_epoch:03d}-step{cur_step}.pt"),
                    )

    def on_validation_epoch_start(self) -> None:
        if self.phase != 3:
            return

        self.batch_res = {"version": "VERSION 1.0",
                          "results": defaultdict(list),
                          "external_data": {"used": "true", "details": "ay"}}

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
        if self.phase != 3:
            return

        video = batch.get("video", batch)

        iframes = video.get("iframe", None)
        motion_vectors = video.get("motion_vector", video.get("motion_vectors", video.get("motion", None)))
        input_mask_mv = video.get("input_mask_mv", None)
        if motion_vectors is None:
            return
        if iframes is None:
            raise RuntimeError(
                "Phase 3 validation requires raw iframe batches (mp4 mode). "
                "Set ++model.use_preextracted_features=false"
            )
        if input_mask_mv is None:
            B, G, M = motion_vectors.shape[:3]
            input_mask_mv = torch.zeros((B, G, M), dtype=torch.long, device=motion_vectors.device)
        
        gen_sentences = []
        for i in range(motion_vectors.size(0)):
            sentence = self.model.generate(
                iframes=iframes[i:i+1],
                motion_vectors=motion_vectors[i:i+1],
                input_mask_mv=input_mask_mv[i:i+1],
                tokenizer=self.tokenizer,
                prompt="",
                min_new_tokens=4,
                max_new_tokens=30,
            )
            gen_sentences.append(sentence)

        for example_idx, (cur_gen_sen, cur_meta) in enumerate(zip(gen_sentences, batch['metadata'][1])):
            cur_data = {
                "sentence": cur_gen_sen,
                "gt_sentence": cur_meta
            }
            self.batch_res["results"][batch['metadata'][0][example_idx].split("video")[-1]].append(cur_data)

    def on_validation_epoch_end(self) -> None:
        if self.phase != 3:
            return

        json_res = copy.deepcopy(self.batch_res)
        if dist.is_initialized():
            all_results = gather_object_multiple_gpu(list(json_res["results"].items()))
            json_res['results'] = {k: v for k, v in all_results}
            logger.debug("Caption test length: %s", len(json_res["results"].items()))

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

            if not json_ref or not json_res.get("results"):
                logger.warning("Skipping caption metric evaluation: empty references or predictions.")
                return

            metrics = evaluate(json_res, json_ref)
            pretty_metrics = {
                k: (round(float(v), 4) if isinstance(v, (float, int)) else v)
                for k, v in metrics.items()
            }
            self.print(
                f"\n================ Validation Metrics ================\n"
                f"{pretty_metrics}\n"
                f"=================================================="
            )
            self.log_dict(metrics, on_step=False, on_epoch=True, logger=True, sync_dist=False)

        if dist.is_initialized():
            dist.barrier()


# =====================================================================
# HYDRA CONFIGURATION BUILDERS
# =====================================================================
from cocap.modules.compressed_video.motion_encoder import MotionTransformer

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

hav_cocap_cfg = builds(
    HavCoCapGPT2,
    gpt2_model_path=os.path.join(_BASE_DIR, "model_zoo", "gpt2_model"), 
    clip_state_dict=os.path.join(_BASE_DIR, "model_zoo", "clip_model", "ViT-B-16.pt"),
    motion_encoder=builds(MotionTransformer, populate_full_signature=True), 
    populate_full_signature=True
)

cocap_lm_cfg = builds(
    CoCapLM,
    cocap_model=hav_cocap_cfg, 
    loss=phase_aware_loss_cfg, 
    populate_full_signature=True
)