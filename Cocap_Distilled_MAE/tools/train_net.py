# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : train_net.py

import logging
from pathlib import Path
import sys
import os
from termcolor import colored

# Add the project root to sys.path so 'cocap' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
import torch
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping


from cocap.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def train(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    trainer: pl.Trainer,
    ckpt_path: str = None,
):
    try:
        from torchinfo import summary
        # Print a detailed hierarchical table of the model's architecture and parameters
        # `depth=5` controls how deeply nested modules are displayed
        model_summary = summary(model, depth=5, verbose=0, col_names=("num_params", "trainable"))
        print("\n" + "="*80)
        print("MODEL SUMMARY (includes Trainable & Evaluation/Non-trainable params)")
        print("="*80)
        print(model_summary)
        print("="*80 + "\n")
    except ImportError:
        logger.warning("torchinfo is not installed. Please `pip install torchinfo` to see the model summary.")
    except Exception as e:
        logger.warning(f"Could not print model summary: {e}")

    if ckpt_path is None:
        ckpt_path = os.environ.get("CKPT_PATH")

    # Phase-aware defaults for training length (driven by model.phase_hparams in config).
    phase = int(getattr(model, "phase", 0) or 0)
    phase_hparams = getattr(model, "phase_hparams", {}) or {}
    phase_cfg = None
    phase_key = f"phase{phase}"
    if hasattr(phase_hparams, "get"):
        phase_cfg = phase_hparams.get(phase_key, None)

    if phase_cfg is not None and hasattr(phase_cfg, "get"):
        # LR for phase1/2 is unified in optimizer code, so set model.lr from config profile.
        lr_cfg = phase_cfg.get("learning_rates", None)
        if lr_cfg is not None and hasattr(lr_cfg, "get"):
            if phase == 1:
                model.lr = float(lr_cfg.get("motion_student", getattr(model, "lr", 1.5e-4)))
            elif phase == 2:
                model.lr = float(lr_cfg.get("mgdtr_selector", getattr(model, "lr", 5e-5)))

        # Keep user overrides; adjust only when still at base default max_epochs.
        new_epochs = int(phase_cfg.get("max_epochs", getattr(trainer, "max_epochs", 100)))
        if getattr(trainer, "max_epochs", None) == 100 and phase in (2, 3) and new_epochs != 100:
            if hasattr(trainer, "fit_loop") and hasattr(trainer.fit_loop, "max_epochs"):
                trainer.fit_loop.max_epochs = new_epochs
            else:
                logger.warning("Could not set phase-aware max_epochs on this PL version.")

        new_patience = int(phase_cfg.get("early_stopping_patience", 7))
        for cb in trainer.callbacks:
            if isinstance(cb, EarlyStopping) and cb.monitor == "train_loss":
                cb.patience = new_patience

    # Phase policy: run caption metrics only in phase 3.
    if phase in (1, 2):
        logger.info("Phase %s: disabling validation/eval (train split only for this phase).", phase)
        val_dataloader = None

    # If val dataset is empty (e.g., filtered VATEX train-only subset), skip validation cleanly.
    val_len = None
    if val_dataloader is not None:
        try:
            val_dataset = getattr(val_dataloader, "dataset", None)
            if val_dataset is not None:
                val_len = len(val_dataset)
        except Exception:
            val_len = None
        if val_len == 0:
            logger.warning("Validation dataset is empty; disabling validation dataloader for this run.")
            val_dataloader = None

    # Cross-phase handoff safety:
    # If user passes ckpt_path from a different phase, do model-weight init only
    # (avoid optimizer/scheduler restore mismatch), and do NOT full-resume trainer state.
    if ckpt_path:
        phase = int(getattr(model, "phase", 0) or 0)
        ckpt_norm = ckpt_path.replace("\\", "/")
        same_phase = (f"/phase{phase}/" in ckpt_norm) or (f"phase{phase}/" in ckpt_norm)
        if phase in (2, 3) and not same_phase:
            if phase == 2 and hasattr(model, "init_motion_ckpt") and not getattr(model, "init_motion_ckpt"):
                model.init_motion_ckpt = ckpt_path
                info = "init_motion_ckpt"
            elif hasattr(model, "init_weights_ckpt") and not getattr(model, "init_weights_ckpt"):
                model.init_weights_ckpt = ckpt_path
                info = "init_weights_ckpt"
            else:
                info = "manual_init_already_set"
            print(colored(
                f"INFO: phase handoff detected (phase={phase}). Using ckpt as {info} and disabling full-state resume: {ckpt_path}",
                "yellow",
                attrs=["bold"],
            ))
            ckpt_path = None

    print(colored(f"DEBUG: Setting PL Trainer ckpt_path to: '{ckpt_path}'", "green", attrs=["bold"]))
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    store(
        train,
        model=cocap_lm_cfg,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            populate_full_signature=True,
        ),
        trainer=builds(pl.Trainer, gradient_clip_val=1.0, populate_full_signature=True),
        populate_full_signature=True,
        name="train",
    )
    store.add_to_hydra_store()

    zen(train).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        version_base=None,
    )
