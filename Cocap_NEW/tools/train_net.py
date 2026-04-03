# -*- coding: utf-8 -*-
# @Time    : 6/17/25
# @Author  : Yaojie Shen
# @Project : CoCap
# @File    : train_net.py

import logging
from pathlib import Path
import sys
from termcolor import colored

# Add the project root to sys.path so 'cocap' can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytorch_lightning as pl
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader


from cocap.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)


def train(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer
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

    # Retrieve ckpt_path directly from Hydra config before passing to trainer kwargs mapping
    ckpt_path = None
    if "+ckpt_path" in sys.argv:
        try:
            ckpt_path = sys.argv[sys.argv.index("+ckpt_path") + 1].strip('"\'')
        except ValueError:
            pass
    elif "++ckpt_path" in sys.argv:
        try:
            ckpt_path = sys.argv[sys.argv.index("++ckpt_path") + 1].strip('"\'')
        except (ValueError, IndexError):
            pass
    
    for arg in sys.argv:
        if arg.startswith("++ckpt_path="):
            # Split only on the first '=' in case there are '=' symbols in the checkpoint name itself 
            ckpt_path = arg.split("=", 1)[1].strip('"\'')
        elif arg.startswith("+ckpt_path="):
            # Split only on the first '=' in case there are '=' symbols in the checkpoint name itself
            ckpt_path = arg.split("=", 1)[1].strip('"\'')
            
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
