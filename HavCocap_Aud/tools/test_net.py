import logging
from pathlib import Path
import os
os.environ["PATH"] = "/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new/temp_jre/jdk-11.0.2/bin:" + os.environ.get("PATH", "")
import pytorch_lightning as pl
from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from havcocap_new.modeling.lm_cocap import cocap_lm_cfg

logger = logging.getLogger(__name__)

def test(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer,
        ckpt_path: str = None
):
    # Testing mode using val_dataloader which will be pointed to the test split
    trainer.validate(model=model, dataloaders=val_dataloader, ckpt_path=ckpt_path)

if __name__ == '__main__':
    store(
        test,
        model=cocap_lm_cfg,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=2,
            num_workers=6,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=2,
            num_workers=6,
            populate_full_signature=True,
        ),
        trainer=builds(pl.Trainer, populate_full_signature=True),
        populate_full_signature=True,
        name="train", # Keep this as 'train' so it acts as the base configuration for yaml files that inherit from /train
    )
    store.add_to_hydra_store()

    import sys
    # Make sure we use charades captioning correctly, and set the split to test
    if not any(arg.startswith("+exp/train=") for arg in sys.argv):
        sys.argv.append("+exp/train=charades_captioning")
    if not any(arg.startswith("++val_dataloader.dataset.split=") for arg in sys.argv):
        sys.argv.append("++val_dataloader.dataset.split=test")
    if not any(arg.startswith("++val_dataloader.batch_size=") for arg in sys.argv):
        sys.argv.append("++val_dataloader.batch_size=2")
    if not any(arg.startswith("++val_dataloader.num_workers=") for arg in sys.argv):
        sys.argv.append("++val_dataloader.num_workers=6")
    if not any(arg.startswith("++train_dataloader.batch_size=") for arg in sys.argv):
        sys.argv.append("++train_dataloader.batch_size=2")
    if not any(arg.startswith("++train_dataloader.num_workers=") for arg in sys.argv):
        sys.argv.append("++train_dataloader.num_workers=6")
    if not any(arg.startswith("ckpt_path=") for arg in sys.argv):
        sys.argv.append("ckpt_path=HavCocap_new/logs/charades_captioning/lightning_logs/version_58/checkpoints/best_epoch_11.ckpt")

    zen(test).hydra_main(
        config_path=(Path(__file__).parent.parent / "configs").as_posix(),
        config_name="train",
        version_base=None,
    )
