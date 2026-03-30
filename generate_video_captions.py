import json
import torch
import os
import sys
from pathlib import Path
import pytorch_lightning as pl

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'HavCocap_new'))
os.environ["PATH"] = "/home/blaze/Hav-Cocap/HavCocap_new/temp_jre/jdk-11.0.2/bin:" + os.environ.get("PATH", "")

from hydra_zen import builds, store, zen
from omegaconf import MISSING
from torch.utils.data import DataLoader
from havcocap_new.modeling.lm_cocap import cocap_lm_cfg, convert_ids_to_sentence

VIDEO_IDS = ["3W6CP", "5XAMJ", "IQM7A", "IZTHW", "L76ND", "9MXDX", "XFLQH", "A3MOW"]

def generate_json(
        model: pl.LightningModule,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        trainer: pl.Trainer,
        ckpt_path: str = None
):
    print("Loading model from checkpoint...")
    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Checkpoint loaded.")
    else:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    results = []
    
    print("Iterating over val_dataloader to find and process target videos...")
    for batch_idx, batch in enumerate(val_dataloader):
        video_id = batch['metadata'][0][0].split("video")[-1]
        
        if video_id in VIDEO_IDS:
            print(f"Found {video_id}, running inference...")
            
            # Move to GPU
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            inputs_ids = batch["input_ids"]
            input_masks = batch["input_mask"]
            max_t_len = model.model.caption_head.cap_config.max_t_len
            
            inputs_ids[:, :] = 0.
            input_masks[:, :] = 0.
            
            bsz = len(inputs_ids)
            next_symbols = torch.IntTensor([model.model.caption_head.cap_config.BOS_id] * bsz)
            
            with torch.no_grad():
                for dec_idx in range(max_t_len):
                    inputs_ids[:, dec_idx] = next_symbols.cuda().clone()
                    input_masks[:, dec_idx] = 1
                    outputs = model.model(batch)
                    pred_scores = outputs["prediction_scores"]
                    next_words = pred_scores[:, dec_idx].max(1)[1]
                    next_symbols = next_words.cpu()
                    if "visual_output" in outputs:
                        batch["visual_output"] = outputs["visual_output"]

            predicted_caption = convert_ids_to_sentence(inputs_ids[0].cpu().tolist())
            gt_caption = batch['metadata'][1][0]
            
            results.append({
                "video_id": video_id,
                "predicted_caption": predicted_caption,
                "gt_caption": gt_caption
            })
            
            # Stop early if we found all
            if len(results) == len(VIDEO_IDS):
                break

    output_file = "../epoch16_captions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Done! Saved {len(results)} captions to {output_file}")


if __name__ == '__main__':
    store(
        generate_json,
        model=cocap_lm_cfg,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=1,
            num_workers=4,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=1,
            num_workers=4,
            populate_full_signature=True,
        ),
        trainer=builds(pl.Trainer, populate_full_signature=True),
        populate_full_signature=True,
        name="train",
    )
    store.add_to_hydra_store()

    if not any(arg.startswith("+exp/train=") for arg in sys.argv):
        sys.argv.append("+exp/train=charades_captioning")
    if not any(arg.startswith("++val_dataloader.dataset.split=") for arg in sys.argv):
        sys.argv.append("++val_dataloader.dataset.split=test")
    if not any(arg.startswith("ckpt_path=") for arg in sys.argv):
        sys.argv.append('ckpt_path="logs/charades_captioning/lightning_logs/version_72/checkpoints/epoch=16-step=17153.ckpt"')

    zen(generate_json).hydra_main(
        config_path=(Path(__file__).parent / "HavCocap_new" / "configs").as_posix(),
        config_name="train",
        version_base=None,
    )
