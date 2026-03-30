import gradio as gr
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

def run_gradio(
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
        print(f"Warning: Checkpoint not found at {ckpt_path}")
    
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    def process_video(video_path):
        if not video_path:
            return "No video uploaded."
        
        basename = os.path.basename(video_path)
        video_id = os.path.splitext(basename)[0]
        
        # User uploaded the video, for charades it might be A3MOW_something in Gradio tmp folder
        # So we better make sure we just match the base ID ignoring tmp folder hashes if possible
        # For instance, /tmp/gradio/123910/A3MOW.mp4
        
        print(f"Searching for video {video_id} in val_dataloader...")
        target_batch = None
        for batch in val_dataloader:
            if batch['metadata'][0][0].split("video")[-1] == video_id:
                target_batch = batch
                break
                
        if target_batch is None:
            return f"Error: Video ID '{video_id}' not found in val_dataloader."

        print(f"Running inference for {video_id}...")
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in target_batch.items()}
        
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
        return predicted_caption

    with gr.Blocks(title="Charades Live Inference UI") as demo:
        gr.Markdown("# Hav-CoCap Live Video Captioning")
        gr.Markdown(f"Upload a Charades `.mp4` video (e.g., `A3MOW.mp4`). The system will parse the filename, locate its features via dataloader, and predict a caption dynamically using `{ckpt_path}`.")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                run_btn = gr.Button("Generate Caption", variant="primary")
            with gr.Column():
                pred_text = gr.Textbox(label="Predicted Caption", lines=2)
                
        run_btn.click(fn=process_video, inputs=[video_input], outputs=[pred_text])
        
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == '__main__':
    store(
        run_gradio,
        model=cocap_lm_cfg,
        train_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=1,
            num_workers=2,
            populate_full_signature=True,
        ),
        val_dataloader=builds(
            DataLoader,
            dataset=MISSING,
            batch_size=1,
            num_workers=2,
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

    zen(run_gradio).hydra_main(
        config_path=(Path(__file__).parent / "HavCocap_new" / "configs").as_posix(),
        config_name="train",
        version_base=None,
    )
