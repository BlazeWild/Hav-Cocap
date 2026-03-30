import gradio as gr
import json
import torch
import os
import sys
from pathlib import Path
import pytorch_lightning as pl

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
havcocap_new_dir = os.path.join(base_dir, 'HavCocap_new')
sys.path.append(havcocap_new_dir)
os.chdir(havcocap_new_dir) # Change working directory so relative paths in yaml configs resolve correctly
os.environ["PATH"] = os.path.join(havcocap_new_dir, "temp_jre/jdk-11.0.2/bin") + ":" + os.environ.get("PATH", "")

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
            return "No video uploaded.", gr.update(visible=False)
        
        print(f"Processing uploaded video: {video_path}")
        
        try:
            dataset = val_dataloader.dataset
            from havcocap_new.data.datasets.compressed_video.video_text_base import get_video, extract_audio_from_video
            
            video, video_mask = get_video(
                video_reader=dataset.video_reader,
                video_path=video_path,
                max_frames=dataset.max_frames,
                sample="uniform",
                hevc_config=dataset.h265_cfg
            )
            
            audio = extract_audio_from_video(video_path)
            
            if dataset.transform is not None:
                video = dataset.transform(video)
                
            # The 'video' returned by the transform is a dictionary of modality tensors
            if isinstance(video, dict):
                video_batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in video.items()}
            else:
                video_batch = video.unsqueeze(0)
                
            # Create a mock batch of size 1
            max_words = dataset.max_words
            input_ids = torch.zeros(max_words, dtype=torch.long)
            # BOS
            input_ids[0] = model.model.caption_head.cap_config.BOS_id
            input_mask = torch.zeros(max_words, dtype=torch.long)
            input_mask[0] = 1
            
            target_batch = {
                "video": video_batch,
                "video_mask": video_mask.unsqueeze(0),
                "audio": audio.unsqueeze(0) if audio is not None else None,
                "input_ids": input_ids.unsqueeze(0),
                "input_mask": input_mask.unsqueeze(0),
                "metadata": [["upload"], [""]] 
            }
            
            print("Extracted features successfully. Running inference...")
            
            # Recursively move tensors to CUDA
            def to_cuda(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cuda()
                elif isinstance(obj, dict):
                    return {k: to_cuda(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_cuda(x) for x in obj]
                else:
                    return obj
                    
            batch = to_cuda(target_batch)
            
            max_t_len = model.model.caption_head.cap_config.max_t_len
            bsz = 1
            next_symbols = torch.IntTensor([model.model.caption_head.cap_config.BOS_id] * bsz)
            
            with torch.no_grad():
                for dec_idx in range(max_t_len):
                    batch["input_ids"][:, dec_idx] = next_symbols.cuda().clone()
                    batch["input_mask"][:, dec_idx] = 1
                    outputs = model.model(batch)
                    pred_scores = outputs["prediction_scores"]
                    next_words = pred_scores[:, dec_idx].max(1)[1]
                    next_symbols = next_words.cpu()
                    if "visual_output" in outputs:
                        batch["visual_output"] = outputs["visual_output"]

            predicted_caption = convert_ids_to_sentence(batch["input_ids"][0].cpu().tolist())
            return predicted_caption, gr.update(value=f"### {predicted_caption}", visible=True)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            err_msg = f"Error processing video: {str(e)}"
            return err_msg, gr.update(value=f"### ❌ {err_msg}", visible=True)

    # Premium Dark Theme
    custom_theme = gr.themes.Monochrome(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
        body_text_color="white",
        block_background_fill="rgba(30, 41, 59, 0.7)",
        block_border_width="1px",
        block_border_color="rgba(255, 255, 255, 0.1)",
        block_radius="12px",
        button_primary_background_fill="linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #2563eb 0%, #7c3aed 100%)",
        button_primary_text_color="white",
    )

    custom_css = """
    .gradio-container { border-radius: 12px; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); padding: 20px; backdrop-filter: blur(10px); }
    .generated-caption { font-size: 1.25rem; font-weight: 500; text-align: center; color: #a78bfa; padding: 15px; background: rgba(0,0,0,0.3); border-radius: 8px; margin-top: 10px; }
    h1 { text-align: center; background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    """

    with gr.Blocks(theme=custom_theme, css=custom_css, title="Hav-CoCap L4 GPU Inference") as demo:
        gr.Markdown("# Hav-CoCap Video Captioning")
        gr.Markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.1rem;'>Upload a video and our L4 GPU will generate a highly accurate caption instantly.</p>")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Upload Video File", interactive=True)
                run_btn = gr.Button("Generate Caption 🚀", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # The text box is still used programmatically but we display a nicer markdown block
                pred_text_hidden = gr.Textbox(visible=False) 
                pred_display = gr.Markdown("### Generated Caption will appear here...", elem_classes="generated-caption", visible=False)
                
        # Update the hidden textbox first, then the markdown display
        run_btn.click(
            fn=process_video, 
            inputs=[video_input], 
            outputs=[pred_text_hidden, pred_display]
        )
        
    # Bind to 0.0.0.0 for GCP external access.
    # Note: If you want to use port 80 (HTTP) directly, you may need to run this script with `sudo` 
    # e.g., `sudo /home/blaze/Hav-Cocap/venv/bin/python run_web.py` and change server_port below to 80.
    # Otherwise, you can use port 8080 or port 7860 depending on your GCP firewall rules.
    demo.launch(server_name="0.0.0.0", server_port=80, share=False)

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
