import gradio as gr
import json
import torch
import os
import sys
from pathlib import Path

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'HavCocap_new'))
# Need to set JAVA_HOME/PATH if testing uses temp_jre
os.environ["PATH"] = "/home/blaze/Hav-Cocap/HavCocap_new/temp_jre/jdk-11.0.2/bin:" + os.environ.get("PATH", "")

from hydra import initialize, compose
from hydra.utils import instantiate
from havcocap_new.modeling.lm_cocap import CoCapLM, convert_ids_to_sentence

# 1. Load video list
with open('selected_captions.json', 'r') as f:
    selected_data = json.load(f)

# Combine both good and bad for the dropdown
videos = selected_data['good_captions'] + selected_data['bad_captions']
video_options = [v['video_id'] for v in videos]
video_dict = {v['video_id']: v for v in videos}

# 2. Setup model and dataloader
print("Initializing configuration...")
with initialize(version_base=None, config_path="HavCocap_new/configs"):
    cfg = compose(config_name="train", overrides=["+exp/train=charades_captioning", "++val_dataloader.dataset.split=test", "++val_dataloader.batch_size=1"])

print("Instantiating validation dataloader...")
val_dataloader = instantiate(cfg.val_dataloader)

print("Loading model from checkpoint...")
ckpt_path = "HavCocap_new/logs/charades_captioning/lightning_logs/version_72/checkpoints/epoch=16-step=17153.ckpt"
model = CoCapLM.load_from_checkpoint(ckpt_path)
model.eval()
model.cuda()

# Create a mapping from video_id to its batch in the dataloader
# Running through the whole dataloader might be slow, so let's build an index or fetch on demand.
print("App ready!")

def predict_caption(video_id):
    # Search for the video batch in the dataloader
    print(f"Searching for video {video_id} in dataloader...")
    target_batch = None
    for batch in val_dataloader:
        if batch['metadata'][0][0].split("video")[-1] == video_id:
            target_batch = batch
            break
            
    if target_batch is None:
        return None, "Error: Video not found in dataloader", ""

    print(f"Running inference for {video_id}...")
    # Inference loop (from validation_step)
    # Move batch to GPU
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
    gt_caption = target_batch['metadata'][1][0]
    
    video_path = f"HavCocap_new/dataset/Charades/Charades_filtered_240/{video_id}.mp4"
    if not os.path.exists(video_path):
        video_path = None # Return None if video file not found
        
    return video_path, predicted_caption, gt_caption

def update_ui(video_id):
    vid_path, pred, gt = predict_caption(video_id)
    return vid_path, pred, gt

with gr.Blocks(title="Charades Captioning UI") as demo:
    gr.Markdown("# Hav-CoCap Video Captioning Evaluator")
    gr.Markdown(f"Loading checkpoint: `epoch=16-step=17153.ckpt`")
    
    with gr.Row():
        with gr.Column():
            video_dropdown = gr.Dropdown(choices=video_options, label="Select Video ID", value=video_options[0])
            run_btn = gr.Button("Generate Caption", variant="primary")
        with gr.Column():
            video_player = gr.Video(label="Input Video")
            gt_text = gr.Textbox(label="Ground Truth Caption", lines=2)
            pred_text = gr.Textbox(label="Predicted Caption", lines=2)
            
    run_btn.click(fn=update_ui, inputs=[video_dropdown], outputs=[video_player, pred_text, gt_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
