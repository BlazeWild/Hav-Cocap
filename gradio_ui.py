import gradio as gr
import json
import torch
import os
import sys

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'HavCocap_new'))
from havcocap_new.modeling.lm_cocap import CoCapLM

# Load data
print("Loading selected captions...")
with open('selected_captions.json', 'r') as f:
    selected_data = json.load(f)

# Combine both good and bad for the dropdown
videos = selected_data['good_captions'] + selected_data['bad_captions']
video_options = [v['video_id'] for v in videos]
video_dict = {v['video_id']: v for v in videos}

# Checkpoint path
ckpt_path = "HavCocap_new/logs/charades_captioning/lightning_logs/version_72/checkpoints/epoch=16-step=17153.ckpt"

print(f"Loading checkpoint from {ckpt_path}...")
try:
    # Load the checkpoint as requested by user
    model = CoCapLM.load_from_checkpoint(ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    model_loaded = True
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False

def predict_caption(video_id):
    # As instructed: "there is already caption so uyo write both the predicted and gt captions, see from these selected_captions.json"
    
    video_path = f"HavCocap_new/dataset/Charades/Charades_filtered_240/{video_id}.mp4"
    if not os.path.exists(video_path):
        video_path = None # Gradio will show an error or empty block if None
    else:
        video_path = os.path.abspath(video_path)
        
    data = video_dict.get(video_id, {})
    predicted_caption = data.get("sentence", "No prediction available.")
    gt_caption = data.get("gt_sentence", "No GT available.")
        
    return video_path, predicted_caption, gt_caption

with gr.Blocks(title="Charades Captioning UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Hav-CoCap Video Captioning Evaluator")
    gr.Markdown(f"**Loaded Checkpoint:** `{ckpt_path}`")
    
    if not model_loaded:
        gr.Markdown("*Warning: Model checkpoint could not be loaded into memory, using cached predictions.*", elem_id="warning")
        
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select a Video")
            video_dropdown = gr.Dropdown(choices=video_options, label="Video ID", value=video_options[0])
            run_btn = gr.Button("Analyze Video", variant="primary")
            
        with gr.Column(scale=2):
            video_player = gr.Video(label="Input Video (Charades_filtered_240)")
            pred_text = gr.Textbox(label="Predicted Caption", lines=2, interactive=False)
            gt_text = gr.Textbox(label="Ground Truth Caption", lines=2, interactive=False)
            
    run_btn.click(fn=predict_caption, inputs=[video_dropdown], outputs=[video_player, pred_text, gt_text])
    
    # Enable automatic population on change
    video_dropdown.change(fn=predict_caption, inputs=[video_dropdown], outputs=[video_player, pred_text, gt_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, allowed_paths=[os.path.abspath("HavCocap_new/dataset/Charades/Charades_filtered_240")])
