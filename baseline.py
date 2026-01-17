import gradio as gr
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np

# --- Configuration ---
# 'microsoft/git-base-msrvtt' is the standard CLIP-based generative baseline for MSR-VTT
MODEL_NAME = "microsoft/git-base-msrvtt-captioning"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP-based Baseline: {MODEL_NAME} on {DEVICE}...")

# --- Load Model & Processor ---
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Falling back to git-base.")
    MODEL_NAME = "microsoft/git-base"
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

def sample_frames(video_path, num_frames=6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def generate_caption(video_path):
    if video_path is None:
        return "Please upload a video."
    try:
        # CLIP4Caption logic usually takes 6-12 frames
        frames = sample_frames(video_path, num_frames=6)
        inputs = processor(images=frames, return_tensors="pt").to(DEVICE)
        
        # Generation with beam search for better results
        generated_ids = model.generate(
            pixel_values=inputs.pixel_values, 
            max_length=30,
            num_beams=5,
            no_repeat_ngram_size=2
        )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown(f"## 🎥 MSR-VTT Baseline: CLIP-based Captioning")
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        output_text = gr.Textbox(label="Generated Caption")
    submit_btn = gr.Button("Run Baseline Inference", variant="primary")
    submit_btn.click(fn=generate_caption, inputs=video_input, outputs=output_text)

if __name__ == "__main__":
    demo.launch()