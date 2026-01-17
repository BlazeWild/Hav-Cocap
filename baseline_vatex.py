import gradio as gr
import cv2
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np

# --- Configuration ---
# 'microsoft/git-base-vatex' is fine-tuned for video captioning.
# It is lightweight (base size) and runs reasonably fast on CPU or low-end GPUs.
MODEL_NAME = "microsoft/git-base-vatex"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_NAME} on {DEVICE}...")

# --- Load Model & Processor ---
try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def sample_frames(video_path, num_frames=6):
    """
    Extracts 'num_frames' evenly spaced frames from the video.
    The GIT model specifically expects 6 frames for video tasks.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If video is too short, just take what we have, otherwise calculate stride
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL/Transformers)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    cap.release()
    
    # Pad if we couldn't get enough frames (rare edge case for corrupted videos)
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])
        
    return frames

def generate_caption(video_path):
    """
    Main function to process video and generate caption.
    """
    if video_path is None:
        return "Please upload a video."

    try:
        # 1. Extract frames
        frames = sample_frames(video_path, num_frames=6)
        if not frames:
            return "Error: Could not extract frames from video."

        # 2. Preprocess inputs
        # The processor automatically handles resizing and normalizing
        inputs = processor(images=frames, return_tensors="pt").to(DEVICE)

        # 3. Generate caption
        # pixel_values shape: [batch_size, num_frames, 3, height, width]
        generated_ids = model.generate(pixel_values=inputs.pixel_values, max_length=50)

        # 4. Decode output
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_caption

    except Exception as e:
        return f"An error occurred: {str(e)}"

# --- Gradio UI Setup ---
with gr.Blocks(title="Simple Video Captioning") as demo:
    gr.Markdown("## 🎥 Video Captioning baseline performance test")
    
    with gr.Row():
        with gr.Column():
            # Input: Video uploader
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            submit_btn = gr.Button("Generate Caption", variant="primary")
        
        with gr.Column():
            # Output: Text caption
            output_text = gr.Textbox(label="Generated Caption", lines=3)
            
    # Connect the button to the function
    submit_btn.click(fn=generate_caption, inputs=video_input, outputs=output_text)
    
    # Example usage instructions
    gr.Markdown("### Notes:\n- Processing time depends on your hardware (CPU vs GPU).")

if __name__ == "__main__":
    demo.launch()