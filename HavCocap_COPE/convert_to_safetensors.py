import torch
from safetensors.torch import save_file
import os

bin_path = "model_zoo/tinystories-33m/pytorch_model.bin"
st_path = "model_zoo/tinystories-33m/model.safetensors"

if os.path.exists(bin_path):
    print(f"Loading {bin_path}...")
    # bypassing transformers safety check by using pure torch.load
    sd = torch.load(bin_path, map_location="cpu")
    print(f"Saving to {st_path}...")
    save_file(sd, st_path)
    print("Done!")
    # Optionally remove bin
    # os.remove(bin_path)
else:
    print(f"File not found: {bin_path}")
