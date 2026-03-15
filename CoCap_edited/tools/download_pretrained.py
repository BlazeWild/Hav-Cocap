import os
import requests
import torch
from tqdm import tqdm

MODEL_ZOO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_zoo")
os.makedirs(MODEL_ZOO_DIR, exist_ok=True)

MODELS = {
    "RN50.pt": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "Cnn14_mAP=0.431.pth": "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1",
    "BEATs_iter3_plus_AS2M.pt": "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=r&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXjglVPk%3D"
}

def download_file(url, filename):
    filepath = os.path.join(MODEL_ZOO_DIR, filename)
    if os.path.exists(filepath):
        print(f"{filename} already exists at {filepath}")
        return filepath
    
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)
            
    print(f"Downloaded {filepath}")
    return filepath

if __name__ == "__main__":
    print(f"Downloading models to {MODEL_ZOO_DIR}...")
    for name, url in MODELS.items():
        download_file(url, name)
