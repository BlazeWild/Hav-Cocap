import os
import urllib.request
from pathlib import Path

def download_file(url, dest_path):
    print(f"Downloading {url.split('/')[-1]}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"✅ Successfully downloaded to {dest_path.name}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def main():
    # 1. Dynamically get the folder where this script lives (which is your model_zoo folder)
    script_dir = Path(__file__).parent.absolute()
    
    # 2. Build the correct target path inside it
    target_dir = script_dir / "audio_model" / "vggish"
    target_dir.mkdir(parents=True, exist_ok=True)

    # Official torchvggish release URLs
    weights_url = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"
    pca_url = "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish_pca_params-970ea276.pth"

    weights_dest = target_dir / "vggish-10086976.pth"
    pca_dest = target_dir / "vggish_pca_params-970ea276.pth"

    if not weights_dest.exists():
        download_file(weights_url, weights_dest)
    else:
        print("✅ VGGish weights already exist.")

    if not pca_dest.exists():
        download_file(pca_url, pca_dest)
    else:
        print("✅ VGGish PCA params already exist.")

if __name__ == "__main__":
    main()
