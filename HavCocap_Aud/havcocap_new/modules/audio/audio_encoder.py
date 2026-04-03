import torch
import torch.nn as nn
import os
from pathlib import Path

class VGGishAudioEncoder(nn.Module):
    def __init__(
        self, 
        output_dim=768, 
        embed_dim=128,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        
        # --- BULLETPROOF PATH RESOLUTION ---
        # 1. Find where this audio_encoder.py file is located
        current_file_dir = Path(__file__).parent.absolute()
        
        # 2. Jump up to the project root (assuming modules -> havcocap_new -> root)
        # Adjust the number of .parents if your folder structure is deeper
        project_root = current_file_dir.parent.parent.parent
        
        primary_vggish_dir = project_root / "model_zoo" / "audio_model" / "vggish"
        fallback_vggish_dir = project_root / "model_zoo" / "model_zoo" / "audio_model" / "vggish"

        vggish_dir = primary_vggish_dir if primary_vggish_dir.exists() else fallback_vggish_dir

        self.model_path = vggish_dir / "vggish-10086976.pth"
        self.pca_path = vggish_dir / "vggish_pca_params-970ea276.pth"
        
        # 1. Load VGGish locally
        self.vggish = self._load_vggish_model()
        
        # 2. Completely freeze VGGish parameters
        if self.vggish is not None:
            self.vggish.eval()
            for param in self.vggish.parameters():
                param.requires_grad = False
                
        # 3. The trainable Projection Layer
        self.projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.ReLU(True),
            nn.LayerNorm(self.output_dim, eps=1e-12)
        )

    def _load_vggish_model(self):
        """
        Loads the VGGish architecture from the Hub, but injects our local weights.
        """
        if not os.path.exists(self.model_path):
            print(f"⚠️ Warning: Model weights not found at {self.model_path}. Run download_vggish.py first.")
            return None

        try:
            # 1. Load the architecture blueprint ONLY (no downloading weights)
            model = torch.hub.load('harritaylor/torchvggish', 'vggish', pretrained=False)
            
            # 2. Inject our local, pre-downloaded weights
            model.load_state_dict(torch.load(self.model_path))
            
            # 3. Load the local PCA parameters (required for the exact 128-d math)
            model.postprocess.load_params(self.pca_path)
            
            print("🚀 Successfully loaded local VGGish weights from model_zoo!")
            return model
            
        except Exception as e:
            print(f"⚠️ Error loading local VGGish: {e}")
            return None

    def forward(self, audio_tensor):
        """
        Input: [Batch, Num_GOPs, 16000]
        Output: [Batch, Num_GOPs, output_dim]
        """
        batch_size, num_gop, num_samples = audio_tensor.shape
        
        if self.vggish is None:
            fake_features = torch.randn(batch_size, num_gop, self.embed_dim, device=audio_tensor.device)
            return self.projection(fake_features)

        flat_audio = audio_tensor.view(-1, num_samples)
        
        self.vggish.eval()
        with torch.no_grad():
            features = self.vggish(flat_audio) 
            
        projected = self.projection(features)
        projected_tokens = projected.view(batch_size, num_gop, self.output_dim)
        
        return projected_tokens