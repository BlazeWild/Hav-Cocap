import torch
import torch.nn as nn
import os
import sys
from pathlib import Path

class BEATsAudioEncoder(nn.Module):
    def __init__(
        self, 
        output_dim=768, 
        embed_dim=768,
        model_path="BEATs_iter3_plus_AS2M.pt",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.model_path = model_path
        
        # --- BULLETPROOF PATH RESOLUTION ---
        # 1. Find where this audio_encoder.py file is located
        current_file_dir = Path(__file__).parent.absolute()
        
        # 2. Jump up to the project root (assuming modules -> havcocap_new -> root)
        # Adjust the number of .parents if your folder structure is deeper
        project_root = current_file_dir.parent.parent.parent
        self.project_root = project_root
        
        primary_beats_path = project_root / "model_zoo" / self.model_path
        fallback_beats_path = project_root / self.model_path
        self.beats_ckpt_path = primary_beats_path if primary_beats_path.exists() else fallback_beats_path
        
        # 1. Load BEATs locally
        self.beats = self._load_beats_model()
        
        # 2. Completely freeze BEATs backbone parameters
        if self.beats is not None:
            self.beats.eval()
            for param in self.beats.parameters():
                param.requires_grad = False
                
        # 3. Projection to caption hidden size
        self.projection = nn.Sequential(
            nn.LayerNorm(self.embed_dim, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, self.output_dim),
            nn.ReLU(True),
            nn.LayerNorm(self.output_dim, eps=1e-12)
        )

    def _load_beats_model(self):
        """Load BEATs checkpoint if BEATs package is available in environment."""
        if not os.path.exists(self.beats_ckpt_path):
            print(f"⚠️ Warning: BEATs checkpoint not found at {self.beats_ckpt_path}. Run model_zoo/download_beats.sh first.")
            return None

        # Try to discover local BEATs.py/module in common places and add to PYTHONPATH at runtime.
        search_roots = [self.project_root, self.project_root / "model_zoo", self.project_root / "third_party"]
        for root in search_roots:
            if not root.exists():
                continue
            beats_files = list(root.rglob("BEATs.py"))
            if beats_files:
                beats_parent = str(beats_files[0].parent)
                if beats_parent not in sys.path:
                    sys.path.insert(0, beats_parent)
                break

        try:
            # Official BEATs API (if installed):
            # pip install git+https://github.com/microsoft/unilm.git#subdirectory=beats
            try:
                from BEATs import BEATs, BEATsConfig
            except Exception:
                # Some installs expose lowercase package path.
                from beats.BEATs import BEATs, BEATsConfig

            checkpoint = torch.load(self.beats_ckpt_path, map_location="cpu")
            cfg = BEATsConfig(checkpoint["cfg"])
            model = BEATs(cfg)
            model.load_state_dict(checkpoint["model"], strict=False)

            print(f"🚀 Successfully loaded BEATs checkpoint from {self.beats_ckpt_path}")
            return model
        except Exception as e:
            print(f"⚠️ Error loading BEATs model: {e}")
            return None

    def forward(self, audio_tensor):
        """
        Input: [Batch, Num_GOPs, 16000]
        Output: [Batch, Num_GOPs, output_dim]
        """
        batch_size, num_gop, num_samples = audio_tensor.shape
        
        if self.beats is None:
            # Deterministic safe fallback when BEATs is unavailable.
            zero_features = torch.zeros(batch_size, num_gop, self.embed_dim, device=audio_tensor.device)
            return self.projection(zero_features)

        flat_audio = audio_tensor.view(-1, num_samples)
        
        self.beats.eval()
        with torch.no_grad():
            if hasattr(self.beats, "extract_features"):
                features, _ = self.beats.extract_features(flat_audio)
            else:
                features = self.beats(flat_audio)

            if isinstance(features, (tuple, list)):
                features = features[0]

            if features.dim() == 2:
                pooled = features
            else:
                pooled = features.mean(dim=1)

        projected = self.projection(pooled)
        projected_tokens = projected.view(batch_size, num_gop, self.output_dim)
        
        return projected_tokens


# Backward compatibility for older imports.
VGGishAudioEncoder = BEATsAudioEncoder
