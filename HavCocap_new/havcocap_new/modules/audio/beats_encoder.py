import importlib
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class BEATsAudioEncoder(nn.Module):
    def __init__(self, model_path="BEATs_iter3_plus_AS2M.pt", output_dim=512, embed_dim=768):
        super().__init__()
        self.model_path = str(model_path)
        self.output_dim = output_dim
        self.embed_dim = embed_dim

        # Resolve project root and likely checkpoint locations.
        self.project_root = Path(__file__).resolve().parents[3]
        candidate_ckpts = [
            Path(self.model_path),
            self.project_root / "model_zoo" / self.model_path,
            self.project_root / self.model_path,
            self.project_root / "model_zoo" / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            self.project_root / "BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
        ]
        self.beats_ckpt_path = next((p for p in candidate_ckpts if p.exists()), candidate_ckpts[1])
        
        # Load BEATs
        self.beats = self._load_beats_model()
        
        # Freeze BEATs parameters
        if self.beats is not None:
            self.beats.eval()
            for param in self.beats.parameters():
                param.requires_grad = False
                
        # Projection layer: mapping from BEATs hidden (typically 768) to video features (typically 512)
        self.projection = nn.Linear(self.embed_dim, self.output_dim)

    def _import_beats_classes(self):
        """Try multiple import paths to locate BEATs implementation."""
        # 0) pip-installed BEATs package variants
        try:
            from BEATs import BEATs, BEATsConfig
            return BEATs, BEATsConfig
        except Exception:
            pass

        try:
            from beats.BEATs import BEATs, BEATsConfig
            return BEATs, BEATsConfig
        except Exception:
            pass

        # 1) preferred local package path (if vendored into this project)
        try:
            from havcocap_new.modules.audio.beats.BEATs import BEATs, BEATsConfig
            return BEATs, BEATsConfig
        except Exception:
            pass

        # 2) alternative package path used by sibling repos
        try:
            from cocap.modules.beats.BEATs import BEATs, BEATsConfig
            return BEATs, BEATsConfig
        except Exception:
            pass

        # 3) import from sibling source trees by adding the package root to sys.path.
        # NOTE: importing BEATs.py directly via importlib.spec fails because BEATs.py
        # uses relative imports (e.g. from .backbone import ...).
        candidate_pkg_roots = [
            self.project_root.parent / "CoCap_edited" / "cocap" / "modules",
            self.project_root.parent / "Hav-Cocap" / "Havcocap" / "modules",
        ]

        for pkg_root in candidate_pkg_roots:
            beats_py = pkg_root / "beats" / "BEATs.py"
            if not beats_py.exists():
                continue

            pkg_root_str = str(pkg_root)
            if pkg_root_str not in sys.path:
                sys.path.insert(0, pkg_root_str)

            try:
                module = importlib.import_module("beats.BEATs")
                if hasattr(module, "BEATs") and hasattr(module, "BEATsConfig"):
                    return module.BEATs, module.BEATsConfig
            except Exception:
                continue

        return None, None

    def _load_beats_model(self):
        if not self.beats_ckpt_path.exists():
            logger.warning(
                "BEATs checkpoint not found at %s. Audio backbone will be disabled.",
                self.beats_ckpt_path
            )
            return None

        BEATs, BEATsConfig = self._import_beats_classes()
        if BEATs is None or BEATsConfig is None:
            logger.warning(
                "BEATs python module not found. Install BEATs or vendor BEATs.py/backbone.py/modules.py. "
                "Audio backbone will be disabled."
            )
            return None

        try:
            checkpoint = torch.load(self.beats_ckpt_path, map_location="cpu")
            cfg_dict = checkpoint.get("cfg", checkpoint.get("args", {}))
            if not isinstance(cfg_dict, dict):
                # Namespace/dataclass-like checkpoints
                cfg_dict = vars(cfg_dict)
            model = BEATs(BEATsConfig(cfg_dict))
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=False)
            # Some released checkpoints are classifier-finetuned and return
            # class logits (e.g., 527) instead of encoder embeddings.
            # We always need encoder features for captioning.
            if hasattr(model, "predictor"):
                model.predictor = None
            if hasattr(model, "cfg") and hasattr(model.cfg, "finetuned_model"):
                model.cfg.finetuned_model = False
            logger.info("Loaded BEATs checkpoint from %s", self.beats_ckpt_path)
            return model
        except Exception as e:
            logger.warning("Failed to load BEATs from %s: %s", self.beats_ckpt_path, e)
        return None

    def forward(self, audio_tensor):
        """
        Input: audio_tensor shape [Batch, samples] or [Batch, num_gop, samples]
        Output: [Batch, tokens, output_dim] if 2D input, or [Batch, num_gop, output_dim] if 3D input.
        """
        if audio_tensor.dim() == 3:
            bsz, n_gop, n_samples = audio_tensor.shape
            flat_audio = audio_tensor.reshape(bsz * n_gop, n_samples)
        elif audio_tensor.dim() == 2:
            bsz = audio_tensor.shape[0]
            n_gop = None
            flat_audio = audio_tensor
        else:
            raise ValueError(f"audio_tensor must be 2D or 3D, got shape={tuple(audio_tensor.shape)}")

        if self.beats is None:
            # Deterministic fallback to keep the pipeline alive without injecting random noise.
            if n_gop is None:
                fake_features = torch.zeros(bsz, 1, self.embed_dim, device=audio_tensor.device)
                return self.projection(fake_features)
            fake_features = torch.zeros(bsz, n_gop, self.embed_dim, device=audio_tensor.device)
            return self.projection(fake_features)

        self.beats.eval()
        with torch.no_grad():
            if hasattr(self.beats, "extract_features"):
                features, _ = self.beats.extract_features(flat_audio)
            else:
                features = self.beats(flat_audio)

            if isinstance(features, (tuple, list)):
                features = features[0]

        if features.dim() == 3:
            # [B, T, C] -> global pooled [B, C]
            features = features.mean(dim=1)

        projected = self.projection(features)

        if n_gop is not None:
            projected = projected.view(bsz, n_gop, self.output_dim)
        return projected
        
    def reshape_for_gop(self, audio_tokens, num_gop=16):
        """
        Reshape the sequence of tokens to group them by GOP bins.
        """
        if audio_tokens.dim() == 3 and audio_tokens.size(1) == num_gop:
            return audio_tokens.unsqueeze(2)

        batch_size, n_tokens, feat_dim = audio_tokens.shape
        if num_gop <= 0:
            raise ValueError(f"num_gop must be > 0, got {num_gop}")

        # Make sequence length divisible by num_gop to avoid invalid `.view(...)`
        tokens_per_gop = max(1, (n_tokens + num_gop - 1) // num_gop)  # ceil division
        target_tokens = tokens_per_gop * num_gop

        if n_tokens < target_tokens:
            pad_tokens = target_tokens - n_tokens
            pad = audio_tokens.new_zeros(batch_size, pad_tokens, feat_dim)
            audio_tokens = torch.cat([audio_tokens, pad], dim=1)
        elif n_tokens > target_tokens:
            audio_tokens = audio_tokens[:, :target_tokens, :]

        return audio_tokens.view(batch_size, num_gop, tokens_per_gop, feat_dim)
