import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# =====================================================================
# 1. PRE-PROCESSING: CAUSAL MASKING & SAFE POOLER
# =====================================================================
def safe_causal_motion_pooling(p_frame_mvs, eps=0.01, cut_threshold=0.15):
    """
    Strips out imposter P-frames and duplicate zero-MV frames before 
    pooling the 29 GOP MVs down to 8 tokens.
    """
    B_G, C, T, H, W = p_frame_mvs.shape
    
    # Filter Duplicates
    mag = p_frame_mvs.abs().mean(dim=(1, 3, 4))
    is_real = (mag > eps).float() 
    
    # Detect Scene Cuts (Cosine similarity between consecutive MVs)
    flat_mvs = p_frame_mvs.permute(0, 2, 1, 3, 4).reshape(B_G, T, -1)
    sim = F.cosine_similarity(flat_mvs[:, :-1, :], flat_mvs[:, 1:, :], dim=2)
    sim = torch.cat([torch.ones(B_G, 1, device=sim.device), sim], dim=1)
    
    # Causal Mask: Once a cut is detected, everything after it is ignored
    valid_scene = ((sim < cut_threshold).float().cumsum(dim=1) == 0).float()
    final_mask = (is_real * valid_scene).view(B_G, 1, T, 1, 1)
    
    # Safe Masked Pooling
    pool_raw = F.adaptive_avg_pool3d(p_frame_mvs * final_mask, (8, H, W))
    pool_mask = F.adaptive_avg_pool3d(final_mask, (8, 1, 1))
    
    corrected = pool_raw / pool_mask.clamp(min=1e-3)
    return corrected.masked_fill(pool_mask < eps, 0.0)

# =====================================================================
# 2. HELPER: RANDOM MASKING (MAE STYLE)
# =====================================================================
def random_masking(x, mask_ratio):
    """
    Standard MAE masking logic.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

# =====================================================================
# 3. CLASS: MOTION STUDENT (The Inference Encoder)
# =====================================================================
class MotionStudent(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=3):
        super().__init__()
        # Conv3D Stem: Stride (1, 2, 2) to maintain the 8 temporal tokens
        self.stem = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=(5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.Conv3d(64, embed_dim, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim),
            nn.GELU()
        )
        self.pool = nn.AdaptiveAvgPool3d((8, 1, 1))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, mvs):
        # MVs come in as [B*G, 29, 2, 56, 56]
        x = mvs.permute(0, 2, 1, 3, 4)      # [B*G, 2, 29, 56, 56]
        x = safe_causal_motion_pooling(x)   # [B*G, 2, 8, 56, 56]
        
        x = self.stem(x) 
        x = self.pool(x).flatten(2).transpose(1, 2) # [B*G, 8, embed_dim]
        return self.transformer(x)

# =====================================================================
# 4. CLASS: DISTILLATION DECODER (The Phase 1 Teacher)
# =====================================================================
class DistillationDecoder(nn.Module):
    def __init__(self, student_dim=256, d_model=512, n_layers=2):
        super().__init__()
        self.proj_in = nn.Linear(student_dim, d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj_delta = nn.Linear(d_model, 512) # CLIP latent dimension

    def forward(self, latent, iframe_spatial, mask_ratio=0.75):
        # 1. Align Student latent (256) to Decoder (512)
        motion_tokens = self.proj_in(latent) # [B*G, 8, 512]
        
        # 2. Mask the current I-frame spatial patches (The Anchor)
        x_masked, _, ids_restore = random_masking(iframe_spatial, mask_ratio)
        
        # 3. Build sequence: [Motion Tokens (8) + Masked I-frame + Mask Tokens]
        combined = torch.cat([motion_tokens, x_masked], dim=1)
        
        # Add mask tokens to fill back to 196 + 8
        num_mask_tokens = ids_restore.shape[1] - x_masked.shape[1]
        mask_tokens = self.mask_token.repeat(combined.shape[0], num_mask_tokens, 1)
        
        x_full = torch.cat([combined, mask_tokens], dim=1) 
        
        # Re-order patches to correct spatial order (keeping motion tokens at index 0-7)
        x_restored = torch.cat([
            x_full[:, :8, :], 
            torch.gather(x_full[:, 8:, :], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[-1]))
        ], dim=1)
        
        # 4. Decode the CLIP Delta
        decoded = self.decoder(x_restored)
        return self.proj_delta(decoded[:, 8:, :]) # Extract only the 196 spatial deltas

# =====================================================================
# 5. CLASS: MOTION TRANSFORMER (The Hydra-Targetable Wrapper)
# =====================================================================
class MotionTransformer(nn.Module):
    def __init__(self, embed_dim: int = 256, d_model: int = 512, num_heads: int = 4, num_layers: int = 3):
        super().__init__()
        self.student = MotionStudent(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.decoder = DistillationDecoder(student_dim=embed_dim, d_model=d_model)
        
        # GPT-2 projection head
        self.proj_gpt = nn.Linear(embed_dim, 768)

    def forward(self, mvs: torch.Tensor, iframe_spatial: Optional[torch.Tensor] = None, 
                mask_ratio: float = 0.75, return_for_gpt: bool = True):
        
        # 1. Run the core student encoder
        latent = self.student(mvs) # [B*G, 8, 256]
        
        if return_for_gpt:
            # PHASE 2 & 3: Return features for GPT-2
            return self.proj_gpt(latent)
        
        # 2. Run the Distillation Decoder (PHASE 1 ONLY)
        # This predicted delta will be compared against CLIP(I_t+1) - CLIP(I_t)
        return self.decoder(latent, iframe_spatial, mask_ratio)