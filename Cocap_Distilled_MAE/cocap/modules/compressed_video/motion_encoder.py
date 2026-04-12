import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# =====================================================================
# 1. HELPER: RANDOM MASKING (MAE STYLE)
# =====================================================================
def random_masking(x, mask_ratio):
    """
    Standard MAE masking logic for the spatial patches.
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
# 2. CLASS: MOTION STUDENT (The Phase 1 Encoder & Phase 2 Backbone)
# =====================================================================
class MotionStudent(nn.Module):
    def __init__(self, in_channels=2, embed_dim=256, num_heads=4, num_layers=4):
        super().__init__()
        
        # 1. Spatial Stem (Process variable frames independently)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)) # Condense each frame to a single vector
        )
        
        # 2. The Bottleneck: 8 Learnable Motion Queries
        self.motion_queries = nn.Parameter(torch.randn(1, 8, embed_dim) / (embed_dim ** 0.5))
        
        # 3. Cross-Attention (Squashes variable frames -> exactly 8 tokens)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        
        # 4. Temporal Transformer (Chronological reasoning)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, mvs, input_mask_mv=None):
        """
        mvs: [B*G, T, 2, H, W]  (Note: T is up to 29)
        input_mask_mv: [B*G, T] (1 = padding/ignore, 0 = valid frame)
        """
        BG, T, C, H, W = mvs.shape
        
        # --- A. Frame-by-Frame Spatial Extraction ---
        x = mvs.view(BG * T, C, H, W)
        x = self.stem(x)              # [BG*T, embed_dim, 1, 1]
        x = x.view(BG, T, -1)         # [BG, T, embed_dim]
        
        # --- B. Prevent PyTorch NaN Crash ---
        # If a GOP is completely empty (all True padding), MultiheadAttention returns NaN.
        # We safely unmask the first frame of entirely empty sequences (loss ignores them later).
        if input_mask_mv is not None:
            safe_mask = input_mask_mv.bool().clone()
            all_masked = safe_mask.all(dim=1)
            safe_mask[all_masked, 0] = False 
        else:
            safe_mask = None
            
        # --- C. The Cross-Attention Bottleneck ---
        q = self.motion_queries.expand(BG, -1, -1) # [BG, 8, embed_dim]
        
        q_norm = self.ln_q(q)
        kv_norm = self.ln_kv(x)
        
        # The mask physically blocks queries from seeing the padded zero frames
        attn_out, _ = self.cross_attn(
            query=q_norm, 
            key=kv_norm, 
            value=kv_norm, 
            key_padding_mask=safe_mask 
        ) 
        q = q + attn_out # Residual connection
        
        # --- D. Temporal Interaction ---
        return self.transformer(q) # Output is exactly [BG, 8, embed_dim]

# =====================================================================
# 3. CLASS: DISTILLATION DECODER (The Phase 1 Teacher)
# =====================================================================
class DistillationDecoder(nn.Module):
    def __init__(self, student_dim=256, d_model=768, clip_dim=768, n_layers=2):
        super().__init__()
        # Align Student latent to Decoder
        self.proj_in = nn.Linear(student_dim, d_model)
        
        # Project CLIP spatial patches (Optional, if clip_dim == d_model this is just a linear mix)
        self.proj_clip = nn.Linear(clip_dim, d_model)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model*4, batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.proj_delta = nn.Linear(d_model, clip_dim)

    def forward(self, latent, iframe_spatial, mask_ratio=0.75):
        # 1. Align Student latent (256) to Decoder (768)
        motion_tokens = self.proj_in(latent) # [B*G, 8, 768]
        
        # 2. Project CLIP spatial patches (768)
        iframe_projected = self.proj_clip(iframe_spatial.float()) # [B*G, 196, 768]
        
        # 3. Mask the current I-frame spatial patches (The Anchor)
        x_masked, _, ids_restore = random_masking(iframe_projected, mask_ratio)
        
        # 4. Build sequence: [Motion Tokens (8) + Masked I-frame + Mask Tokens]
        combined = torch.cat([motion_tokens, x_masked], dim=1)
        
        num_mask_tokens = ids_restore.shape[1] - x_masked.shape[1]
        mask_tokens = self.mask_token.repeat(combined.shape[0], num_mask_tokens, 1)
        
        x_full = torch.cat([combined, mask_tokens], dim=1) 
        
        # Re-order patches to correct spatial order (keeping motion tokens at index 0-7)
        x_restored = torch.cat([
            x_full[:, :8, :], 
            torch.gather(x_full[:, 8:, :], dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[-1]))
        ], dim=1)
        
        # 5. Decode the CLIP Delta
        decoded = self.decoder(x_restored)
        
        # Extract only the 196 spatial deltas (Ignore the 8 motion tokens at the front)
        return self.proj_delta(decoded[:, 8:, :]) # [B*G, 196, 768]

# =====================================================================
# 4. CLASS: MOTION TRANSFORMER (The Phase 1 Pretraining Wrapper)
# =====================================================================
class MotionTransformer(nn.Module):
    """
    This is strictly a wrapper for Phase 1 Distilled MAE Pretraining.
    It holds the Student and Decoder together to calculate the loss.
    """
    def __init__(self, embed_dim: int = 256, d_model: int = 768, num_heads: int = 8, num_layers: int = 3):
        super().__init__()
        # We explicitly enforce in_channels=2 for the dx/dy motion vectors
        self.student = MotionStudent(in_channels=2, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers)
        self.decoder = DistillationDecoder(student_dim=embed_dim, d_model=d_model, n_layers=2)

    def forward(self, mvs: torch.Tensor, input_mask_mv: torch.Tensor, iframe_spatial: torch.Tensor, mask_ratio: float = 0.75):
        
        # 1. Run the core student encoder with the Frame-Level Padding Mask
        latent = self.student(mvs, input_mask_mv) # [B*G, 8, 256]
        
        # 2. Run the Distillation Decoder to predict the CLIP delta
        predicted_delta = self.decoder(latent, iframe_spatial, mask_ratio) # [B*G, 196, 768]
        
        return predicted_delta