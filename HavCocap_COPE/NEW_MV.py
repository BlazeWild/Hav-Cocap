import torch
import torch.nn as nn

class MotionCompressor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, embed_dim=768, num_queries=4, num_heads=8):
        super().__init__()
        
        # 1. The Coordinate MLP (Translates 2D MV math into feature space)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # 2. The Learnable Query Tokens (The "Sponges")
        # We define a parameter tensor of shape [1, 4, 768].
        # It starts as random noise, but learns during backpropagation.
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, embed_dim))
        
        # 3. The Cross-Attention Mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        
        # 4. Standard Transformer Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, embed_dim)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, mv_patches):
        """
        mv_patches: Tensor of shape [Batch, 49, 512] 
        (49 patches from a 56x56 grid, 512 channels per patch)
        """
        batch_size = mv_patches.size(0)
        
        # Step 1: Pass raw MV math through the MLP
        # Output shape: [Batch, 49, 768]
        kv_features = self.mlp(mv_patches)
        
        # Step 2: Expand our 4 Query Tokens so every item in the batch gets a copy
        # Output shape: [Batch, 4, 768]
        q = self.query_tokens.expand(batch_size, -1, -1)
        
        # Step 3: Cross-Attention
        # The 4 Queries look at the 49 Keys/Values and extract the motion
        q_norm = self.norm_q(q)
        kv_norm = self.norm_k(kv_features)
        
        attn_output, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        
        # Add residual connection
        out = q + attn_output
        
        # Step 4: Final Feed-Forward
        out = out + self.ffn(self.norm_ffn(out))
        
        # Final Output Shape: [Batch, 4, 768]
        return out