"""
BEATs Architecture Implementation from Scratch
This file implements the complete BEATs architecture with detailed documentation.
Can load pretrained weights from the official checkpoint.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Parameter
import torchaudio.compliance.kaldi as ta_kaldi
from typing import Optional, Tuple


# ============================================================
# UTILITY MODULES
# ============================================================

class SamePad(nn.Module):
    """Padding module to handle same padding for convolutions."""
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, :-self.remove]
        return x


class GradMultiply(torch.autograd.Function):
    """Custom gradient scaling function."""
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

class MultiheadAttention(nn.Module):
    """
    Multi-head self-attention mechanism with optional relative position bias.
    
    Args:
        embed_dim: Total dimension of the model
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in projections
        has_relative_attention_bias: Whether to use relative position bias
        gru_rel_pos: Whether to use GRU-style relative position
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
        self_attention: bool = True,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 320,
        max_distance: int = 1280,
        gru_rel_pos: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.self_attention = self_attention
        
        # Calculate head dimension
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        
        # Scaling factor for attention scores
        self.scaling = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Relative position bias (optional)
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.num_buckets = num_buckets
            self.max_distance = max_distance
            self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        
        # GRU-style relative position (optional)
        self.gru_rel_pos = gru_rel_pos
        if gru_rel_pos:
            self.grep_linear = nn.Linear(self.head_dim, 8)
            self.grep_a = nn.Parameter(torch.ones(1, num_heads, 1, 1))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        
        if self.has_relative_attention_bias:
            nn.init.xavier_normal_(self.relative_attention_bias.weight)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            query: (batch, seq_len, embed_dim)
            key: (batch, seq_len, embed_dim)
            value: (batch, seq_len, embed_dim)
            key_padding_mask: (batch, seq_len) - True for positions to mask
        
        Returns:
            attn_output: (batch, seq_len, embed_dim)
            attn_weights: Optional attention weights
            position_bias: Optional position bias
        """
        batch_size, seq_len, embed_dim = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scale queries
        q = q * self.scaling
        
        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Add position bias if available
        if position_bias is not None:
            attn_weights = attn_weights + position_bias
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask
        
        # Apply key padding mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax over last dimension
        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, v)
        
        # Reshape back: (batch, seq_len, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights if need_weights else None, position_bias


# ============================================================
# TRANSFORMER ENCODER LAYER
# ============================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feedforward network.
    
    Architecture:
        1. Layer Norm (optional, if layer_norm_first=True)
        2. Multi-head Self-Attention
        3. Dropout + Residual
        4. Layer Norm
        5. Feedforward (Linear -> Activation -> Dropout -> Linear)
        6. Dropout + Residual
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        activation_fn: str = "gelu",
        layer_norm_first: bool = False,
        has_relative_attention_bias: bool = False,
        num_buckets: int = 320,
        max_distance: int = 1280,
        gru_rel_pos: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.layer_norm_first = layer_norm_first
        
        # Self-attention
        self.self_attn = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            gru_rel_pos=gru_rel_pos,
        )
        
        # Dropouts
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Layer norms
        self.self_attn_layer_norm = LayerNorm(embed_dim)
        self.final_layer_norm = LayerNorm(embed_dim)
        
        # Feedforward network
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        
        # Activation function
        if activation_fn == "gelu":
            self.activation_fn = nn.GELU()
        elif activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            attn_mask: Optional attention mask
            padding_mask: Optional padding mask
        
        Returns:
            x: (batch, seq_len, embed_dim)
        """
        residual = x
        
        if self.layer_norm_first:
            # Pre-norm architecture
            x = self.self_attn_layer_norm(x)
            x, _, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x
            
            residual = x
            x = self.final_layer_norm(x)
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            # Post-norm architecture
            x, _, _ = self.self_attn(
                query=x, key=x, value=x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            
            residual = x
            x = self.fc1(x)
            x = self.activation_fn(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        
        return x


# ============================================================
# TRANSFORMER ENCODER (STACK OF LAYERS)
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers with positional encoding.
    
    Args:
        num_layers: Number of transformer layers
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        ffn_dim: Feedforward network dimension
        dropout: Dropout probability
    """
    def __init__(
        self,
        num_layers: int = 12,
        embed_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        layer_norm_first: bool = False,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        num_buckets: int = 320,
        max_distance: int = 1280,
        gru_rel_pos: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.embed_dim = embed_dim
        self.layer_norm_first = layer_norm_first
        
        # Convolutional positional encoding
        self.pos_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        
        # Initialize positional encoding
        std = math.sqrt((4 * (1.0 - 0)) / (conv_pos * embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)
        
        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(conv_pos), nn.GELU())
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                layer_norm_first=layer_norm_first,
                has_relative_attention_bias=True,  # Enable for all layers
                num_buckets=num_buckets,
                max_distance=max_distance,
                gru_rel_pos=gru_rel_pos,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.layer_norm = LayerNorm(embed_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            padding_mask: (batch, seq_len) - True for positions to mask
        
        Returns:
            x: (batch, seq_len, embed_dim)
            layer_results: List of intermediate outputs
        """
        # Apply convolutional positional encoding
        # Need to transpose for conv1d: (batch, embed_dim, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = self.pos_conv(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv
        
        # Apply dropout
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Pass through transformer layers
        layer_results = []
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
            layer_results.append(x)
        
        # Final layer norm (if not layer_norm_first)
        if not self.layer_norm_first:
            x = self.layer_norm(x)
        
        return x, layer_results


# ============================================================
# BEATS MODEL
# ============================================================

class BEATsConfig:
    """Configuration class for BEATs model."""
    def __init__(self, cfg=None):
        # Patch embedding
        self.input_patch_size: int = 16
        self.embed_dim: int = 512
        self.conv_bias: bool = False
        
        # Transformer encoder
        self.encoder_layers: int = 12
        self.encoder_embed_dim: int = 768
        self.encoder_ffn_embed_dim: int = 3072
        self.encoder_attention_heads: int = 12
        self.activation_fn: str = "gelu"
        
        # Dropouts
        self.dropout: float = 0.1
        self.attention_dropout: float = 0.1
        self.activation_dropout: float = 0.0
        self.dropout_input: float = 0.0
        
        # Positional encoding
        self.conv_pos: int = 128
        self.conv_pos_groups: int = 16
        
        # Layer norm
        self.layer_norm_first: bool = False
        
        if cfg is not None:
            self.update(cfg)
    
    def update(self, cfg: dict):
        """Update config from dictionary."""
        self.__dict__.update(cfg)


class BEATsModel(nn.Module):
    """
    BEATs: Audio Pre-Training with Acoustic Tokenizers
    
    Architecture Overview:
    1. Mel-Spectrogram Extraction: Convert raw audio to mel-spectrogram
       - Input: (batch, samples)
       - Output: (batch, time_frames, freq_bins) = (batch, T, 128)
    
    2. 2D Patch Embedding: Divide mel-spectrogram into patches
       - Conv2d with kernel_size=stride=16
       - Input: (batch, 1, T, 128)
       - Output: (batch, embed_dim, T//16, 128//16)
       - Reshape to: (batch, T//16 * 128//16, embed_dim)
    
    3. Linear Projection (optional): Project to encoder dimension
       - Input: (batch, num_patches, embed_dim)
       - Output: (batch, num_patches, encoder_embed_dim)
    
    4. Transformer Encoder: Self-attention layers with positional encoding
       - Input: (batch, num_patches, encoder_embed_dim)
       - Output: (batch, num_patches, encoder_embed_dim)
    """
    def __init__(self, cfg: BEATsConfig):
        super().__init__()
        print(f"\n{'='*60}")
        print("Initializing BEATs Model")
        print(f"{'='*60}")
        print(f"Config:")
        for key, value in cfg.__dict__.items():
            print(f"  {key}: {value}")
        print(f"{'='*60}\n")
        
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        
        # 1. Patch embedding: 2D convolution on mel-spectrogram
        self.patch_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.embed_dim,
            kernel_size=cfg.input_patch_size,
            stride=cfg.input_patch_size,
            bias=cfg.conv_bias
        )
        
        # 2. Projection to encoder dimension (if different)
        self.post_extract_proj = (
            nn.Linear(self.embed_dim, cfg.encoder_embed_dim)
            if self.embed_dim != cfg.encoder_embed_dim
            else None
        )
        
        # 3. Input dropout
        self.dropout_input = nn.Dropout(cfg.dropout_input)
        
        # 4. Layer norm before encoder
        self.layer_norm = LayerNorm(self.embed_dim)
        
        # 5. Transformer encoder
        self.encoder = TransformerEncoder(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            num_heads=cfg.encoder_attention_heads,
            ffn_dim=cfg.encoder_ffn_embed_dim,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            activation_dropout=cfg.activation_dropout,
            layer_norm_first=cfg.layer_norm_first,
            conv_pos=cfg.conv_pos,
            conv_pos_groups=cfg.conv_pos_groups,
            num_buckets=getattr(cfg, 'num_buckets', 320),
            max_distance=getattr(cfg, 'max_distance', 1280),
            gru_rel_pos=getattr(cfg, 'gru_rel_pos', False),
        )
    
    def preprocess(
        self,
        waveform: torch.Tensor,
        fbank_mean: float = 15.41663,
        fbank_std: float = 6.55582,
    ) -> torch.Tensor:
        """
        Convert waveform to mel-spectrogram.
        
        Args:
            waveform: (batch, num_samples) - Audio waveform at 16kHz
            fbank_mean: Mean for normalization
            fbank_std: Std for normalization
        
        Returns:
            fbank: (batch, time_frames, freq_bins=128) - Normalized mel-spectrogram
        """
        fbanks = []
        for wav in waveform:
            # Scale to int16 range
            wav = wav.unsqueeze(0) * (2 ** 15)
            
            # Extract mel-filterbank features
            # frame_length=25ms, frame_shift=10ms, num_mel_bins=128
            fbank = ta_kaldi.fbank(
                wav,
                num_mel_bins=128,
                sample_frequency=16000,
                frame_length=25,
                frame_shift=10
            )
            fbanks.append(fbank)
        
        fbank = torch.stack(fbanks, dim=0)
        
        # Normalize
        fbank = (fbank - fbank_mean) / (2 * fbank_std)
        
        return fbank
    
    def forward(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through BEATs model.
        
        Args:
            waveform: (batch, num_samples) - Audio at 16kHz
            padding_mask: Optional padding mask
        
        Returns:
            features: (batch, num_patches, encoder_embed_dim)
            padding_mask: Updated padding mask
        """
        # Step 1: Extract mel-spectrogram
        fbank = self.preprocess(waveform)
        # Shape: (batch, time_frames, freq_bins=128)
        
        # Step 2: Add channel dimension for Conv2d
        fbank = fbank.unsqueeze(1)
        # Shape: (batch, 1, time_frames, freq_bins=128)
        
        # Step 3: Apply 2D patch embedding
        features = self.patch_embedding(fbank)
        # Shape: (batch, embed_dim, time_patches, freq_patches)
        
        # Step 4: Reshape to sequence format
        batch_size = features.shape[0]
        features = features.reshape(batch_size, self.embed_dim, -1)
        # Shape: (batch, embed_dim, num_patches)
        
        features = features.transpose(1, 2)
        # Shape: (batch, num_patches, embed_dim)
        
        # Step 5: Apply layer norm
        features = self.layer_norm(features)
        
        # Step 6: Project to encoder dimension (if needed)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        # Shape: (batch, num_patches, encoder_embed_dim)
        
        # Step 7: Apply dropout
        features = self.dropout_input(features)
        
        # Step 8: Pass through transformer encoder
        features, _ = self.encoder(features, padding_mask=padding_mask)
        # Shape: (batch, num_patches, encoder_embed_dim)
        
        return features, padding_mask
    
    def extract_features(
        self,
        waveform: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Alias for forward pass."""
        return self.forward(waveform, padding_mask)


# ============================================================
# MAIN FUNCTION FOR TESTING
# ============================================================

def load_beats_model(checkpoint_path: str, device: str = "cpu") -> BEATsModel:
    """
    Load BEATs model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        model: Loaded BEATs model
    """
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create config from checkpoint
    cfg = BEATsConfig(checkpoint["cfg"])
    
    # Create model
    model = BEATsModel(cfg).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint["model"])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    return model


if __name__ == "__main__":
    import torchaudio
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model
    model = load_beats_model("learn/beats/BEATs_iter3_plus_AS2M.pt", device=device)
    
    # Load audio
    waveform, sr = torchaudio.load("learn/beats/audio.wav")
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    waveform = waveform.to(device)
    
    print(f"Audio shape: {waveform.shape}")
    print(f"Audio duration: {waveform.shape[1] / 16000:.2f} seconds\n")
    
    # Extract features
    with torch.no_grad():
        features, _ = model.extract_features(waveform)
    
    print(f"Output features shape: {features.shape}")
    print(f"  - Batch size: {features.shape[0]}")
    print(f"  - Sequence length (num patches): {features.shape[1]}")
    print(f"  - Feature dimension: {features.shape[2]}")
