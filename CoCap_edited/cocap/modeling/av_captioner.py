import torch
import torch.nn as nn
from cocap.modules.clip.model import ModifiedResNet
from cocap.modules.audio_encoder import CNN14, BEATsAudioEncoder

class AVCaptioner(nn.Module):
    def __init__(
        self,
        embed_dim=1024, 
        vision_layers=[3, 4, 6, 3], 
        vision_width=64,
        vision_heads=32, 
        audio_pretrained=False,
        vocab_size=49408, 
        max_len=77,
        num_decoder_layers=6,
        num_heads=8,
        dim_feedforward=2048,
        dropout=0.1,
        clip_path=None,
        audio_path=None,
        audio_enc_type="beats" # "cnn14" or "beats"
    ):
        super().__init__()
        
        # Visual Encoder
        self.visual_encoder = ModifiedResNet(
            layers=vision_layers,
            output_dim=1024, 
            heads=vision_heads,
            input_resolution=224,
            width=vision_width
        )
        
        embed_dim = 1024 
        
        # Audio Encoder
        if audio_enc_type == "beats":
            self.audio_encoder = BEATsAudioEncoder(model_path=audio_path)
            # BEATs Base outputs 768. Large 1024.
            # Assuming Base (768) for "iter3+ as2m" standard checkpoint.
            audio_dim = 768 
            # If large, use 1024. Check config? 
            # We can check self.audio_encoder.cfg.encoder_embed_dim
            if hasattr(self.audio_encoder, 'cfg'):
                audio_dim = self.audio_encoder.cfg.encoder_embed_dim
                
        else:
            self.audio_encoder = CNN14() 
            audio_dim = 2048
            if audio_path:
                self.audio_encoder.load_from_pretrain(audio_path)
            
        self.audio_proj = nn.Linear(audio_dim, embed_dim)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.randn(max_len, embed_dim))
        
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Share weights
        self.head.weight = self.token_embedding.weight
        
        if clip_path:
            self.load_visual_weights(clip_path)

    def load_visual_weights(self, path):
        import os
        if not os.path.exists(path):
            print(f"CLIP model not found at {path}")
            return
            
        print(f"Loading CLIP from {path}")
        jit_model = torch.jit.load(path, map_location="cpu")
        state_dict = jit_model.state_dict()
        
        visual_sd = {}
        for k, v in state_dict.items():
            if k.startswith("visual."):
                new_key = k[7:] # remove "visual."
                visual_sd[new_key] = v
        
        msg = self.visual_encoder.load_state_dict(visual_sd, strict=False)
        print(f"Visual Encoder Loaded: {msg}")

    def forward(self, images, audios, captions, caption_mask=None):
        """
        images: (B, 3, H, W)
        audios: (B, T_audio, F_mel) or (B, 1, T, F)
        captions: (B, L_cap) indices
        caption_mask: (B, L_cap) boolean or 0/1 mask. True/1 means valid.
        """
        
        # 1. Encode Images
        # ModifiedResNet returns (B, embed_dim) ?? 
        # Wait, let's check ModifiedResNet output. 
        # It calls attnpool which returns (B, embed_dim).
        # We usually want a sequence for cross-attention, or at least we want the spatial features.
        # But the prompt says "Visual Encoder: Use the ModifiedResNet ... AttentionPool2d"
        # AttentionPool2d reduces spatial dim to 1 vector in standard CLIP.
        # If we want detailed attention, we might want to skip the final pooling or unsqueeze it.
        # Let's assume we use the global visual feature for now as typically done in simple captioners,
        # OR we can modify it to return spatial features.
        # For "High performance", spatial features are better. 
        # The attnpool in existing code:
        # x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # x, _ = F.multi_head_attention_forward(...) -> returns x.squeeze(0) which is (B, D).
        # So it returns a single vector.
        # Let's stick to using this single vector 'v' plus audio vector 'a'. 
        # Or expand them.
        
        v_features = self.visual_encoder(images) # (B, D)
        v_features = v_features.unsqueeze(1) # (B, 1, D)
        
        # 2. Encode Audio
        a_features = self.audio_encoder(audios) # (B, 2048)
        a_features = self.audio_proj(a_features) # (B, D)
        a_features = a_features.unsqueeze(1) # (B, 1, D)
        
        # 3. Concatenate
        memory = torch.cat([v_features, a_features], dim=1) # (B, 2, D)
        
        # 4. Decode
        # captions: (B, L)
        # Shift captions for teacher forcing
        # targets usually: captions[:, 1:]
        # inputs: captions[:, :-1]
        
        tgt_seq = captions
        tgt_emb = self.token_embedding(tgt_seq) # (B, L, D)
        
        # Add position embeddings
        seq_len = tgt_seq.size(1)
        positions = torch.arange(0, seq_len, device=tgt_seq.device).unsqueeze(0)
        tgt_emb = tgt_emb + self.positional_embedding[positions]
        
        # Causal Mask
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_seq.device)
        
        # Padding Mask (for key_padding_mask)
        # caption_mask is 1 for valid, 0 for pad.
        # nn.Transformer expects True for PAD (to ignore).
        # If caption_mask is provided as 1=valid, 0=pad, we invert it.
        key_padding_mask = None
        if caption_mask is not None:
             key_padding_mask = (caption_mask == 0)

        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=key_padding_mask
        )
        
        logits = self.head(output)
        return logits

    @torch.no_grad()
    def generate(self, images, audios, max_len=20, start_token=49406, end_token=49407):
        """
        Greedy decoding.
        """
        B = images.size(0)
        
        # Encode
        v_features = self.visual_encoder(images).unsqueeze(1) # (B, 1, D)
        a_features = self.audio_proj(self.audio_encoder(audios)).unsqueeze(1) # (B, 1, D)
        memory = torch.cat([v_features, a_features], dim=1)
        
        # Decode
        # Start token
        tgt = torch.full((B, 1), start_token, dtype=torch.long, device=images.device)
        
        finished = torch.zeros(B, dtype=torch.bool, device=images.device)
        
        for i in range(max_len):
            tgt_emb = self.token_embedding(tgt)
            # Positional
            seq_len = tgt.size(1)
            positions = torch.arange(0, seq_len, device=images.device).unsqueeze(0)
            tgt_emb = tgt_emb + self.positional_embedding[positions]
            
            # Mask ? Not needed for inference if we pass all past tokens, 
            # transformer decoder is autoregressive but usually we pass full sequence so far 
            # and it outputs all steps. Use the last one.
            # Efficient implementation uses cache, but here we just re-forward.
            
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(images.device)
            
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            logits = self.head(output[:, -1, :]) # Take last step
            
            next_token = logits.argmax(dim=-1, keepdim=True) # (B, 1)
            
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check for end token
            finished |= (next_token.squeeze(-1) == end_token)
            if finished.all():
                break
                
        return tgt
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
