import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

import einops
import numpy as np

from typing import Union, Tuple

class LayerNorm(nn.LayerNorm):
    """Custom LayerNorm to avoid fp16 precision issues:
    does computation in fp32 for stability, then casts back to original dtype."""
    
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # Cast to float32 → prevents numerical instability in mean/var calc with fp16
        ret = super().forward(x.type(torch.float32))
        # Cast back to original dtype → keeps model in mixed precision (saves speed & memory)
        return ret.type(orig_type)

class QuickGELU(nn.Module):
    """Quick GELU activation function."""
    def forward(self, x:torch.tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    """It is just a encoder block from attention is all you need paper with some modifications
    1. pre-norm instead of post-norm
    2. quick gelu instead of relu
    
    the architecture is liek this
        Input x
        │
        ▼
        LayerNorm (ln_1)
        │
        ▼
        Multi-Head Self-Attention
        │
        ▼
        Residual Add (x + attn_out)
        │
        ▼
        LayerNorm (ln_2)
        │
        ▼
        Feed-Forward MLP (Linear → GELU → Linear)
        │
        ▼
        Residual Add (x + mlp_out)
        │
        ▼
        Output (x, attention_weights)

    """
    
    def __init__(self, d_model:int, n_head:int, attn_mask:torch.Tensor = None):
        """Attn_mask is optional: it is set to None by default, i.e no causal masking
        Args:
            d_model (int): Dimension of the model (input and output feature size).eg. 768 for GPT 2
            n_head (int): Number of attention heads.
            attn_mask (torch.Tensor, optional): Attention mask to be applied. Defaults to None.
        Note:
            No need for seq_len(attention done between no of tokens) does not need to be passed to MultiheadAttention. It is determined by the input tensor shape."""
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            "c_fc", nn.Linear(d_model, d_model * 4),
            "gelu", QuickGELU(),
            "c_proj", nn.Linear(d_model * 4, d_model)
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask #whether to apply attention mask or not ,it is optional 
        
    def attention(self, x:torch.Tensor, padding_mask:torch.Tensor = None):
        """
        Compute multi-head self-attention for the input.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model), 
                            where d_model is the token embedding size.
            padding_mask (torch.Tensor, optional): Binary mask of shape (batch_size, seq_len), 
                            with 1 for positions to ignore (padding), 0 to keep. Default is None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - output tensor of same shape as x (seq_len, batch_size, d_model)
                - attention weights per head (batch_size, n_head, seq_len, seq_len)

        Why this is done:
            - x, x, x: Self-attention (queries, keys, values all same)
            - Cast attn_mask to x's dtype/device to avoid mismatch errors
            - key_padding_mask ignores padding tokens in attention computation
            - average_attn_weights=False returns per-head attention
        """
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(
            x,x,x, 
            need_weights=True, 
            attn_mask=self.attn_mask, 
            key_padding_mask = padding_mask, 
            average_attn_weights=False)
    
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the ResidualAttentionBlock.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, d_model)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Updated token embeddings (seq_len, batch_size, d_model)
                - Attention weights per head (batch_size, n_head, seq_len, seq_len)

        Steps:
            1. Apply LayerNorm (ln_1) before multi-head self-attention (Pre-Norm)
            2. Compute attention and add residual connection
            3. Apply LayerNorm (ln_2) before MLP
            4. Pass through MLP and add residual
            5. Return updated embeddings and attention weights
        """
        attention_res = self.attention(self.ln_1(x)) # prenorm on input x.
        # attention_res is a TUPLE returned by nn.MultiheadAttention(d_model, n_head)

        # attention_res[0]: Concatenated Output (Context Vectors). 
        # Shape (B, S, N) | Result of (Weights @ V) then merged heads; added to x for the residual update.

        # attention_res[1]: Attention Weight Matrix (Relational Map). 
        # Shape (B, S, S) | Result of Softmax(Q @ K.T / sqrt(dk)); stored for visualization or analysis.
        x, weights = x + attention_res[0], attention_res[1] # as tuple is returned
        x = x + self.mlp(self.ln_2(x))
        return x, weights
        
class Transformer(nn.Module):
    """
    Transformer model: stack of multiple ResidualAttentionBlocks.
    
    Args:
        width (int): Token embedding dimension (d_model).
        layers (int): Number of transformer blocks.
        heads (int): Number of attention heads per block.
        attn_mask (torch.Tensor, optional): Attention mask (e.g., causal mask).
    """
    def __init__(self, width:int, layers:int, heads:int, attn_mask:torch.Tensor=None):

        super().__init__()
        self.width = width
        self.layers = layers
        #creates a chain of modules that are executed in order.eg[Block1 -> Block2 -> Block3]
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
    def forward(self, x:torch.Tensor):
        """
        Forward pass through the stacked transformer blocks.
        
        Args:
            x (torch.Tensor): Input embeddings of shape (seq_len, batch, width).
        
        Returns:
            x (torch.Tensor): Final transformed embeddings of shape (seq_len, batch, width).
            weights_all_blocks (torch.Tensor): Attention weights from all layers,
                                               shape (layers, batch, heads, seq_len, seq_len).
        """
        #At the beginning, you don’t have any attention weights yet.
        weights_all_blocks=[]
        
        #Go through all the blocks (modules)
        for block in self.resblocks:
            x, weight = block(x)
            #We append attention weights from each block into a list, and at the end we stack them into one tensor across layers.
            weights_all_blocks.append(weight)
            
        return x, torch.stack(weights_all_blocks) #stack along new dimension
    
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution:int, patch_size:int, width:int, layers:int, heads:int, output_dim:int):
        """class VisionTransformer(nn.Module):

        Vision Transformer (ViT) for image classification.

        Splits the input image into patches, projects each patch into a `width`-dim embedding, prepends a 
        learnable class token, adds learnable positional embeddings, and passes the sequence through a 
        Transformer encoder. The output of the class token is used as the global image representation.

        Args:
            input_resolution (int): Input image size (assumed square), e.g., 224 for 224x224 images.
            patch_size (int): Size of each square patch, e.g., 16.
            width (int): Embedding dimension for patches and class token (d_model), e.g., 768.
            layers (int): Number of Transformer blocks to stack.
            heads (int): Number of attention heads per block.
            output_dim (int): Dimension of final output (e.g., number of classes).
            in_channels (int, optional): Number of image channels (default 3 for RGB).

        Notes:
            - class_embedding is learnable and summarizes the image.It is same for all images initially as it is randomly initialized.After training it learns to be a good summary of the image.And is different for different images.
            Eg. Sequence for one image: [Class_embedding, patch1, patch2, patch3, patch4]
            - positional_embedding is learnable and encodes spatial info.
            - conv1 splits image into patches and projects them.
            - scale = width**-0.5 ensures stable initialization.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5 #eg, for dim=768, scale = 1/sqrt(768)=0.036
        #initializing embeddings with huge random numbers (e.g. N(0,1)),we shrink them so their average magnitude ≈ 0.036 → stable dot products, stable softmax.
        #class embedding is a learnable parameter that is added to the sequence of patch embeddings to represent the entire image
        self.class_embedding = nn.Parameter(scale*torch.randn(width)) # a random vector of size width
        # learnable positional embeddings for each patch + 1 for class embedding
        self.positional_embedding = nn.Parameter(scale*torch.randn((input_resolution // patch_size) ** 2 + 1, width)) # as it is learnable, we write parameter when imnitializing, so that it can be updated during training.
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads)
        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale*torch.randn(width, output_dim))
        
    def forward(self, x: torch.Tensor, output_all_features: bool = False, output_attention_map: bool = False):
        """
        Forward pass of the Vision Transformer.

        Args:
            x (torch.Tensor): Input images of shape [batch_size, in_channels, height, width].
            output_all_features (bool, optional): If True, the function will also return all patch embeddings, not just the class token. Useful for things like visualizing patch features or doing segmentation tasks. Default: False.
            output_attention_map (bool, optional): If True, the function will return attention maps from the class token to all patches. Useful for visualizing where the model “looks” in the image.Default: False.

        Returns:
            tuple: Contains at least the class token features (cls_feature) of shape [batch_size, output_dim].
                Optionally:
                - Patch embeddings of shape [batch_size, num_patches, width] (if output_all_features=True)
                - Attention maps of shape [n_layers, batch_size, n_heads, grid, grid] (if output_attention_map=True)

        Notes:
            - Images are first converted to patch embeddings via a Conv2d layer.
            - A learnable class token is prepended to the sequence to aggregate image-level information.
            - Learnable positional embeddings are added to each token (including the class token).
            - The sequence is normalized (LayerNorm) and passed through the Transformer blocks.
            - The class token embedding is extracted, normalized, and projected to output_dim for downstream tasks.
            - Patch embeddings and attention maps are optional outputs useful for visualization or analysis.

            Input image: [B, 3, 224, 224]
                    │
            Conv2d → Patch embeddings: [B, 768, 14, 14]
                    │
            Flatten → [B, 196, 768]
                    │
            Add CLS token → [B, 197, 768]
                    │
            Add positional embeddings → [B, 197, 768]
                    │
            LayerNorm → [B, 197, 768]
                    │
            Transformer → [B, 197, 768], attn maps [layers, B, heads, 197, 197]
                    │
            Extract CLS token → [B, output_dim]
                    │
            Optional outputs → patch embeddings, attention maps

        """
        #split image into non-overlapping patches and project to `width` dimensions
        x = self.conv1(x) #shape=[*, width, grid, grid], eg. [*, 768, 14, 14] for 224x224 input and 16x16 patches
        grid = x.size(2)
        #flatten the 2D grid into a sequence of patches
        x = x.reshape(x.shape[0], x.shape[1],-1) #shape=[*, width, grid**2] eg. [*, 768, 196]
        x = x.permute(0,2,1) #shape=[*, grid**2, width]
        #add class token to the beginning of the sequence
        # self.class_embedding has shape (width,) → 1D vector, this exmplanation is written in the vision_transformer_explanation.ipynb file with exmaple
        batch_class_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x=torch.cat(
            [batch_class_token, x],
             dim=1) #shape=[*, grid**2+1, width]
        x=x + self.positional_embedding.to(x.dtype) #add positional embeddings
        
        #pre normalize all sequqnce elements including class token before feeding to the transfprmer though there is ln_1 (layernorm) in each block of transformer as the ln_1 normalizes per block, so both are needed
        x=self.ln_pre(x)
        # as transformer expects input of shape (seq_len, batch, width)
        x.permute(1,0,2) #NLD-> LND , shape=[grid**2+1, *, width] ,(* is batch size)
        x,attn = self.transformer(x) #shape=[grid**2+1, *, width], attn shape=[layers, *, heads, grid**2+1, grid**2+1]
        x = x.permute(1,0,2) #LND-> NLD
        
        
        
        # this is for class_feature extraction and is used for feature extraction or classification, ...It is not strictly needed. but in the cocap, it is included for feature extraction
        # ln_post normalizes the class token embedding
        # @ self.proj projects the embedding from width -> output_dim. as it is matrix multiplication
        # Shape after projection: [batch_size, output_dim]
        #x[:, 0, :] → selects the class token embedding for all images in the batch.
        cls_feature = self.ln_post(x[:,0,:]) @ self.proj ## cls_feature.shape = [batch_size, output_dim]
        
        # 1️⃣1️⃣ Prepare outputs tuple
        # Start with just the class token feature as primary output
        outputs = (cls_feature,)
        
        # Optional: include patch embeddings
        if output_all_features:
            # x[:, 1:, :] excludes the class token and keeps only patch embeddings
            # Shape: [batch_size, num_patches, width]
            outputs += (x[:, 1:, :],)
            # Purpose:
            # - Useful for tasks where individual patch features are needed
            #   e.g., segmentation, attention visualization, or feature extraction

        # Optional: include attention maps
        if output_attention_map:
            # attn.shape = [n_layers, batch_size, n_heads, seq_len, seq_len]
            # attn[:, :, :, 0, 1:] selects attention from the class token to all patches
            # Shape: [n_layers, batch_size, n_heads, num_patches]
            # einops rearranges it to match the 2D grid layout of patches: [n_layers, batch_size, n_heads, h, w]
            outputs += (einops.rearrange(
                # [B, h, seq_len, seq_len] → [8, 12, 197, 197] for single layer of tranformer block ,but for n_layers of tranfomer blocks stacked together it will be:
                # attn.shape = [n_layers, B, n_heads, seq_len, seq_len] → [12, 8, 12, 197, 197]
                attn[:, :, :, 0, 1:], # [n_layers, B, n_heads, num_patches, num_patches]
                # in above one, 0 → CLS token index, 1: → all patch tokens
                "n_layers b n_heads (h w) -> n_layers b n_heads h w",
                h=grid, w=grid
            ),)
            # Purpose:
            # - Visualizes where the class token "looks" in the image
            # - Helpful for interpretability of attention

        # 1️⃣2️⃣ Return final outputs
        # Tuple contains:
        # 1. cls_feature: image-level representation for classification
        # 2. (optional) patch embeddings: individual patch features
        # 3. (optional) attention maps: visualization of attention from class token to patches
        return outputs
    
class CrossResidualAttentionBlock(ResidualAttentionBlock):
    """modified version of ResidualAttentionBlock to support the encoder-decoder attention between I-frame tokens and
    motion vector/residual"""
    
    def __init__(self, 
                 d_model:int, 
                 n_head:int,
                 attn_mask:torch.Tensor = None,
                 enc_dec_attn_mask:torch.Tensor = None):
        super().__init__(d_model=d_model, n_head=n_head, attn_mask=attn_mask)
        self.attn2 = nn.MultiheadAttention(d_model, n_head)
        self.ln_3 = LayerNorm(d_model)
        self.ln_4 = LayerNorm(d_model)
        self.enc_dec_attn_mask = enc_dec_attn_mask
        
    def enc_dec_attention(self, highway:torch.Tensor, iframe:torch.Tensor):
        self.enc_dec_attn_mask = self.enc_dec_attn_mask.to(dtype=highway.dtype,device=highway.device) if self.enc_dec_attn_mask is not None else None
        return self.attn2(highway, iframe, iframe, need_weights = False, attn_mask = self.enc_dec_attn_mask)[0]

    def forward(self, x:[torch.Tensor, torch.Tensor, torch.LongTensor]):
        x = x[0]
# class CLIP(nn.Module):
#     def __init__(self,
#                  embed_dim:int,
#                  #vision
#                  image_resolution:int,
#                  vision_layers:Union[Tuple[int, int, int, int], int],
#                  )