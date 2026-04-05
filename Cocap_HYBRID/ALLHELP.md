Homoscedastic Task Uncertainty (The "Kendall" Method)
This is the most elegant and widely used method for this exact problem, introduced by Kendall et al. (2018). Instead of a fixed $\lambda$, you turn the weights into learnable parameters and let the network figure it out using backpropagation.You introduce two learnable scalar parameters, $\sigma_{MSE}$ and $\sigma_{CE}$ (representing the uncertainty of each task). Your total loss becomes:$$Loss_{Total} = \frac{1}{2\sigma_{MSE}^2} Loss_{MSE} + \frac{1}{\sigma_{CE}^2} Loss_{CE} + \log(\sigma_{MSE}) + \log(\sigma_{CE})$$


TEACHER SIGNAL (target):
  P-frame ──► CLIP ViT-B/16 (frozen) ──► drop CLS ──► z_p  [196 × 768]

STUDENT PIPELINE:
  I-frame ──► CLIP ViT-B/16 (frozen) ──► drop CLS ──► z_i  [196 × 768]
  Raw MVs ──► MLP ──► Motion Encoder ──► z_mv  [16 × 768]

  CONCAT ──► [z_i ; z_mv]  [212 × 768]  (context/memory)
                    │
                    ▼
        ┌─────────────────────────┐
        │   Reference Transformer  │  ← TRAINABLE, dropped at inference
        │                         │
        │  196 learned queries     │  ← Q
        │  212 tokens as K, V      │  ← KV from concat
        │                         │
        │  Cross-attention layers  │
        └─────────────────────────┘
                    │
                    ▼
              z_pred  [196 × 768]

DISTILLATION:  MSE(z_pred, z_p)
CAPTIONING:    CE(caption_logits, gt)
COMBINED:      Kendall uncertainty weighting