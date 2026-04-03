# CoCap_NEW architecture (token flow + teacher MSE path)

## Legend (active state)
- 🟢 active in **TRAIN + VAL**
- 🔵 active in **TRAIN only**
- ⚪ inactive for that phase

---

## 0) Input tensors from compressed video reader

```
|| video reader output ||
iframe        : [B, G, 3, 224, 224]          (1 I-frame per GOP)
motion_vector : [B, G, P, 4, 56, 56]         (P motion fields per GOP)
residual      : [B, G, P, 3, 224, 224]
bp_rgb        : [B, G, P, 3, 224, 224]       (decoded P-frame RGB)
```

In your setup, usually: `G=8`, `P=7`, so per sample P-frame count is:

$$
N_p = B\times G\times P = B\times 8\times 7
$$

If `B=1`, then $N_p=56$.

---

## 1) Main visual branch to caption head

```
🟢 iframe [B,G,3,224,224]
   ->---- flatten GOP
   ->---- [(B*G),3,224,224]
   ->---- IFrameEncoder (CLIP ViT-B/16 style)
   ->---- feature_context_cls      [B,G,512]
   ->---- feature_context_spatial  [B,G,196,512]

🟢 motion_vector [B,G,P,4,56,56]
   ->---- flatten GOP+P
   ->---- [(B*G*P),4,56,56]
   ->---- MotionPerceiver
           || conv2d -> kv tokens [BGP, H*W, width]
           || learnable query tokens [BGP, 8, width]
           || for each layer:
              query/key-value cross-attn -> residual add
              MLP (Linear width->4*width, GELU, Linear 4*width->width) -> residual add
           || ln_post
   ->---- mv_tokens [BGP,8,width]
   ->---- reshape
   ->---- feature_motion [B,G,P,8,width]

🟢 mv projection before caption head (if needed)
   feature_motion [...,width]
   ->---- if width != caption video_feature_size(=512): Linear(width->512)
   ->---- else Identity
   ->---- feature_motion [B,G,P,8,512]

🟢 CaptionHead input build
   feature_context [B,G,512]
   feature_motion  [B,G,P,8,512]
   ->---- flatten motion tokens over (G,P,8)
   ->---- f_mot_flat [B,(G*P*8),512]
   ->---- concat visual sequence
           visual_concat=[feature_context ; f_mot_flat]
           shape [B, G + (G*P*8), 512]
   ->---- with text tokens (input_ids,input_mask)
   ->---- BertSelfEncoder
   ->---- BertLMPredictionHead
   ->---- prediction_scores [B,T,49408]
```

So yes: motion encoder has cross-attn + **MLP block** in every layer, and there is an explicit **`mv_proj` linear layer** before caption head when dims mismatch.

---

## 2) Teacher MSE branch (what exactly predicts what)

### 2.1 Target side (`teacher_target`) 🔵 TRAIN only

```
🔵 bp_rgb [B,G,P,3,224,224]
🔵 residual [B,G,P,3,224,224]
   ->---- pure_motion_rgb = bp_rgb - residual
   ->---- clamp [0,1]
   ->---- flatten -> [BGP,3,224,224]
   ->---- rgb_encoder (no grad)
   ->---- spatial tokens
   ->---- gt_spatial [BGP,196,512]   (= teacher_target)
```

This is per P-frame item independently. Not cumulative over previous P-frames.

### 2.2 Prediction side (`teacher_predicted`) 🔵 TRAIN only

```
🔵 i_spatial = feature_context_spatial [B,G,196,512]
   ->---- expand over P
   ->---- [B,G,P,196,512]
   ->---- flatten -> [BGP,196,512]

🔵 mv_tokens = feature_motion [B,G,P,8,512]
   ->---- flatten -> [BGP,8,512]

🔵 TeacherTransformer:
   [BGP,196,512] + [BGP,8,512]
   ->---- concat on token dim -> [BGP,204,512]
   ->---- TransformerEncoder
   ->---- take first 196 tokens
   ->---- Linear proj
   ->---- predicted_spatial [BGP,196,512] (= teacher_predicted)
```

### 2.3 MSE pairing

```
🔵 MSE compares elementwise:
teacher_predicted [BGP,196,512]
teacher_target    [BGP,196,512]
->---- F.mse_loss(..., reduction="mean")
```

All scalar elements are equally averaged in MSE. No per-frame custom weighting here.

---

## 3) Your specific confusion: cumulative or not?

It is **NOT** coded as:

- `pf3 <- iframe + mv2 + mv3`
- `pf4 <- iframe + mv2 + mv3 + mv4`

Instead it is per-P-frame conditioning:

- `pred(g,p1) <- Teacher( I_tokens(g), MV_tokens(g,p1) )`
- `pred(g,p2) <- Teacher( I_tokens(g), MV_tokens(g,p2) )`
- ...
- `pred(g,p7) <- Teacher( I_tokens(g), MV_tokens(g,p7) )`

So each P-frame prediction uses the GOP I-frame tokens + that same P-frame MV tokens.

---

## 4) Active/inactive by phase

```
TRAIN:
🟢 reader -> IFrameEncoder -> MotionPerceiver -> (mv_proj) -> CaptionHead
🔵 Teacher target path (bp_rgb-residual -> rgb_encoder)
🔵 Teacher prediction path (I spatial + MV tokens -> TeacherTransformer)
🔵 losses: CE + MSE + COS (Kendall weighted)

VAL:
🟢 reader -> IFrameEncoder -> MotionPerceiver -> (mv_proj) -> CaptionHead
⚪ teacher target path OFF
⚪ teacher prediction path OFF
🟢 loss fallback: CE only (for val step computation)
🟢 generation loop for caption metrics (BLEU/METEOR/ROUGE/CIDEr/SPICE)
```

---

## 5) One concrete dimension example (B=1, G=8, P=7)

```
iframe        : [1,8,3,224,224]
motion_vector : [1,8,7,4,56,56]

feature_context        : [1,8,512]
feature_context_spatial: [1,8,196,512]
feature_motion         : [1,8,7,8,512]

BGP = 1*8*7 = 56
teacher_predicted: [56,196,512]
teacher_target   : [56,196,512]
```

This is exactly why you see `56,196,512` in teacher MSE branch.
