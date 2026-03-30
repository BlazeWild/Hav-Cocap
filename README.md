---
license: mit
task_categories:
- video-classification
- audio-classification
- image-to-text
pretty_name: Hav-Cocap AVCaps Dataset
tags:
- video-captioning
- audio-visual
- multimodal
- cocap
size_categories:
- 10K<n<100K
---

# Hav-Cocap AVCaps Dataset

This dataset contains the complete CoCap and Hav-Cocap implementations for audio-visual captioning.

## Contents

- **CoCap/**: Original CoCap framework implementation
- **Hav-Cocap/**: Enhanced Hav-Cocap implementation with improvements
- **requirements.txt**: Python dependencies for the project

## Project Structure

### CoCap
- `cocap/`: Core modules for audio-visual captioning
  - `data/`: Dataset loading and preprocessing
  - `modeling/`: Model architectures and training
  - `modules/`: Audio encoder, CLIP, BEATs, etc.
  - `utils/`: Utility functions
- `configs/`: Configuration files for different datasets (MSRVTT, MSVD, VATEX)
- `tools/`: Training and evaluation scripts
- `model_zoo/`: Pre-trained model checkpoints

### Hav-Cocap
- `dataset/AVCaps/`: AVCaps dataset with videos and captions
  - Train/Val/Test splits
  - Video files (240p H.264 format)
  - Caption annotations (JSON format)
- `learn/`: Educational notebooks and examples
- `model/`: Model architecture implementations

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Details

The AVCaps dataset includes:
- Audio-visual caption pairs
- Multiple splits (train/val/test)
- Preprocessed video files
- JSON caption annotations

## Usage

See the training scripts in `CoCap/tools/` for training examples:
- `train_avcaps.py`: Main training script for AVCaps dataset
- `train_net.py`: General training script

## Citation

If you use this dataset or code, please cite the original CoCap paper and this dataset.

## License

MIT License - See LICENSE file for details















_    __TransformerEncoder: 2-6                                              --
_    _    __ModuleList: 3-17                                                --
_    _    _    __TransformerEncoderLayer: 4-13                              5,513,984
_    __Linear: 2-7                                                          590,592
__Dropout: 1-5                                                              --
====================================================================================================
Total params: 204,140,800
Trainable params: 117,553,408
Non-trainable params: 86,587,392
====================================================================================================
==================================

Epoch 3: 100%|_| 8072/8072 [32:36<00:00,  4.13it/s, v_num=1, loss_step=172.0, Bleu_1=0.00133, Bleu_2=1.2e-11, Bleu_3=2.56e-14, Bleu_4=1.2e-15, METEOR=0.00447, ROUGE_L=                                                               PTBTokenizer tokenized 31386 tokens at 244854.87 tokens per second.___________________________________________________________________________________________________________________________________| 874/874 [21:17<00:00,  0.68it/s]
PTBTokenizer tokenized 12651 tokens at 114614.12 tokens per second.
{'testlen': 11983, 'reflen': 9838, 'guess': [11983, 11445, 10907, 10369], 'correct': [14, 0, 0, 0]}
ratio: 1.2180321203495408
Epoch 4:   0%|                                | 0/8072 [00:00<?, ?it/s, v_num=1, loss_step=172.0, Bleu_1=0.00117, Bleu_2=1.01e-11, Bleu_3=2.11e-14, Bleu_4=9.75e-16, METEOR=0.00708, ROUGE_L=0.00133, CIDEr=0.000171, loss_epoch=204.0]--- EPOCH 4 START ---                                                                                                                                                                                                                  Phase 2: End-to-End Fine-Tuning - TinyStories is TRAINING
Epoch 4: 100%|_____________________| 8072/8072 [33:10<00:00,  4.06it/s, v_num=1, loss_step=165.0, Bleu_1=0.00117, Bleu_2=1.01e-11, Bleu_3=2.11e-14, Bleu_4=9.75e-16, METEOR=0.00708, ROUGE_L=0.00133, CIDEr=0.000171, loss_epoch=204.0PTBTokenizer tokenized 31386 tokens at 253352.10 tokens per second.___________________________________________________________________________________________________________________________________| 874/874 [21:13<00:00,  0.69it/s]
PTBTokenizer tokenized 8818 tokens at 83007.04 tokens per second.
{'testlen': 8202, 'reflen': 8324, 'guess': [8202, 7664, 7143, 6630], 'correct': [6, 0, 0, 0]}
ratio: 0.9853435848148744
Epoch 5:   0%|                              | 0/8072 [00:00<?, ?it/s, v_num=1, loss_step=165.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14, Bleu_4=1.17e-15, METEOR=0.00559, ROUGE_L=0.000615, CIDEr=0.000116, loss_epoch=201.0]--- EPOCH 5 START ---                                                                                                                                                                                                                  
Phase 2: End-to-End Fine-Tuning - TinyStories is TRAINING
Epoch 5:  66%|_____________      | 5344/8072 [22:12<11:20,  4.01it/s, v_num=1, loss_step=411.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14, Bleu_4=1.17e-15, METEpoch 5:  66%|_| 5345/8072 [22:13<11:20,  4.01it/s, v_num=1, loss_step=411.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14, Bleu_4=1.17e-15, METEOR=0.00559, ROUGEEpoch 5:  67%|_| 5406/8072 [22:30<11:06,  4.00it/s, v_num=1, loss_step=159.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14,Epoch 5:  69%|_____________      | 5553/8072 [23:06<10:29,  4.00it/s, v_num=1, loss_step=154.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14, Bleu_4=1.17e-15, METEOR=0.00559, ROUGE_L=0.000615, CIDEr=0.000116, loss_epoch=201.0]
Detected KeyboardInterrupt, attempting graceful shutdown ...
Epoch 5:  69%|_____________      | 5553/8072 [23:08<10:29,  4.00it/s, v_num=1, loss_step=154.0, Bleu_1=0.000721, Bleu_2=9.63e-12, Bleu_3=2.34e-14, Bleu_4=1.17e-15, METEOR=0.00559, ROUGE_L=0.000615, CIDEr=0.000116, loss_epoch=201.0]
(venv) (base) blaze@cslam-training:~/Hav-Cocap/HavCocap_COPE$ 