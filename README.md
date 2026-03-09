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
