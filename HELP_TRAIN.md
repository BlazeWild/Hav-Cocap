# USE THIS.
1. cloen the repo 
https://github.com/BlazeWild/Hav-Cocap


1. create venv
python3 -m venv venv
2. activate venv
source venv/bin/activate
3. install dependencies
pip install -r requirements.txt
4. download dataset  (first be at the root dir of Hav-Cocap)
cd HavCocap_new/dataset/Charades && gdown "https://drive.google.com/uc?id=1l0rSVRJbGTYwNhHlwqOpRxdYNDeBDx24"
5. Unzip
unzip Charades_filtered_vid_and_caption.zip

6. Train (be at the root dir of Hav-Cocap)
cd HavCocap_new 
export PYTHONPATH=".:$PYTHONPATH"
source ../venv/bin/activate
python tools/train_net.py --config-name exp/train/charades_captioning

#if to train from a certain epoch


cd HavCocap_new 
export PYTHONPATH=".:$PYTHONPATH"
source ../venv/bin/activate
python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path='"logs/charades_captioning/lightning_logs/version_64/checkpoints/epoch=13-step=14126.ckpt"'


cd HavCocap_new 
export PYTHONPATH=".:$PYTHONPATH" 
python tools/train_net.py --config-name exp/train/valor_captioning 

++ckpt_path='"logs/charades_captioning/lightning_logs/version_64/checkpoints/epoch=13-step=14126.ckpt"'


python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path='"logs/charades_captioning/lightning_logs/version_64/checkpoints/epoch=13-step=14126.ckpt"'



source venv/bin/activate && cd HavCocap_COPE && CUDA_LAUNCH_BLOCKING=1 python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path='"logs/charades_captioning/lightning_logs/version_16/checkpoints/epoch=2-step=1515.ckpt"'



cd /teamspace/studios/this_studio/Hav-Cocap/HavCocap_new &&PYTHONPATH=. python tools/train_net.py --config-name exp/train/valor_captioning ++ckpt_path='"logs/msvd_captioning/lightning_logs/version_1/checkpoints/epoch=0-step=3049.ckpt"'



cd /teamspace/studios/this_studio/Hav-Cocap/HavCocap_Aud &&PYTHONPATH=. python tools/train_net.py --config-name exp/train/charades_captioning 


#TRAIN COCAP_NEW
cd /teamspace/studios/this_studio/Hav-Cocap/Cocap_NEW && PYTHONPATH=. python tools/train_net.py --config-name exp/train/msvd_captioning ++ckpt_path='logs/msvd_captioning/lightning_logs/version_1/checkpoints/epoch=0-step=3049.ckpt'

#TRAIN HAVCOCAP_AUD
cd /teamspace/studios/this_studio/Hav-Cocap/HavCocap_Aud && PYTHONPATH=. python tools/train_net.py --config-name exp/train/charades_captioning


# ==============================
# Cocap_Distilled_MAE (VATEX) - 3 Phase Training
# ==============================

# 0) Enter project
cd /teamspace/studios/this_studio/Hav-Cocap/Cocap_Distilled_MAE
export PYTHONPATH=".:$PYTHONPATH"

# ------------------------------------------
# Phase 1 (from epoch 0) - use MP4 videos
# ------------------------------------------
# Uses only videos referenced by dataset/vatex/VATEX_caption.json
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=1 \
	++model.use_preextracted_features=false

# Quick smoke test (single-step). If num_workers=0, set prefetch_factor=null.
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=1 \
	++model.use_preextracted_features=false \
	++trainer.fast_dev_run=1 \
	++train_dataloader.num_workers=0 \
	++val_dataloader.num_workers=0 \
	++train_dataloader.prefetch_factor=null \
	++val_dataloader.prefetch_factor=null \
	++train_dataloader.multiprocessing_context=null \
	++val_dataloader.multiprocessing_context=null

# Resume Phase 1 (full trainer state)
# Recommended: use last.ckpt (no '=' in filename, easy for Hydra parsing)
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=1 \
	++model.use_preextracted_features=false \
	++ckpt_path='logs/vatex_captioning/phase1/checkpoints/best/last.ckpt'

# If you must pass a specific epoch ckpt and its name contains '=',
# escape '=' as '\=' in the override value.
# Example:
# ++ckpt_path='/teamspace/.../best-epochepoch\=002-stepstep\=12.ckpt'

# ------------------------------------------
# Phase 2 (initialize from Phase 1 modules)
# ------------------------------------------
# Default behavior: if no init checkpoint is passed, phase2 auto-loads
# motion encoder module from:
# logs/vatex_captioning/phase1/checkpoints/modules/motion_encoder_best.pt
# (fallback to phase1 last.ckpt only if module file is missing)
# Start Phase 2 fresh optimizer/scheduler, loading ONLY motion from phase1.
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=2 \
	++model.use_preextracted_features=true \
	++model.init_motion_ckpt='logs/vatex_captioning/phase1/checkpoints/modules/motion_encoder_best.pt'

# Simple default phase2 start (auto handoff from phase1 motion_encoder_best.pt)
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=2

# Resume Phase 2 fully
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=2 \
	++ckpt_path='logs/vatex_captioning/phase2/checkpoints/best/last.ckpt'

# Note:
# - For phase transitions (phase1->phase2, phase2->phase3), prefer init checkpoints
#   (model.init_weights_ckpt / init_motion_ckpt / init_mgdtr_ckpt).
# - ckpt_path is for full-state resume of the SAME phase.

# ------------------------------------------
# Phase 3
# ------------------------------------------
# Default behavior: if no init checkpoint is passed, phase3 auto-loads
# motion from phase1 modules and MGDTR from phase2 modules
# (fallback to phase2 last.ckpt only if module files are missing)
# Default project-relative paths checked in code:
# logs/vatex_captioning/phase1/checkpoints/modules/motion_encoder_best.pt
# logs/vatex_captioning/phase2/checkpoints/modules/mgdtr_selector_best.pt
# Note: "Option A" and "Option B" are labels. Do not type the label as a command.
# Option A (required if NO full phase3 ckpt): provide motion + MGDTR checkpoints (compulsory)
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=3 \
	++model.use_preextracted_features=false \
	++model.init_motion_ckpt='logs/vatex_captioning/phase1/checkpoints/modules/motion_encoder_best.pt' \
	++model.init_mgdtr_ckpt='logs/vatex_captioning/phase2/checkpoints/modules/mgdtr_selector_best.pt'

# Option B (resume phase3 full model): full ckpt is enough, no module ckpts needed
python tools/train_net.py --config-name exp/train/vatex_captioning \
	++model.phase=3 \
	++ckpt_path='logs/vatex_captioning/phase3/checkpoints/best/last.ckpt'

# You can pass module paths manually (.pt or .ckpt both supported):
# ++model.init_motion_ckpt='.../motion_encoder_best.pt'
# ++model.init_mgdtr_ckpt='.../mgdtr_selector_best.pt'

