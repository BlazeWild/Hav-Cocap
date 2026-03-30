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





python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path='"logs/charades_captioning/lightning_logs/version_64/checkpoints/epoch=13-step=14126.ckpt"'



source venv/bin/activate && cd HavCocap_COPE && CUDA_LAUNCH_BLOCKING=1 python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path='"logs/charades_captioning/lightning_logs/version_16/checkpoints/epoch=2-step=1515.ckpt"'
