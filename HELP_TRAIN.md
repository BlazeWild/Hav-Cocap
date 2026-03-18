python tools/train_net.py +exp/train=charades_captioning


python HavCocap_new/tools/train_net.py --config-name exp/train/charades_captioning


cd /home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new
export PYTHONPATH="/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new:$PYTHONPATH"
source /home/ashok/Documents/blaze/Hav-Cocap/venv/bin/activate && python tools/train_net.py --config-name exp/train/charades_captioning



cd /home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new
export PYTHONPATH="/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new:$PYTHONPATH"
export PATH="/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new/temp_bin:$PATH"
source ../venv/bin/activate


source venv/bin/activate && python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path=\'/HavCocap_new/logs/charades_captioning/lightning_logs/version_54/checkpoints/epoch=5-step=6054.ckpt\'




# USE THIS.

cd /home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new
export PYTHONPATH="/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new:$PYTHONPATH"
source ../venv/bin/activate
python tools/train_net.py --config-name exp/train/charades_captioning ++ckpt_path=\'/home/ashok/Documents/blaze/Hav-Cocap/HavCocap_new/logs/charades_captioning/lightning_logs/version_54/checkpoints/epoch=5-step=6054.ckpt\'