#TRAIN COCAP_HYBRID
cd /home/blaze/Hav-Cocap/Cocap_HYBRID && PYTHONPATH=. python tools/train_net.py --config-name exp/train/msvd_captioning ++ckpt_path='logs/msvd_captioning/lightning_logs/version_1/checkpoints/epoch=0-step=3049.ckpt'



 source /home/blaze/Hav-Cocap/venv/bin/activate && cd /home/blaze/Hav-Cocap/Cocap_Distilled_MAE && PYTHONPATH=. python tools/train_net.py --config-name exp/train/vatex_captioning '++ckpt_path=logs/vatex_captioning/lightning_logs/version_0/checkpoints/epoch=6-step=10675.ckpt'
<!-- 
cd /teamspace/studios/this_studio/Hav-Cocap/Cocap_HYBRID && GLIBC_TUNABLES=glibc.rtld.optional_static_tls=262144 OMP_NUM_THREADS=4 PYTHONPATH=. python tools/train_net.py --config-name exp/train/msvd_captioning ++ckpt_path=’logs/msvd_captioning/lightning_logs/version_2/checkpoints/epoch=0-step=1525.ckpt' -->