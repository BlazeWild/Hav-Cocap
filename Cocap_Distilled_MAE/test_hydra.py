import hydra
from hydra import compose, initialize

with initialize(version_base=None, config_path="configs"):
    cfg = compose(config_name="exp/train/vatex_captioning")
    print(cfg.model.cocap_model)
