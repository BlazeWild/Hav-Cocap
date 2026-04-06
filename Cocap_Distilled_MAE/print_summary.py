import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())

from hydra_zen import instantiate
from cocap.modeling.lm_cocap import cocap_lm_cfg

try:
    from torchinfo import summary
    # Instantiate the model from config
    model = instantiate(cocap_lm_cfg)
    
    # Print the summary
    model_summary = summary(model, depth=5, verbose=0, col_names=("num_params", "trainable"))
    print("\n" + "="*80)
    print("MODEL SUMMARY")
    print("="*80)
    print(model_summary)
    print("="*80 + "\n")
except Exception as e:
    print(f"Error: {e}")
