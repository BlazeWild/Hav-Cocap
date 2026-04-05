#!/bin/bash
set -euo pipefail

echo "Downloading BEATs Iter3+ (AS2M) Model..."
wget "https://huggingface.co/datasets/Blazewild/processed_msvd_8fps/resolve/main/Cocap/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?download=true" -O BEATs_iter3_plus_AS2M.pt

if [[ ! -s BEATs_iter3_plus_AS2M.pt ]]; then
	echo "Download failed or produced empty file: model_zoo/BEATs_iter3_plus_AS2M.pt"
	exit 1
fi

echo "Downloaded successfully to model_zoo/BEATs_iter3_plus_AS2M.pt"
