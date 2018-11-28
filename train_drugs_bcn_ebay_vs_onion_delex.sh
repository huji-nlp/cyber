#!/usr/bin/env bash
#SBATCH --mem=30G
#SBATCH --time=1-0

python run.py train experiments/drugs_bcn_ebay_vs_onion_delex_without_cuda.json  --include-package drugs.dataset_readers --include-package drugs.models -s models/drugs_bcn_delex
