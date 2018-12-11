#!/usr/bin/env bash
#SBATCH --mem=40G
#SBATCH --time=1-0

python run.py train experiments/drugs_bcn_without_cuda.json  --include-package cyber.dataset_readers --include-package cyber.models -s models/drugs_bcn
