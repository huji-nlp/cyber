#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=0-1
#SBATCH --gres=gpu:1

EXPERIMENT=$1
INPUT=$2
MODEL=models/${EXPERIMENT}/model.tar.gz

if [[ ! -f ${MODEL} ]]; then
    echo "Not found: ${MODEL}"
fi

python run.py predict --include-package cyber.dataset_readers --include-package cyber.models \
    --include-package cyber.predictors ${MODEL} ${INPUT}
