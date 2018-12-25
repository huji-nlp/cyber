#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=0-1
#SBATCH --gres=gpu:1

EXPERIMENT=$1
MODEL=models/${EXPERIMENT}

if [[ ! -f ${MODEL} ]]; then
    echo "Not found: ${MODEL}"
fi

python run.py evaluate ${MODEL}/model.tar.gz test
