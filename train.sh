#!/usr/bin/env bash
#SBATCH --mem=15G
#SBATCH --time=1-0
#SBATCH --gres=gpu:1

EXPERIMENT=$1
SUFFIX=$2
JSON=experiments/${EXPERIMENT}.json
MODEL=models/${EXPERIMENT}${SUFFIX}

if [[ ! -f ${JSON} ]]; then
    echo "Not found: ${JSON}"
fi

rm -rf ${MODEL}

python run.py train ${JSON} --include-package cyber.dataset_readers --include-package cyber.models -s ${MODEL}
