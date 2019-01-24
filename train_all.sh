#!/usr/bin/env bash
#SBATCH --mem=15G
#SBATCH --time=3-0
#SBATCH --gres=gpu:1

for JSON in experiments/*/*.json; do
    EXPERIMENT=$(basename ${JSON} .json)
    MODEL=models/${EXPERIMENT}
    echo -n ${EXPERIMENT} >> f1.txt
    python run.py train ${JSON} --include-package cyber.dataset_readers --include-package cyber.models -s ${MODEL}
    grep -h test_f1 ${MODEL}/stdout.log | sed 's/.*://;s/,//' >> f1.txt
done