#!/usr/bin/env bash
#SBATCH --mem=15G
#SBATCH --time=0-6
#SBATCH --gres=gpu:1
#SBATCH --array=0-99

JSONS=($(cat experiments.txt))
if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
  JSONS=(${JSONS[${SLURM_ARRAY_TASK_ID}]})
fi

for JSON in ${JSONS[@]}; do
    SETTING="$(basename $(dirname ${JSON}))"
    EXPERIMENT="$(basename ${JSON} .json)"
    mkdir -p "models/${SETTING}"
    MODEL="models/${SETTING}/${EXPERIMENT}"
    python run.py train "${JSON}" --include-package cyber.dataset_readers --include-package cyber.models -s "${MODEL}" \
        || grep Error "${MODEL}/stderr.log" >> accuracy.tsv
    echo -e "${SETTING}\t${EXPERIMENT}\t$(grep -h test_accuracy ${MODEL}/stdout.log | sed 's/.*://;s/,//;s/ //g')" \
        >> accuracy.tsv
done