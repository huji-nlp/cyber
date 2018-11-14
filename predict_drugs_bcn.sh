#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=0-1

python run.py predict models/drugs_bcn/model.tar.gz $1
