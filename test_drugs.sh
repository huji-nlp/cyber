#!/usr/bin/env bash
#SBATCH --mem=20G
#SBATCH --time=0-1

python run.py evaluate models/drugs/model.tar.gz test
