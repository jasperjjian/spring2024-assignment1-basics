#!/bin/bash
#SBATCH --partition=batch-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00


python3 -u cs336_basics/bpe_tokenizer.py '/data/owt_train.txt' 'results/' 32000 'owt'
