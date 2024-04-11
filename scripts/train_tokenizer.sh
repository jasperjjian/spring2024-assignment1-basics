#!/bin/bash
#SBATCH --partition=batch-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=15:00:00


python3 -u cs336_basics/bpe_tokenizer.py '/data/TinyStoriesV2-GPT4-train.txt' 'results/' 10000 'tinystories_2'
