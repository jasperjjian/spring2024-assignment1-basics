#!/bin/bash
#SBATCH --partition=batch-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1:00:00


python3 bpe_tokenizer.py '/data/TinyStoriesV2-GPT4-train.txt' 'results' 10000 'tinystories'