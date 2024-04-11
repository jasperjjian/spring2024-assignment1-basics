#!/bin/bash
#SBATCH --partition=batch-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36G
#SBATCH --time=1:00:00


python3 cs336_basics/bpe_tokenizer.py '/data/' 'results' 300 'tester'