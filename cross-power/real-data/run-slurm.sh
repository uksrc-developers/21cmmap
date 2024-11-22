#!/bin/bash

#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --time 12:00:00
#SBATCH --mem 32G
#SBATCH --job-name meerpower
#SBATCH --output ./slurm-out/%J.out

source ~/.bashrc

singularity run $@ 