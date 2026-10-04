#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=test.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/ablations_kv_reps_fsdp.py --max_concurrent 6