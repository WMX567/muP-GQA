#!/bin/bash
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

If second argument is 'standard_param_impl', then use the standard parameter implementation.
if [ "$2" == "standard_param_impl" ]; then
    mup_flag="--no_mup"
else
    mup_flag=""
fi


srun python run_lambda_sweep.py \
    --d_emb $1 \
    --impl $2 \
    --exp_name $3 \
    --seed $4 \
    --batch_size $5 \
    --grad_accumulation_steps $6 \
    --n_layers $7 \
    --iters $8 \
    --n_kv_heads 2 \
    ${mup_flag}