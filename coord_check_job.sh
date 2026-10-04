#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=coord.out

eval "$(conda shell.bash hook)"
conda activate nanogpt

now=$(date +%Y-%m-%d_%H-%M-%S)
out_dir=coord-check-impl/${now}
mkdir -p ${out_dir}

widths=(512 768 1024 1536 2048 3072 4096)
depths=(3 3 3 3 3 3 3)
heads=(4 6 8 12 16 24 32)
kv_heads=(4 3 4 4 8 8 8)
kv_reps=(1 2 2 3 2 3 4)

for seed in {0..3}; do
    for i in ${!widths[@]}; do
        emb=${widths[$i]}
        n_layer=${depths[$i]}
        n_heads=${heads[$i]}
        n_kv_heads=${kv_heads[$i]}
        kv_rep=${kv_reps[$i]}
        mup_multiplier=$((emb / 512))

        echo "mup_muliplier: ${mup_multiplier}, n_heads: ${n_heads}, n_kv_heads: ${n_kv_heads}, emb: ${emb}, n_layer: ${n_layer}, seed: ${seed}"
        srun python coord_check.py \
            --out_dir=${out_dir} \
            --eval_interval=10000000 \
            --log_interval=10000000 \
            --eval_iters=1 \
            --eval_only=False \
            --init_from='scratch' \
            --wandb_log=False \
            --dataset='openwebtext' \
            --gradient_accumulation_steps=1 \
            --batch_size=1 \
            --block_size=1024 \
            --n_layer=${n_layer} \
            --n_head=${n_heads} \
            --n_kv_head=${n_kv_heads} \
            --n_embd=${emb} \
            --dropout=0.0 \
            --bias=False \
            --init_std=0.02 \
            --learning_rate=4e-5 \
            --max_iters=4 \
            --weight_decay=0.0 \
            --beta1=0.9 \
            --beta2=0.95 \
            --grad_clip=1.0 \
            --decay_lr=False \
            --mup=True \
            --mup_multiplier=${mup_multiplier} \
            --seed=${seed} \
            --backend='nccl' \
            --device='cuda' \
            --dtype='float32' \
            --compile=False \
            --impl='mengxi_impl'
    done
done