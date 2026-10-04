# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/moe_expert_width_2.py --max_concurrent 30
# python crawl_wandb.py --project moe-ablate-expert-width-large --entity krchickering-uc-davis --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/moe_ablations/moe-ablate-expert-width-large

import numpy as np
from copy import deepcopy 

PROD = True
NO_PROD_ITERS = 100

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'moe-ablate-expert-width-large'

HIDDEN_DIM_BASE = 256
FFN_HIDDEN_DIM_BASE = 64

MOE_SEQ_AUX_LOSS_COEFF = 0.002

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 250,
    'eval_iters': 20,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 512,
    'dataset': 'openwebtext',
    'use_moe': True,
    'block_size': 8192,
    'moe_seq_aux_loss_coeff': MOE_SEQ_AUX_LOSS_COEFF,
    'enable_fsdp': True,
    'compile': False,
    'dtype': 'bfloat16',
    'n_embd': 256,
    'n_head': 4,
    'n_kv_head': 4,
    'n_layer': 3,
    'router_topk': 8,
    'num_experts': 64,
    'sbatch_timeout': '02:00:00' if PROD else '00:05:00',   
    'partition': 'main',
    'qos': 'k2m'
}

model_configs = [
    {
        'moe_ffn_hidden_size': 32,
        'n_gpus': 2,
        'gradient_accumulation_steps': 2,
        'batch_size': 4,
        'max_iters': 774 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 64,
        'n_gpus': 2,
        'gradient_accumulation_steps': 2,
        'batch_size': 5,
        'max_iters': 737 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 128,
        'n_gpus': 3,
        'gradient_accumulation_steps': 3,
        'batch_size': 4,
        'max_iters': 806 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 256,
        'n_gpus': 3,
        'gradient_accumulation_steps': 3,
        'batch_size': 4,
        'max_iters': 1191 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 512,
        'n_gpus': 4,
        'gradient_accumulation_steps': 4,
        'batch_size': 4,
        'max_iters': 1470 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 1024,
        'n_gpus': 6,
        'gradient_accumulation_steps': 6,
        'batch_size': 4,
        'max_iters': 1749 if PROD else NO_PROD_ITERS,
    },
    {
        'moe_ffn_hidden_size': 2048,
        'n_gpus': 8,
        'gradient_accumulation_steps': 8,
        'batch_size': 4,
        'max_iters': 2467 if PROD else NO_PROD_ITERS,
    }
]

LEARNING_RATE_SAMPLES = 9
learning_rates = [
    10**p for p in np.linspace(-3.25, -2, LEARNING_RATE_SAMPLES)
] if PROD else [4e-4]
seeds = [43] if PROD else [42]

configs = []
for mup in [False]: #[True, False] if PROD else [True]:
    for seed in seeds:
        for conf in model_configs:
            for lr in learning_rates:
                conf = deepcopy(conf)
                conf.update(shared_params)

                # if not mup:
                #     lr = lr / (conf['n_embd'] / HIDDEN_DIM_BASE)

                conf['learning_rate'] = lr
                conf['warmup_iters'] = int(0.07 * conf['max_iters'])
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / HIDDEN_DIM_BASE if mup else 1
                conf['impl'] = 'tpv_left_impl_new_kv_2' if mup else 'standard_param_impl'

                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))