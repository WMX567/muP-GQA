# Config for the sweeps over all parameters. Sweeping to large models
# We adjust learning rate to track where the minimum is to save compute


# Width: 128 -> 1280
# Depth: 4 -> 18
# head_size: 16, 32, 64
# n_kv_heads: 2 -> 5

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/test_fsdp.py --max_concurrent 30
# everything-runs
# crawl_wandb.py --entity krchickering-uc-davis --project fsdp-shakedown --output-dir ./mup_paper_experiments/results/fsdp/fsdp-shakedown

import numpy as np
from copy import deepcopy 

PROD = False

WEIGHT_DECAY = 0.005
WANDB_PROJECT = 'fsdp-shakedown'

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'use_fsdp': 'true',
    'sbatch_mem': 768,
}

model_configs = [
    # {
    #     'n_embd': 128, 
    #     'n_head': 4, 
    #     'n_kv_head': 2, 
    #     'n_layer': 4,
    #     'n_gpus': 1,
    #     'gradient_accumulation_steps': 1,
    #     'batch_size': 82,
    #     'max_iters': 191 if PROD else 30,
    # },
    # {
    #     'n_embd': 256, 
    #     'n_head': 8, 
    #     'n_kv_head': 2, 
    #     'n_layer': 7,
    #     'n_gpus': 2,
    #     'gradient_accumulation_steps': 2,
    #     'batch_size': 64,
    #     'max_iters': 300 if PROD else 30,
    # },
    # {
    #     'n_embd': 384, 
    #     'n_head': 12, 
    #     'n_kv_head': 4, 
    #     'n_layer': 10,
    #     'n_gpus': 4,
    #     'gradient_accumulation_steps': 4,
    #     'batch_size': 45,
    #     'max_iters': 298 if PROD else 30,
    # },
    # {
    #     'n_embd': 512, 
    #     'n_head': 16, 
    #     'n_kv_head': 4, 
    #     'n_layer': 13,
    #     'n_gpus': 4,
    #     'gradient_accumulation_steps': 4,
    #     'batch_size': 60,
    #     'max_iters': 558 if PROD else 30,
    # },
    # {
    #     'n_embd': 768,
    #     'n_head': 16,
    #     'n_kv_head': 4,
    #     'n_layer': 16,
    #     'n_gpus': 6,
    #     'gradient_accumulation_steps': 6,
    #     'batch_size': 60,
    #     'max_iters': 836 if PROD else 30,
    # },
    # {
    #     'n_embd': 1024, 
    #     'n_head': 16, 
    #     'n_kv_head': 8, 
    #     'n_layer': 19,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 1,
    #     'gradient_accumulation_steps': 8,
    #     'batch_size': 63,
    #     'max_iters': 1172 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1024, 
    #     'n_head': 16, 
    #     'n_kv_head': 4, 
    #     'n_layer': 19,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 1,
    #     'gradient_accumulation_steps': 8,
    #     'batch_size': 61,
    #     'max_iters': 1150 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1280, 
    #     'n_head': 20, 
    #     'n_kv_head': 10, 
    #     'n_layer': 22,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 41,
    #     'max_iters': 1529 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1280, 
    #     'n_head': 20, 
    #     'n_kv_head': 5, 
    #     'n_layer': 22,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 40,
    #     'max_iters': 1499 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1536, 
    #     'n_head': 12, 
    #     'n_kv_head': 6, 
    #     'n_layer': 25,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 51,
    #     'max_iters': 1919 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1536, 
    #     'n_head': 12, 
    #     'n_kv_head': 3, 
    #     'n_layer': 25,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 50,
    #     'max_iters': 1880 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2048, 
    #     'n_head': 16, 
    #     'n_kv_head': 8, 
    #     'n_layer': 28,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 36,
    #     'max_iters': 3762 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2048, 
    #     'n_head': 16, 
    #     'n_kv_head': 4, 
    #     'n_layer': 28,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 35,
    #     'max_iters': 3682 if PROD else 30,
    # },
    # {
    #     'n_embd': 2560, 
    #     'n_head': 10, 
    #     'n_kv_head': 5, 
    #     'n_layer': 31,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 29,
    #     'max_iters': 3329 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    {
        'n_embd': 2560, 
        'n_head': 10, 
        'n_kv_head': 2, 
        'n_layer': 31,
        'n_gpus': 8,
        'sbatch_nodes': 2,
        'gradient_accumulation_steps': 32,
        'batch_size': 28,
        'max_iters': 3264 if PROD else 30,
    },
]

# Try this range for -6, -1...
LEARNING_RATE_SAMPLES = 11 if PROD else 1
learning_rates = [
    # 10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
    # 10**p for p in np.linspace(-4.0, -2.0, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-4.5, -1.5, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43, 44, 45, 46] if PROD else [42]

configs = []
for mup in [True]: #[True, False] if PROD else [True]:
    for lr in learning_rates:
        for seed in seeds:
            for conf in model_configs:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / 384 if mup else 1
                conf['impl'] = 'tpv_left_impl' if mup else 'standard_param_impl'

                conf.update(shared_params)
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))