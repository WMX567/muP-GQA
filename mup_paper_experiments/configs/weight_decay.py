# Weight decay ablation experiment
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/weight_decay.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-new --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-new
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-3 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-4 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-4
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-5 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-5
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-6 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-6
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-0-embedding-wd --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-0-embedding-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-big-3 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-big-3
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-sweep-big-4 --output-dir ./mup_paper_experiments/results/ablations/wd-sweep-big-4

import numpy as np
from copy import deepcopy

PROD = True

MODEL_DEPTH = 4

LEARNING_RATE_SAMPLES = 7 if PROD else 1
learning_rates = [
    10**p for p in np.linspace(-3.25, -1.5, LEARNING_RATE_SAMPLES)
]
WEIGHT_DECAY_SAMPLES = 9 if PROD else 1
weight_decays = [
    10**p for p in np.linspace(-2.5, -0.46, WEIGHT_DECAY_SAMPLES)
]

seeds = [42, 43] if PROD else [42]

WANDB_PROJECT = 'wd-sweep-big-4'

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 100,
    'eval_iters': 25,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'avg_interval': 120,
    'dataset': 'openwebtext',
    'sbatch_timeout': '03:00:00' if PROD else '00:45:00',   
    'partition': 'main',
    'qos': 'k2m',
    'block_size': 8192,
    'enable_fsdp': 'true',
}

# model_configs = [
#     {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 61, 'max_iters': 709 },
#     {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,  'n_gpus': 2, 'gradient_accumulation_steps': 2,  'batch_size': 63, 'max_iters': 874 },
#     {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32, 'n_gpus': 6, 'gradient_accumulation_steps': 6,  'batch_size': 66, 'max_iters': 2756, 'sbatch_mem': 256},
# ]

# Constant data. Keeps training curves "stacked"
model_configs = [
    {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,  'n_gpus': 2, 'n_layer': 4,  'gradient_accumulation_steps': 2,  'batch_size': 3, 'max_iters': 1353 if PROD else 30},
    {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,  'n_gpus': 4, 'n_layer': 6,  'gradient_accumulation_steps': 4,  'batch_size': 2, 'max_iters': 1759 if PROD else 30 },
    {'n_embd': 1024, 'n_head': 16,  'n_kv_head': 16, 'n_gpus': 8, 'n_layer': 8,  'gradient_accumulation_steps': 8,  'batch_size': 2, 'max_iters': 3249 if PROD else 30 },
    # {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32, 'n_gpus': 8, 'n_layer': 10, 'sbatch_nodes': 2, 'gradient_accumulation_steps': 16,  'batch_size': 2, 'max_iters': 6484, 'sbatch_mem': 1024},
    # {'n_embd': 4096, 'n_head': 64,  'n_kv_head': 64, 'n_gpus': 8, 'n_layer': 12, 'gradient_accumulation_steps': 8,  'batch_size': 1, 'max_iters': 1250 if PROD else 30, 'sbatch_mem': 1024},
]

configs = []
for seed in seeds:
    # for impl in ['tpv_left_impl', 'tpv_left_impl_unit_wd', 'standard_param_impl'] if PROD else ['tpv_left_impl']: #, 'tpv_left_impl', 'tpv_left_impl_unit_wd']:
    for cfg in model_configs:
        for impl in ['tpv_left_impl_unit_wd']: # ['tpv_left_impl']
            for lr in learning_rates:
                for wd in weight_decays:
                    conf = deepcopy(cfg)
                    conf['learning_rate'] = lr
                    conf['weight_decay'] = wd
                    conf['seed'] = seed

                    conf['min_lr'] = lr / 10

                    if impl == 'standard_param_impl':
                        conf['mup'] = 'false'
                        conf['mup_multiplier'] = 1.0
                        conf['impl'] = 'standard_param_impl'
                    else:
                        conf['mup'] = 'true'
                        conf['mup_multiplier'] = float(conf['n_embd'] / 384)
                        conf['impl'] = impl
                        
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
