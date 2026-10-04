# Config for the sweeps over all parameters. Sweeping to large models
# We adjust learning rate to track where the minimum is to save compute


# Width: 128 -> 1280
# Depth: 4 -> 18
# head_size: 16, 32, 64
# n_kv_heads: 2 -> 5

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/head_size.py --max_concurrent 30
# everything-runs
# python crawl_wandb.py --entity krchickering-uc-davis --project kv-heads-head-size-ablations --output-dir ./mup_paper_experiments/results/ablations/kv-heads-head-size-ablations

import numpy as np
from copy import deepcopy 

PROD = True

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'kv-heads-head-size-ablations'

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 75,
}

model_configs = [
    {
        'n_embd': 768,
        'n_head': 16,
        'n_kv_head': 4,
        'n_layer': 16,
        'n_gpus': 4,
        'gradient_accumulation_steps': 4,
        'batch_size': 90,
        'max_iters': 1000 if PROD else 30,
        'sbatch_mem': 256
    },
]

# Try this range for -6, -1...
LEARNING_RATE_SAMPLES = 5 if PROD else 1
learning_rates = [
    # 10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-2.5, -1.0, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43] if PROD else [42]

head_sizes = [8, 16, 24, 32, 48, 64]

configs = []
for mup in [True, False] if PROD else [True]:
    for lr in learning_rates:
        for seed in seeds:
            for head_size in head_sizes:
                for conf in model_configs:
                    conf = deepcopy(conf)
                    conf['learning_rate'] = lr
                    conf['seed'] = seed
                    conf['min_lr'] = lr / 10
                    conf['mup'] = 'true' if mup else 'false'
                    conf['mup_multiplier'] = conf['n_embd'] / 384 if mup else 1
                    conf['impl'] = 'tpv_left_impl' if mup else 'standard_param_impl'

                    n_heads = 768 // head_size
                    conf['n_head'] = n_heads
                    conf['n_kv_head'] = n_heads

                    conf.update(shared_params)
                    configs.append(conf)

                    conf = deepcopy(conf)
                    conf['n_kv_head'] = n_heads // 2
                    configs.append(conf)

                    conf = deepcopy(conf)
                    conf['n_kv_head'] = n_heads // 4
                    configs.append(conf)

                    conf = deepcopy(conf)
                    conf['n_kv_head'] = 4
                    configs.append(conf)


if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))