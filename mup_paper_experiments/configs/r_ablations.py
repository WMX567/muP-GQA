# Check that we get transfer across the GQA parameter r
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/r_ablations.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition --output-dir ./mup_paper_experiments/results/ablations/ablate-gqa-repetition
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-tpvl-zoom --output-dir ./mup_paper_experiments/results/ablations/ablate-gqa-repetition-tpvl-zoom
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-tpvl-zoom-less-wd --output-dir ./mup_paper_experiments/results/ablations/ablate-gqa-repetition-tpvl-zoom-less-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-tpvl-zoom-less-wd-2 --output-dir ./mup_paper_experiments/results/ablations/ablate-gqa-repetition-tpvl-zoom-less-wd-2

import numpy as np
from copy import deepcopy 

PROD = True

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'ablate-gqa-repetition-tpvl-zoom-less-wd-3'

config = {
    'n_embd': 384,
    'n_layer': 5,
    'n_head': 24,
    'n_gpus': 2,
    'gradient_accumulation_steps': 2,
    'dtype': 'bfloat16',
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 256,
    'dataset': 'openwebtext',
}

# Keep everything pinned to 8 TPP. Otherwise the smallest model hits ~8.4 TPP :/
bs_iters = {
    1: {'batch_size': 90, 'max_iters': 1219 if PROD else 30},
    2: {'batch_size': 90, 'max_iters': 1219 if PROD else 30},
    3: {'batch_size': 90, 'max_iters': 1221 if PROD else 30},
    4: {'batch_size': 90, 'max_iters': 1221 if PROD else 30},
    6: {'batch_size': 90, 'max_iters': 1223 if PROD else 30},
    8: {'batch_size': 90, 'max_iters': 1228 if PROD else 30},
    12: {'batch_size': 92, 'max_iters': 1232 if PROD else 30},
    24: {'batch_size': 92, 'max_iters': 1248 if PROD else 30},
}

repetitions = [1, 2, 3, 4, 6, 8, 12, 24]
learning_rate_samples = 11 if PROD else 1
# learning_rates = [10**p for p in np.linspace(-3.75, -1.0, learning_rate_samples)]
learning_rates = [10**p for p in np.linspace(-2.5, -1.5, learning_rate_samples)] if PROD else [1e-2]

seeds = [42, 43, 44] if PROD else [42]
# seeds = [45, 46, 47] if PROD else [42]
# impls = ['standard_param_impl', 'tpv_left_impl', 'tpv_left_impl_new_kv']
impls = ['tpv_left_impl']

configs = []
for impl in impls:
    for rep in repetitions:
        for lr in learning_rates:
            for seed in seeds:
                conf = deepcopy(config)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                conf['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                conf['mup_multiplier'] = 1.0 
                conf['impl'] = impl
                conf['min_lr'] = lr / 10
                conf['n_kv_head'] = 24 // rep
                conf['batch_size'] = bs_iters[rep]['batch_size']
                conf['max_iters'] = bs_iters[rep]['max_iters']
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))