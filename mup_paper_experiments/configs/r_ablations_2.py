# Check that we get transfer across the GQA parameter r
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/r_ablations_2.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-tpvl-zoom-less-wd-hs-64 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablate-gqa-repetition-tpvl-zoom-less-wd-hs-64
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-mengxi-tpv-left-2 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablate-gqa-repetition-mengxi-tpv-left
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-24 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablate-gqa-repetition-24
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-kyle-impl --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablate-gqa-repetition-kyle-impl
# python crawl_wandb.py --entity krchickering-uc-davis --project ablate-gqa-repetition-kyle-impl-2 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablate-gqa-repetition-kyle-impl-2

import numpy as np
from copy import deepcopy 

PROD = True

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'ablate-gqa-repetition-kyle-impl-2'

config = {
    'n_embd': 384,
    'n_layer': 5,
    'n_head': 8,
    'n_gpus': 3,
    'gradient_accumulation_steps': 3,
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
    'partition': 'main',
    'qos': 'k2m'
}

# Keep everything pinned to 5 TPP.
bs_iters = {
    1: {'batch_size': 60, 'max_iters': 762 if PROD else 30},
    2: {'batch_size': 60, 'max_iters': 762 if PROD else 30},
    3: {'batch_size': 60, 'max_iters': 763 if PROD else 30},
    4: {'batch_size': 60, 'max_iters': 763 if PROD else 30},
    6: {'batch_size': 60, 'max_iters': 765 if PROD else 30},
    8: {'batch_size': 60, 'max_iters': 767 if PROD else 30},
    12: {'batch_size': 61, 'max_iters': 770 if PROD else 30},
    24: {'batch_size': 61, 'max_iters': 780 if PROD else 30},
}

# 5 TPP
# bs_iters = {
#     1: {'batch_size': 60, 'max_iters': 763 if PROD else 30},
#     2: {'batch_size': 60, 'max_iters': 766 if PROD else 30},
#     4: {'batch_size': 61, 'max_iters': 770 if PROD else 30},
#     8: {'batch_size': 61, 'max_iters': 781 if PROD else 30},
# }

# repetitions = [1, 2, 3, 4, 6, 8, 12, 24]
repetitions = [1, 2, 4, 8]
learning_rate_samples = 11 if PROD else 1
# learning_rates = [10**p for p in np.linspace(-3.5, -1.0, learning_rate_samples)]
learning_rates = [10**p for p in np.linspace(-3, -1.75, learning_rate_samples)] if PROD else [1e-2]

seeds = [42, 43, 44] if PROD else [42]
impls = ['kyle_impl'] #['tpv_left_impl_new_kv_2', 'standard_param_impl'] #'tpv_left_impl_new_kv_2', 'mengxi_impl', 'standard_param_impl']

configs = []
for seed in seeds:
    for rep in repetitions:
        for lr in learning_rates:
            for impl in impls:
                conf = deepcopy(config)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                conf['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                conf['mup_multiplier'] = 1.0 
                conf['impl'] = impl
                conf['min_lr'] = lr / 10
                conf['n_kv_head'] = config['n_head'] // rep
                conf['batch_size'] = bs_iters[config['n_head']]['batch_size']
                conf['max_iters'] = bs_iters[config['n_head']]['max_iters']
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))