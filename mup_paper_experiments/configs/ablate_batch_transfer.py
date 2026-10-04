# Ablate that my weight decay scaling allows transferring over batch size at constant data
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/ablate_batch_transfer.py --max_concurrent 30

import numpy as np
from copy import deepcopy 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Enable dry run mode')
args = parser.parse_args()

PROD = not args.dry_run

WANDB_PROJECT = 'ablate-transfer-on-batches'

config = {
    'n_embd': 384,
    'n_layer': 5,
    'n_head': 8,
    'n_kv_head': 8,
    'n_gpus': 2,
    'gradient_accumulation_steps': 2,
    'dtype': 'bfloat16',
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'avg_interval': 120,
    'sbatch_mem': 256,
    'dataset': 'openwebtext',
}

global_batch_size = [60, 90, 140, 180, 240]
weight_decay_values = [0.05, 0.1, 0.25, 0.9]

learning_rate_samples = 11 if PROD else 1
learning_rates = [10**p for p in np.linspace(-3.75, -2.5, learning_rate_samples)]

seeds = [42, 43, 44] if PROD else [42]
impls = ['tpv_left_impl', 'standard_param_impl']

configs = []
for impl in impls:
    for wd in weight_decay_values:
        for bs in global_batch_size:
            for lr in learning_rates:
                for seed in seeds:
                    conf = deepcopy(config)
                    conf['weight_decay'] = wd
                    conf['learning_rate'] = lr
                    conf['seed'] = seed
                    conf['mup'] = 'false' if impl == 'standard_param_impl' else 'true'
                    conf['mup_multiplier'] = 1.0
                    conf['batch_size'] = bs // 2
                    conf['max_iters'] = int(610 * (180 / bs)) if PROD else 30
                    configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))