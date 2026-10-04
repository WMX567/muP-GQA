# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/wd_impulse.py --max_concurrent 56
# wd-impulse-owt
# python crawl_wandb.py --entity krchickering-uc-davis --project wd-impulse-owt --output-dir ./mup_paper_experiments/results/ablations/wd-impulse-owt

import numpy as np
from copy import deepcopy 

PROD = True

LEARNING_RATE = 0.005623
WANDB_PROJECT = 'wd-impulse-owt'

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'learning_rate': LEARNING_RATE,
    'min_lr': LEARNING_RATE / 10,
    'avg_interval': 120,
    'sbatch_mem': 256,
    'dataset': 'openwebtext',
    'mup': 'false',
}

model_config = {
    'n_embd': 256, 
    'n_head': 8, 
    'n_kv_head': 2, 
    'n_layer': 7,
    'n_gpus': 2,
    'gradient_accumulation_steps': 2,
    'batch_size': 53,
    'max_iters': 1361 if PROD else 30,
}

WD_SAMPLES = 11 if PROD else 1
weight_decay_values = [10**p for p in np.linspace(-.5, .5, WD_SAMPLES)]

seeds = [42, 43]

configs = []
for wd in weight_decay_values:
    for seed in seeds:
        conf = deepcopy(model_config)
        conf['weight_decay'] = wd
        conf['seed'] = seed

        conf.update(shared_params)

        configs.append(conf)   

if __name__ == "__main__":
    import json
    import random

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))