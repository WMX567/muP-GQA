# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/reverse_large.py --max_concurrent 15

import numpy as np
from copy import deepcopy

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'reverse-large-sweep'

LEARNING_RATE_SAMPLES = 2 # 12
learning_rates = [
    10**p for p in np.linspace(-4.5, -2.5, LEARNING_RATE_SAMPLES)
]
seeds = [42] #, 43, 44]

model_config_small = {
    'n_embd': 512,  
    'n_head': 8,  
    'n_kv_head': 8,  
    'n_layer': 4, 
    'n_gpus': 2,
    'gradient_accumulation_steps': 2,
    'batch_size': 30, 
    'max_iters': 30,
    'weight_decay': WEIGHT_DECAY, 
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'complete_p_layers': 'true',
    'mup': 'true',
    'mup_multiplier': 1/4,
    'impl': 'tpv_left_impl',
}

model_config_large = {
    'n_embd': 1536, 
    'n_head': 24, 
    'n_kv_head': 24, 
    'n_layer': 8, 
    'n_gpus': 8,
    'sbatch_nodes': 1, # 2 x 8 GPUs = 16 GPUs total
    'gradient_accumulation_steps': 16,
    'batch_size': 10,
    'max_iters': 30,
    'weight_decay': WEIGHT_DECAY,
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'complete_p_layers': 'true',
    'mup': 'false',
    'impl': 'standard_param_impl',
    'dtype': 'float16',
}

configs = []
for lr in learning_rates:
    for seed in seeds:
        for conf in [model_config_small, model_config_large]:
            conf = deepcopy(conf)
            conf['learning_rate'] = lr
            conf['seed'] = seed
            configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))