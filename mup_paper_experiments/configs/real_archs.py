# Config for varying width only
import numpy as np
from copy import deepcopy

WEIGHT_DECAY = 0.0

LEARNING_RATE_SAMPLES = 17
learing_rates = [
    10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43, 44, 45, 46]

WANDB_PROJECT = 'width-only-sweep'

model_configs = [
    {'n_embd': 512, 'n_head': 8, 'n_kv_head': 8, 'n_layer': 27, 'weight_decay': 0.0, 'log_wandb': 'True', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 81, 'max_iters': 639},
]

configs = []
for mup in [True]:#, False]:
    for lr in learing_rates:
        for seed in seeds:
            for conf in model_configs:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                if mup:
                    conf['mup'] = 'True'
                    conf['mup_multiplier'] = conf['n_embd'] / 256
                    conf['impl'] = 'tpv_left_impl'
                else:
                    conf['mup'] = 'False'
                    conf['mup_multiplier'] = 1
                    conf['impl'] = 'standard_param_impl'
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))

# Generate the json for this

