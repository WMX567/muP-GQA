import numpy as np
from copy import deepcopy

WEIGHT_DECAY = 0.0
WANDB_PROJECT = 'depth-only-sweep-4'

LEARNING_RATE_SAMPLES = 17
learning_rates = [
    10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43, 44]

model_configs = [
    # {'n_embd': 256, 'n_head': 4, 'n_kv_head': 4, 'n_layer': 4,  'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 50, 'max_iters': 817},
    # {'n_embd': 320, 'n_head': 5, 'n_kv_head': 5, 'n_layer': 4,  'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3, 'gradient_accumulation_steps': 3, 'batch_size': 55, 'max_iters': 649},
    # {'n_embd': 384, 'n_head': 6, 'n_kv_head': 6, 'n_layer': 4,  'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3, 'gradient_accumulation_steps': 3, 'batch_size': 62, 'max_iters': 727},
    # # {'n_embd': 256, 'n_head': 4, 'n_kv_head': 4, 'n_layer': 16, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 46, 'max_iters': 715},
    # # {'n_embd': 320, 'n_head': 5, 'n_kv_head': 5, 'n_layer': 16, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 54, 'max_iters': 847},
    # # {'n_embd': 384, 'n_head': 6, 'n_kv_head': 6, 'n_layer': 16, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 62, 'max_iters': 976},
    # {'n_embd': 256, 'n_head': 4, 'n_kv_head': 4, 'n_layer': 64, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 72, 'max_iters': 1125},
    # {'n_embd': 320, 'n_head': 5, 'n_kv_head': 5, 'n_layer': 64, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 5, 'gradient_accumulation_steps': 5, 'batch_size': 70, 'max_iters': 1378},
    {'n_embd': 384, 'n_head': 6, 'n_kv_head': 6, 'n_layer': 64, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 6, 'gradient_accumulation_steps': 6, 'batch_size': 69, 'max_iters': 1630},
]

configs = []
for mup in [True, False]:
    for lr in learning_rates:
        for seed in seeds:
            for conf in model_configs:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                if mup:
                    conf['mup'] = 'true' 
                    conf['mup_multiplier'] = 1.0
                    conf['impl'] = 'tpv_left_impl'
                else:
                    conf['mup'] = 'false'
                    conf['mup_multiplier'] = 1.0
                    conf['impl'] = 'standard_param_impl'
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))