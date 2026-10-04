# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/ablations_run_baselines.py --max_concurrent 30 

import numpy as np

from model_family_configs import (
    width_only,
    depth_only,
    joint_width_depth,
    head_size,
    kv_reps,
    joint_gqa
)

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'ablations_baselines'

PROD = False

base_configs = [
    *width_only, 
    *depth_only, 
    *joint_width_depth, 
    *joint_gqa, 
    *head_size, 
    *kv_reps
]
configs = []

learning_rate_samples = 11
learning_rates = [
    10**p for p in np.linspace(-3.2, -1.5, learning_rate_samples)
] if PROD else [1e-3]

seeds = [42, 43, 44] if PROD else [42]

for cfg in base_configs:
    for lr in learning_rates:
        for seed in seeds:
            new_config = cfg.get_config(prod=PROD)
            new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr}-wd_{WEIGHT_DECAY}-seed_{seed}"
            new_config['weight_decay'] = WEIGHT_DECAY
            new_config['learning_rate'] = lr
            new_config['log_wandb'] = 'true'
            new_config['wandb_project'] = WANDB_PROJECT
            new_config['seed'] = seed
            new_config['mup'] = 'false'

            configs.append(new_config)

if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
