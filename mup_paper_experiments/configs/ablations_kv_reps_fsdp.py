# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/ablations_kv_reps_fsdp.py --max_concurrent 30 


import numpy as np
from model_family_configs import (
    kv_reps
)

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'ablations_kv_reps_fsdp-1'

PROD = True

base_configs = [
    *kv_reps,
]
configs = []

learning_rate_samples = 11
learning_rates_sp = [
    10**p for p in np.linspace(-3.8, -2.25, learning_rate_samples)
] if PROD else [1e-4]
learning_rates_mup = [
    10**p for p in np.linspace(-3.15, -1.5, learning_rate_samples)
] if PROD else [1e-4]

seeds = [43, 44, 45] if PROD else [43]

for seed in seeds:
    for cfg in base_configs:
        for impl in ['kyle_impl', 'xllm_impl']: #, 'standard_param_impl']:
            mup = 'false' if impl == 'standard_param_impl' else 'true'
            for lr in learning_rates_sp if mup == 'false' else learning_rates_mup:
                new_config = cfg.get_config(prod=PROD)
                new_config['wandb_run_name'] = f"{new_config['wandb_run_name']}-lr_{lr}-wd_{WEIGHT_DECAY}-seed_{seed}"
                new_config['weight_decay'] = WEIGHT_DECAY
                new_config['learning_rate'] = lr
                new_config['min_lr'] = lr / 10
                new_config['log_wandb'] = 'true'
                new_config['wandb_project'] = WANDB_PROJECT
                new_config['seed'] = seed
                new_config['mup'] = mup
                new_config['decay_lr'] = 'true'
                new_config['decay_profile'] = 'cosine'
                new_config['impl'] = impl

                new_config['mup_multiplier'] = 1.0 # Width and depth are constant

                new_config['eval_interval'] = 250
                new_config['eval_iters'] = 50
                new_config['eval_interval'] = 250

                new_config['enable_fsdp'] = 'true'

                configs.append(new_config)


if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
