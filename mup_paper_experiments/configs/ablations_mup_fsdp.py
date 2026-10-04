# Config for varying width only
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/ablations_mup_fsdp.py --max_concurrent 30 
# python crawl_wandb.py --entity krchickering-uc-davis --project ablations_mup_fsdp-wo-9 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablations_mup_fsdp-wo-9
# python crawl_wandb.py --entity krchickering-uc-davis --project ablations_mup_fsdp-no-fsdp --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablations_mup_fsdp-no-fsdp

# python crawl_wandb.py --entity krchickering-uc-davis --project ablations_mup_fsdp-wo-11 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablations_mup_fsdp-wo-11
# python crawl_wandb.py --entity krchickering-uc-davis --project ablations_mup_fsdp-do-2 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablations_mup_fsdp-do-2
# python crawl_wandb.py --entity krchickering-uc-davis --project ablations_mup_fsdp-jwd-10 --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/ablations/ablations_mup_fsdp-jwd-10

import numpy as np

from model_family_configs import (
    width_only,
    depth_only,
    joint_width_depth,
)

WEIGHT_DECAY = 0.1
WANDB_PROJECT = 'ablations_mup_fsdp-jwd-12'

PROD = True

base_configs = [
    *joint_width_depth, 
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
        for mup in ['true', 'false']:
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
                new_config['impl'] = 'kyle_impl' if mup == 'true' else 'standard_param_impl'
                new_config['mup_multiplier'] = new_config['n_embd'] / 384 if mup == 'true' else 1.0

                new_config['eval_interval'] = 250
                new_config['eval_iters'] = 50
                new_config['eval_interval'] = 250

                new_config['enable_fsdp'] = 'true'
                new_config['partition'] = 'main'
                new_config['qos'] = 'k2m'

                configs.append(new_config)

if __name__ == "__main__":
    import json

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))
