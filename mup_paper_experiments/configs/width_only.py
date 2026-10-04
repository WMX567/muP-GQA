# Config for varying width only

# Run the experiment:
# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/width_only.py --max_concurrent 30

# Get the results
# python crawl_wandb.py --entity krchickering-uc-davis --project width-only-ablation-lr-decay --output-dir ./mup_paper_experiments/results/ablations/width-only-ablation
# python crawl_wandb.py --entity krchickering-uc-davis --project width-only-ablation-lr-decay-coarse-wd --output-dir ./mup_paper_experiments/results/ablations/width-only-ablation-coarse-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project width-only-ablation-11 --output-dir ./mup_paper_experiments/results/ablations/width-only-ablation-11
import numpy as np
from copy import deepcopy 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true', help='Enable dry run mode')
args = parser.parse_args()

PROD = True

WEIGHT_DECAY = 0.1
MODEL_DEPTH = 8

LEARNING_RATE_SAMPLES = 11 if PROD else 1
learning_rates = [
    # 10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-4.0, -2.0, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43] if PROD else [42]

WANDB_PROJECT = f'width-only-ablation-lr-decay-coarse-wd-prod_{PROD}'
WANDB_PROJECT = f'width-only-ablation-11'

model_configs = [
    {'n_embd': 256,  'n_head': 4,   'n_kv_head': 4,   'n_layer': MODEL_DEPTH, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 20, 'batch_size': 5, 'max_iters': 981 if PROD else 30},
    # {'n_embd': 320,  'n_head': 5,   'n_kv_head': 5,   'n_layer': MODEL_DEPTH, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 58, 'max_iters': 1139 if PROD else 30},
    # {'n_embd': 384,  'n_head': 6,   'n_kv_head': 6,   'n_layer': MODEL_DEPTH, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 2, 'gradient_accumulation_steps': 2, 'batch_size': 66, 'max_iters': 1295 if PROD else 30},
    # {'n_embd': 448,  'n_head': 7,   'n_kv_head': 7,   'n_layer': MODEL_DEPTH, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3, 'gradient_accumulation_steps': 3, 'batch_size': 49, 'max_iters': 1447 if PROD else 30},
    # {'n_embd': 512,  'n_head': 8,   'n_kv_head': 8,   'n_layer': MODEL_DEPTH, 'weight_decay': WEIGHT_DECAY, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 3, 'gradient_accumulation_steps': 3, 'batch_size': 54, 'max_iters': 1597 if PROD else 30},
    # # {'n_embd': 768,  'n_head': 12,  'n_kv_head': 12,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 4, 'gradient_accumulation_steps': 4, 'batch_size': 56, 'max_iters': 2184 if PROD else 30},
    # {'n_embd': 1024, 'n_head': 16,  'n_kv_head': 16,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 5, 'gradient_accumulation_steps': 5, 'batch_size': 56, 'max_iters': 2760 if PROD else 30},
    # {'n_embd': 1280, 'n_head': 20,  'n_kv_head': 20,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 5, 'gradient_accumulation_steps': 5, 'batch_size': 68, 'max_iters': 3331 if PROD else 30},
    # {'n_embd': 1536, 'n_head': 24,  'n_kv_head': 24,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 6, 'gradient_accumulation_steps': 6, 'batch_size': 66, 'max_iters': 3900 if PROD else 30},
    # {'n_embd': 2048, 'n_head': 32,  'n_kv_head': 32,  'n_layer': MODEL_DEPTH, 'weight_decay': 0.0, 'log_wandb': 'true', 'wandb_project': WANDB_PROJECT, 'n_gpus': 8, 'gradient_accumulation_steps': 8, 'batch_size': 64, 'max_iters': 5032 if PROD else 30, 'sbatch_mem': 256},
]

configs = []
for mup in [True]: #, False]:
    for seed in seeds:
        for conf in model_configs:
            for lr in learning_rates:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed

                # Shared parameters
                conf['eval_interval'] = 100
                conf['eval_iters'] = 5
                conf['min_lr'] = lr / 10
                conf['decay_profile'] = 'cosine'
                conf['decay_lr'] = 'true'
                conf['dataset'] = 'openwebtext'

                if mup:
                    conf['mup'] = 'true'
                    conf['mup_multiplier'] = conf['n_embd'] / 256
                    conf['impl'] = 'tpv_left_impl'
                else:
                    conf['mup'] = 'false'
                    conf['mup_multiplier'] = 1
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

# Generate the json for this

