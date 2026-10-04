# Experiment for annealing the weight decay parameter

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/owdaadam.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project owdaadam-first-test --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/owdaadam-first-test
# 
import numpy as np
from copy import deepcopy

PROD = True

LEARNING_RATE_SAMPLES = 20 if PROD else 1
learning_rates = [
    10**p for p in np.linspace(-3.5, -1.5, LEARNING_RATE_SAMPLES)
]
base_weight_decay = [
    0.01, 0.05, 0.1, 0.15, 0.2
]

seeds = [42, 43, 44, 45, 46] if PROD else [42]
WANDB_PROJECT = 'owdaadam-first-test'

# 128 width x 16 depth

model_config = {
    'n_embd': 512, 
    'n_head': 8, 
    'n_kv_head': 8, 
    'n_layer': 8, 
    'log_wandb': 'true', 
    'wandb_project': WANDB_PROJECT, 
    'n_gpus': 8, 
    'gradient_accumulation_steps': 8, 
    'batch_size': 64, 
    'max_iters': 2020,
    'wd_warmup_iters': 750,  
    'wd_anneal_iters': 750,  
    'eval_interval': 400,
    'eval_iters': 40,
    'mup': 'false',    
}

configs = []
for lr in learning_rates:
    for seed in seeds:
        for wd in base_weight_decay:
            conf = deepcopy(model_config)
            conf['learning_rate'] = lr
            conf['min_lr'] = lr / 10
            conf['weight_decay'] = wd
            conf['decay_profile'] = 'cosine'
            conf['decay_lr'] = 'true'
            conf['seed'] = seed
            conf['warmup_iters'] = int(0.1 * model_config['max_iters'])
            conf['lr_decay_iters'] = model_config['max_iters']
            conf['cooldown_iters'] = model_config['max_iters'] # This is just max iters... for plotting lol
            conf['anneal_wd'] = 'false'
            configs.append(conf)

            conf = deepcopy(conf)
            conf['adaptive_optimizer'] = 'true'
            configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))