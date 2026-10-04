# Experiment for annealing the weight decay parameter

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/anneal_wd_at_fixed_lr.py --max_concurrent 30
import numpy as np
from copy import deepcopy

LEARNING_RATE_SAMPLES = 7
weight_decay = [0] + [
    # 10**p for p in np.linspace(-3, -1.2, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-4, 0.5, LEARNING_RATE_SAMPLES)
]
# wsd_cooldown_ptcs = [0.1, 0.125, 0.15, 0.175, 0.2]

seeds = [42] #, 43, 44] #, 45, 46]
WANDB_PROJECT = 'wsd-anneal-comp-fixed-lr'

model_config = {
    'n_embd': 512, 
    'n_head': 8, 
    'n_kv_head': 8, 
    'n_layer': 8, 
    'learning_rate': 0.004,
    'min_lr': 0.0004,
    'log_wandb': 'true', 
    'wandb_project': WANDB_PROJECT, 
    'n_gpus': 4, 
    'gradient_accumulation_steps': 4, 
    'batch_size': 128, 
    'max_iters': 6060,
    'wd_warmup_iters': 1500,  
    'wd_anneal_iters': 1500,  
    'eval_interval': 400,
    'eval_iters': 40,
    'mup': 'false',
    'data': 'openwebtext',    
}

configs = []
for wd in weight_decay:
    for seed in seeds:
        conf = deepcopy(model_config)
        conf['weight_decay'] = wd
        conf['min_wd'] = wd / 10
        conf['decay_profile'] = 'cosine'
        conf['decay_lr'] = 'true'
        conf['seed'] = seed
        conf['warmup_iters'] = int(0.1 * model_config['max_iters'])
        conf['lr_decay_iters'] = model_config['max_iters']
        conf['cooldown_iters'] = model_config['max_iters'] # This is just max iters... for plotting lol
        conf['anneal_wd'] = 'false'
        configs.append(conf)

        conf = deepcopy(conf)
        conf['anneal_wd'] = 'true'
        configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))