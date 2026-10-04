# Cosine (coupled weight decay)
# peak_lr
# warmup_ptc
# terminal value

# Integral: 
# 

# WSD (coupled weight decay)
# peak_lr
# warmup_ptc
# cooldown_ptc
# terminal value
# tail shape (cos, linear)

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/wsd_exp.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project wsd-exp-512-width-2020-iters-2 --output-dir ./mup_paper_experiments/results/wsd-exp-512-width-2020-iters-2
import numpy as np
from copy import deepcopy

LEARNING_RATE_SAMPLES = 30
learning_rates = [
    10**p for p in np.linspace(-3.75, -2.5, LEARNING_RATE_SAMPLES)
]
wsd_cooldown_ptcs = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]

base_weight_decay = 0.1

seeds = [44]#, 43, 44]
WANDB_PROJECT = 'wsd-exp-512-width-2020-iters-2'

model_config = {
    'n_embd': 512, 
    'n_head': 8, 
    'n_kv_head': 8, 
    'n_layer': 8, 
    'weight_decay': base_weight_decay, 
    'log_wandb': 'true', 
    'wandb_project': WANDB_PROJECT, 
    'n_gpus': 8, 
    'gradient_accumulation_steps': 8, 
    'batch_size': 64, 
    'max_iters': 2020,
    'mup': 'false',    
}

configs = []
for lr in learning_rates:
    for seed in seeds:
        conf = deepcopy(model_config)
        conf['learning_rate'] = lr
        conf['min_lr'] = lr / 10
        conf['decay_profile'] = 'cosine'
        conf['decay_lr'] = 'true'
        conf['seed'] = seed
        configs.append(conf)
        conf['warmup_iters'] = int(0.1 * model_config['max_iters'])
        conf['lr_decay_iters'] = model_config['max_iters']
        conf['cooldown_iters'] = model_config['max_iters'] # This is just max iters... for plotting lol

        for wsd_cooldown_ptc in wsd_cooldown_ptcs:
            conf = deepcopy(conf)
            conf['decay_profile'] = 'wsd_cosine_tail'
            conf['cooldown_iters'] = int(wsd_cooldown_ptc * conf['max_iters'])
            conf['seed'] = seed
            configs.append(conf)

            conf = deepcopy(conf)
            conf['decay_profile'] = 'wsd'
            configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))