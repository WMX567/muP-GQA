# Experiment for annealing the weight decay parameter

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/anneal_wd.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project cosine-anneal-wd --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/cosine-anneal-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project cosine-anneal-wd-overtrained-2 --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/cosine-anneal-wd-overtrained-2
# python crawl_wandb.py --entity krchickering-uc-davis --project cosine-anneal-wd-valid-shakedown --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/cosine-anneal-wd-overtrained-3
# python crawl_wandb.py --entity krchickering-uc-davis --project wsd-anneal-wd-40TPP --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/wsd-anneal-wd-40TPP
# python crawl_wandb.py --entity krchickering-uc-davis --project wsd-anneal-wd-40TPP-openwebtext --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/wsd-anneal-wd-40TPP-openwebtext
# python crawl_wandb.py --entity krchickering-uc-davis --project wsd-anneal-wd-40TPP-openwebtext-3 --output-dir ./mup_paper_experiments/results/cosine-anneal-wd/wsd-anneal-wd-40TPP-openwebtext-3
import numpy as np
from copy import deepcopy

LEARNING_RATE_SAMPLES = 7
learning_rates = [
    # 10**p for p in np.linspace(-3, -1.2, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-2.75, -1.8, LEARNING_RATE_SAMPLES)
]
# wsd_cooldown_ptcs = [0.1, 0.125, 0.15, 0.175, 0.2]

base_weight_decay = 0.1 / 50

seeds = [42] #, 43, 44] #, 45, 46]
WANDB_PROJECT = 'wsd-anneal-wd-40TPP-openwebtext-reverse'

model_config = {
    'n_embd': 512, 
    'n_head': 8, 
    'n_kv_head': 8, 
    'n_layer': 8, 
    'weight_decay': base_weight_decay, 
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
for lr in learning_rates:
    for seed in seeds:
        conf = deepcopy(model_config)
        conf['learning_rate'] = lr
        conf['min_lr'] = lr / 10
        conf['min_wd'] = base_weight_decay * 50
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


        # for wsd_cooldown_ptc in wsd_cooldown_ptcs:
        #     conf = deepcopy(conf)
        #     conf['decay_profile'] = 'wsd_cosine_tail'
        #     conf['cooldown_iters'] = int(wsd_cooldown_ptc * conf['max_iters'])
        #     conf['seed'] = seed
        #     configs.append(conf)

        #     conf = deepcopy(conf)
        #     conf['decay_profile'] = 'wsd'
        #     configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))