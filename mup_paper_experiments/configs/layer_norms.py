# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/layer_norms.py --max_concurrent 30
# python crawl_wandb.py --entity krchickering-uc-davis --project layernorm-study-shakedown --output-dir ./mup_paper_experiments/results/layer-norm-experiment/layernorm-study-shakedown
# python crawl_wandb.py --entity krchickering-uc-davis --project layernorm-study-l2-table-1 --output-dir ./mup_paper_experiments/results/layer-norm-experiment/layernorm-study-l2-table-1
# python crawl_wandb.py --entity krchickering-uc-davis --project layernorm-study-l2-table-2 --output-dir ./mup_paper_experiments/results/layer-norm-experiment/layernorm-study-l2-table-2
# python crawl_wandb.py --entity krchickering-uc-davis --project layernorm-study-l2-table-skinny --output-dir ./mup_paper_experiments/results/layer-norm-experiment/layernorm-study-l2-table-skinny

# Depth 16 or 32, width 128

import numpy as np
from copy import deepcopy

DEBUG = True

# LEARNING_RATE_SAMPLES = 30
# learning_rates = [
#     10**p for p in np.linspace(-4.0, -2.5, LEARNING_RATE_SAMPLES)
# ]

# LEARNING_RATE_SAMPLES = 10
# learning_rates = [
#     10**p for p in np.linspace(-2.5, -0.5, LEARNING_RATE_SAMPLES)
# ]

LEARNING_RATE_SAMPLES = 50 if not DEBUG else 1
learning_rates = [
    10**p for p in np.linspace(-4.25, -1, LEARNING_RATE_SAMPLES)
]

base_weight_decay = 0.1

seeds = [42, 43, 44] if not DEBUG else [42]
WANDB_PROJECT = 'layernorm-study-l2-table-1b'

# model_config = {
#     'n_embd': 512, 
#     'n_head': 8, 
#     'n_kv_head': 8, 
#     'n_layer': 8, 
#     'weight_decay': base_weight_decay, 
#     'log_wandb': 'true', 
#     'wandb_project': WANDB_PROJECT, 
#     'n_gpus': 8, 
#     'gradient_accumulation_steps': 8, 
#     'batch_size': 64, 
#     'max_iters': 2020,
#     'mup': 'false',    
# }

# model_config = {
#     'n_embd': 128, 
#     'n_head': 8, 
#     'n_kv_head': 8, 
#     'n_layer': 64, 
#     'weight_decay': base_weight_decay, 
#     'log_wandb': 'true', 
#     'wandb_project': WANDB_PROJECT, 
#     'n_gpus': 2, 
#     'gradient_accumulation_steps': 2, 
#     'batch_size': 50, 
#     'max_iters': 3882,
#     'mup': 'false',    
# }

model_config = {
    'n_embd': 1408, 
    'n_head': 22, 
    'n_kv_head': 22, 
    'n_layer': 15, 
    'weight_decay': base_weight_decay, 
    'log_wandb': 'true', 
    'wandb_project': WANDB_PROJECT, 
    'n_gpus': 8, 
    'gradient_accumulation_steps': 8, 
    'batch_size': 40, 
    'max_iters': 3882,
    'mup': 'false',    
    'dtype': 'float16',
}

configs = []
for lr in learning_rates:
    for seed in seeds:
        conf = deepcopy(model_config)

        # 1e-7 (check against megatron)

        # Q_LayerNorm_with_element_affine, K_L2Norm
        # Q_LayerNorm_with_element_affine, K_layernorm_with_element_affine (no bias)
        # Q_RMSNorm_with_element_affine,   K_RMSNorm_with_element_affine
        # Q_RMSNorm_with_element_affine,   K_L2Norm
        # Layernorm with bias              RMS 

        conf['learning_rate'] = lr
        conf['min_lr'] = lr / 10
        conf['decay_profile'] = 'cosine'
        conf['decay_lr'] = 'true'
        conf['warmup_iters'] = int(0.1 * model_config['max_iters'])
        conf['lr_decay_iters'] = model_config['max_iters']
        conf['cooldown_iters'] = model_config['max_iters']
        conf['q_prelayer_normalization'] = 'NoNorm'
        conf['k_prelayer_normalization'] = 'NoNorm'
        conf['seed'] = seed

        configs.append(conf)

        # conf = deepcopy(conf)
        # conf['q_prelayer_normalization'] = 'LayerNormWithBias'
        # conf['k_prelayer_normalization'] = 'L2Norm'
        # configs.append(conf)

        # conf = deepcopy(conf)
        # conf['q_prelayer_normalization'] = 'LayerNormWithBias'
        # conf['k_prelayer_normalization'] = 'LayerNorm'
        # configs.append(conf)

        # conf = deepcopy(conf)
        # conf['q_prelayer_normalization'] = 'L2NormScale'
        # conf['k_prelayer_normalization'] = 'L2NormScale'
        # configs.append(conf)

        # conf = deepcopy(conf)
        # conf['q_prelayer_normalization'] = 'L2NormScale'
        # conf['k_prelayer_normalization'] = 'L2Norm'
        # configs.append(conf)

        # conf = deepcopy(conf)
        # conf['q_prelayer_normalization'] = 'LayerNormWithBias'
        # conf['k_prelayer_normalization'] = 'L2NormScale'
        # configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))