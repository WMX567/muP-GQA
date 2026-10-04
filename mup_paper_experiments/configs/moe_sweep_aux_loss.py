# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/moe_sweep_aux_loss.py --max_concurrent 30
# python crawl_wandb.py --project moe-sweep-aux-loss --entity krchickering-uc-davis --output-dir /mnt/weka/home/kyle.chickering/code/nanoGPT/mup_paper_experiments/results/moe_ablations/moe-sweep-aux-loss

import numpy as np
from copy import deepcopy 

PROD = True
NO_PROD_ITERS = 100

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'moe-sweep-aux-loss'

HIDDEN_DIM_BASE = 256
FFN_HIDDEN_DIM_BASE = 64

shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 250,
    'eval_iters': 20,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 512,
    'dataset': 'openwebtext',
    'use_moe': True,
    'block_size': 8192,
    'enable_fsdp': True,
    'compile': False,
    'dtype': 'bfloat16',
    'n_embd': 512,
    'n_head': 8,
    'n_kv_head': 8,
    'n_layer': 4,
    'router_topk': 8,
    'moe_ffn_hidden_size': 96,
    'sbatch_timeout': '00:45:00' if PROD else '00:05:00',
}

model_configs = [
    {
        'num_experts': 16,
        'n_gpus': 3,
        'gradient_accumulation_steps': 3,
        'batch_size': 4,
        'max_iters': 1108 if PROD else NO_PROD_ITERS,
    },
]

LEARNING_RATE_SAMPLES = 7
learning_rates = [
    10**p for p in np.linspace(-4, -2, LEARNING_RATE_SAMPLES)
] if PROD else [4e-4]
seeds = [42] if PROD else [42]

AUX_LOSS_SAMPLES = 7
aux_loss_coeff = [
    10**p for p in np.linspace(-4, 0, AUX_LOSS_SAMPLES)
] if PROD else [1e-4]

configs = []
for mup in [False]: #[True, False] if PROD else [True]:
    for seed in seeds:
        for conf in model_configs:
            for lr in learning_rates:
                for alc in aux_loss_coeff:
                    conf = deepcopy(conf)
                    conf.update(shared_params)

                    # if not mup:
                    #     lr = lr / (conf['n_embd'] / HIDDEN_DIM_BASE)

                    conf['moe_seq_aux_loss_coeff'] = alc
                    conf['learning_rate'] = lr
                    conf['warmup_iters'] = int(0.07 * conf['max_iters'])
                    conf['seed'] = seed
                    conf['min_lr'] = lr / 10
                    conf['mup'] = 'true' if mup else 'false'
                    conf['mup_multiplier'] = conf['n_embd'] / HIDDEN_DIM_BASE if mup else 1
                    conf['impl'] = 'tpv_left_impl_new_kv_2' if mup else 'standard_param_impl'

                    configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))