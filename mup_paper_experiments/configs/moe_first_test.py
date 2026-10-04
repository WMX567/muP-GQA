# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/moe_first_test.py --max_concurrent 30

from copy import deepcopy 

PROD = True

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'moe-test'

shared_params = {
    'log_wandb': 'true',
    'n_embd': 768,
    'n_head': 12,
    'n_kv_head': 12,
    'n_layer': 5,
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 10,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 256,
    'mup': False,
    'dataset': 'openwebtext',
    'use_moe': True,
    'router_topk': 8,
    'learning_rate': 3.5e-3,
    'n_gpus': 4,
    'sbatch_nodes': 1,
    'batch_size': 90,
    'gradient_accumulation_steps': 4,
    'max_iters': 250,
    'warmup_iters': 10,
    'lr_decay_iters': 250,
    'dtype': 'float32',
}

configs = []
for n_experts in [8, 16, 32, 64]:
    config = deepcopy(shared_params)
    config['num_experts'] = n_experts
    configs.append(config)

if __name__ == "__main__":
    import json
    import random

    random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))


