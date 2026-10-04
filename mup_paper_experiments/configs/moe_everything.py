# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/moe_everything.py --max_concurrent 30

import numpy as np
from copy import deepcopy 

PROD = False
NO_PROD_ITERS = 100

WEIGHT_DECAY = 0.05
WANDB_PROJECT = 'moe-everything'

HIDDEN_DIM_BASE = 256
FFN_HIDDEN_DIM_BASE = 64

MOE_SEQ_AUX_LOSS_COEFF = 1e-4

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
    'moe_seq_aux_loss_coeff': MOE_SEQ_AUX_LOSS_COEFF,
    'enable_fsdp': True,
    'compile': False,
    'dtype': 'bfloat16',
}

model_configs = [
    # {
    #     'n_embd': 256,
    #     'n_head': 8,
    #     'n_kv_head': 2,
    #     'n_layer': 4,
    #     'num_experts': 32,
    #     'router_topk': 3,
    #     'moe_ffn_hidden_size': 64,
    #     'n_gpus': 2,
    #     'gradient_accumulation_steps': 2,
    #     'batch_size': 5,
    #     'max_iters': 652 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '01:00:00' if PROD else '00:10:00',
    # },
    # {
    #     'n_embd': 384,
    #     'n_head': 12,
    #     'n_kv_head': 2,
    #     'n_layer': 5,
    #     'num_experts': 48,
    #     'router_topk': 3,
    #     'moe_ffn_hidden_size': 64,
    #     'n_gpus': 2,
    #     'gradient_accumulation_steps': 2,
    #     'batch_size': 6, # Peaked so far at B=14, T=8192. Optimal throughput at T=8192 for 2 GPUs occurs at batch size 
    #     'max_iters': 1017 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '01:00:00' if PROD else '00:45:00',
    # },
    # {
    #     'n_embd': 512,
    #     'n_head': 8,
    #     'n_kv_head': 2,
    #     'n_layer': 6,
    #     'num_experts': 64,
    #     'router_topk': 4,
    #     'moe_ffn_hidden_size': 80,
    #     'n_gpus': 3,
    #     'gradient_accumulation_steps': 3,
    #     'batch_size': 5,
    #     'max_iters': 1504 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '01:00:00' if PROD else '00:45:00',
    # },
    # # ----------------------------------------
    # #     These stictly need FSDP
    # # ----------------------------------------
    # {
    #     'n_embd': 768,
    #     'n_head': 12,
    #     'n_kv_head': 3,
    #     'n_layer': 7,
    #     'num_experts': 80,
    #     'router_topk': 5,
    #     'moe_ffn_hidden_size': 96,
    #     'n_gpus': 4,
    #     'gradient_accumulation_steps': 4,
    #     'batch_size': 6,
    #     'max_iters': 2022 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    # },
    # {
    #     'n_embd': 1024,
    #     'n_head': 8,
    #     'n_kv_head': 4,
    #     'n_layer': 12,
    #     'num_experts': 96,
    #     'router_topk': 6,
    #     'moe_ffn_hidden_size': 128,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 3,
    #     'max_iters': 2957 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    # },
    # {
    #     'n_embd': 1536,
    #     'n_head': 12,
    #     'n_kv_head': 3,
    #     'n_layer': 15,
    #     'num_experts': 112,
    #     'router_topk': 6,
    #     'moe_ffn_hidden_size': 192,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 3,
    #     'gradient_accumulation_steps': 24,
    #     'batch_size': 3,
    #     'max_iters': 5912 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    # },
    {
        'n_embd': 2048,
        'n_head': 8,
        'n_kv_head': 4,
        'n_layer': 18,
        'num_experts': 128,
        'router_topk': 7,
        'moe_ffn_hidden_size': 256,
        'n_gpus': 8,
        'sbatch_nodes': 6,
        'gradient_accumulation_steps': 48,
        'batch_size': 2,
        'max_iters': 10512 if PROD else NO_PROD_ITERS,
        'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    },
    {
        'n_embd': 3072,
        'n_layer': 21,
        'n_head': 12,
        'n_kv_head': 3,
        'moe_ffn_hidden_size': 384,
        'num_experts': 144,
        'router_topk': 7,
        'n_gpus': 8,
        'sbatch_nodes': 16,
        'gradient_accumulation_steps': 128,
        'batch_size': 1,
        'max_iters': 22329 if PROD else NO_PROD_ITERS,
        'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    },
    # {
    #     'n_embd': 4096,
    #     'n_layer': 24,
    #     'n_head': 8,
    #     'n_kv_head': 4,
    #     'moe_ffn_hidden_size': 512,
    #     'num_experts': 160,
    #     'router_topk': 8,
    #     'n_gpus': 8,
    #     # 'sbatch_nodes': 6,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 1,
    #     'max_iters': 50000 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    # },
    # {
    #     'n_embd': 6144,
    #     'n_layer': 27,
    #     'n_head': 8,
    #     'n_kv_head': 4,
    #     'moe_ffn_hidden_size': 768,
    #     'num_experts': 192,
    #     'router_topk': 8,
    #     'n_gpus': 8,
    #     # 'sbatch_nodes': 7,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 1,
    #     'max_iters': 50000 if PROD else NO_PROD_ITERS,
    #     'sbatch_timeout': '36:00:00' if PROD else '00:45:00',
    # }

]

LEARNING_RATE_SAMPLES = 11
learning_rates = [
    10**p for p in np.linspace(-3.5, -1.5, LEARNING_RATE_SAMPLES)
] if PROD else [4e-4]
seeds = [42, 43] if PROD else [42]

configs = []
for mup in [False]: #[True, False] if PROD else [True]:
    for seed in seeds:
        for conf in model_configs:
            for lr in learning_rates:
                if not mup:
                    lr = lr / (conf['n_embd'] / HIDDEN_DIM_BASE)
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['warmup_iters'] = int(0.07 * conf['max_iters'])
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / HIDDEN_DIM_BASE if mup else 1
                conf['impl'] = 'tpv_left_impl_new_kv_2' if mup else 'standard_param_impl'

                conf.update(shared_params)
                configs.append(conf)

if __name__ == "__main__":
    import json
    import random

    # random.shuffle(configs)

    config_strings = []
    for config in configs:
        config_strings.append(json.dumps(config))

    print('\n'.join(config_strings))