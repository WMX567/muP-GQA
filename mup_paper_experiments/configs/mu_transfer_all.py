# Config for the sweeps over all parameters. Sweeping to large models
# We adjust learning rate to track where the minimum is to save compute


# Width: 128 -> 1280
# Depth: 4 -> 18
# head_size: 16, 32, 64
# n_kv_heads: 2 -> 5

# python mup_paper_experiments/build_orchastrator.py --config_generator_file mup_paper_experiments/configs/mu_transfer_all.py --max_concurrent 30
# everything-runs
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs --output-dir ./mup_paper_experiments/results/ablations/everything-runs
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-2 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-2
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-3 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-3
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-no-wd --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-no-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-small-wd --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-small-wd
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-2 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-2
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-3 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-3
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-4 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-4
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-5 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-5
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-6 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-6
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-7 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-7
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-debug-const-data-250k --output-dir ./mup_paper_experiments/results/ablations/everything-runs-debug-const-data-250k
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-0.075 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-0.075
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-0.0075 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-0.0075
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-0.03 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-0.03
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-owt --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-owt
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-owt-2 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-owt-2
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-owt-3 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-owt-3
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-old-kv-owt --output-dir ./mup_paper_experiments/results/ablations/everything-runs-old-kv-owt
# python crawl_wandb.py --entity krchickering-uc-davis --project everything-runs-new-kv-owt-4 --output-dir ./mup_paper_experiments/results/ablations/everything-runs-new-kv-owt-4



shared_params = {
    'log_wandb': 'true',
    'wandb_project': WANDB_PROJECT,
    'eval_interval': 50,
    'eval_iters': 100,
    'decay_profile': 'cosine',
    'decay_lr': 'true',
    'weight_decay': WEIGHT_DECAY,
    'avg_interval': 120,
    'sbatch_mem': 256,
    'dataset': 'openwebtext',
}

model_configs = [
    # {
    #     'n_embd': 128, 
    #     'n_head': 4, 
    #     'n_kv_head': 2, 
    #     'n_layer': 4,
    #     'n_gpus': 1,
    #     'gradient_accumulation_steps': 1,
    #     'batch_size': 68,
    #     'max_iters': 868 if PROD else 30,
    # },
    # {
    #     'n_embd': 256, 
    #     'n_head': 8, 
    #     'n_kv_head': 2, 
    #     'n_layer': 7,
    #     'n_gpus': 2,
    #     'gradient_accumulation_steps': 2,
    #     'batch_size': 53,
    #     'max_iters': 1361 if PROD else 30,
    # },
    # {
    #     'n_embd': 384, 
    #     'n_head': 12, 
    #     'n_kv_head': 4, 
    #     'n_layer': 10,
    #     'n_gpus': 3,
    #     'gradient_accumulation_steps': 3,
    #     'batch_size': 60,
    #     'max_iters': 1589 if PROD else 30,
    # },
    # {
    #     'n_embd': 512, 
    #     'n_head': 16, 
    #     'n_kv_head': 4, 
    #     'n_layer': 13,
    #     'n_gpus': 3,
    #     'gradient_accumulation_steps': 3,
    #     'batch_size': 66,
    #     'max_iters': 2536 if PROD else 30,
    # },
    # {
    #     'n_embd': 768,
    #     'n_head': 16,
    #     'n_kv_head': 4,
    #     'n_layer': 16,
    #     'n_gpus': 5,
    #     'gradient_accumulation_steps': 5,
    #     'batch_size': 59,
    #     'max_iters': 3795 if PROD else 30,
    # },
    # {
    #     'n_embd': 1024, 
    #     'n_head': 16, 
    #     'n_kv_head': 8, 
    #     'n_layer': 19,
    #     'n_gpus': 6,
    #     'sbatch_nodes': 1,
    #     'gradient_accumulation_steps': 6,
    #     'batch_size': 69,
    #     'max_iters': 5321 if PROD else 30,
    #     'sbatch_mem': 256,
    #     'dtype': 'bfloat16',
    # },
    {
        'n_embd': 1024, 
        'n_head': 16, 
        'n_kv_head': 4, 
        'n_layer': 19,
        'n_gpus': 8,
        'sbatch_nodes': 1,
        'gradient_accumulation_steps': 6,
        'batch_size': 68,
        'max_iters': 5222 if PROD else 30,
        'sbatch_mem': 256,
        'dtype': 'bfloat16',
    },
    # {
    #     'n_embd': 1280, 
    #     'n_head': 20, 
    #     'n_kv_head': 10, 
    #     'n_layer': 22,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 34,
    #     'max_iters': 6944 if PROD else 30,
    #     'sbatch_mem': 256,
    #     'dtype': 'bfloat16',
    # },
    {
        'n_embd': 1280, 
        'n_head': 20, 
        'n_kv_head': 5, 
        'n_layer': 22,
        'n_gpus': 8,
        'sbatch_nodes': 1,
        'gradient_accumulation_steps': 8,
        'batch_size': 66,
        'max_iters': 6807 if PROD else 30,
        'sbatch_mem': 256,
        'dtype': 'bfloat16',
    },
    # Below this line the models have not had their TPP updated yet :/ 
    # Skip first 1536 and the 2048s for 8TPP. Need 2 SlimPJ chunks
    # {
    #     'n_embd': 1536, 
    #     'n_head': 12, 
    #     'n_kv_head': 6, 
    #     'n_layer': 25,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 51,
    #     'max_iters': 1919 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 1536, 
    #     'n_head': 12, 
    #     'n_kv_head': 3, 
    #     'n_layer': 25,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 16,
    #     'batch_size': 42,
    #     'max_iters': 8536 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2048, 
    #     'n_head': 16, 
    #     'n_kv_head': 8, 
    #     'n_layer': 28,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 36,
    #     'max_iters': 3762 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2048, 
    #     'n_head': 16, 
    #     'n_kv_head': 4, 
    #     'n_layer': 28,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 35,
    #     'max_iters': 3682 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2560, 
    #     'n_head': 10, 
    #     'n_kv_head': 5, 
    #     'n_layer': 31,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 29,
    #     'max_iters': 3329 if PROD else 30,
    #     'sbatch_mem': 256
    # },
    # {
    #     'n_embd': 2560, 
    #     'n_head': 10, 
    #     'n_kv_head': 2, 
    #     'n_layer': 31,
    #     'n_gpus': 8,
    #     'sbatch_nodes': 2,
    #     'gradient_accumulation_steps': 32,
    #     'batch_size': 28,
    #     'max_iters': 3264 if PROD else 30,
    #     'sbatch_mem': 256
    # },
]

# Try this range for -6, -1...
LEARNING_RATE_SAMPLES = 11 if PROD else 1
learning_rates = [
    # 10**p for p in np.linspace(-4.25, -2.5, LEARNING_RATE_SAMPLES)
    # 10**p for p in np.linspace(-4.0, -2.0, LEARNING_RATE_SAMPLES)
    # 10**p for p in np.linspace(-4.5, -1.5, LEARNING_RATE_SAMPLES)
    10**p for p in np.linspace(-3.5, -1.5, LEARNING_RATE_SAMPLES)
]
seeds = [42, 43] #, 44, 45, 46] if PROD else [42]

configs = []
for mup in [True]: #[True, False] if PROD else [True]:
    for seed in seeds:
        for conf in model_configs:
            for lr in learning_rates:
                conf = deepcopy(conf)
                conf['learning_rate'] = lr
                conf['seed'] = seed
                conf['min_lr'] = lr / 10
                conf['mup'] = 'true' if mup else 'false'
                conf['mup_multiplier'] = conf['n_embd'] / 384 if mup else 1
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