"""
    Simple script to handle command line arguments for training a model.
    We could write this in Shell but I prefer the argument parsing from Python.
    Anyways, this should allow one to fully parameterize and execute training
    runs from the command line, as well as utilizing SLURM. One nice thing here
    is that we can request more GPUs for larger runs to attempt to get higher
    MFUs.
"""

import argparse
import datetime
import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def gpt_params(seq_len, vocab_size, d_model, num_heads, num_layers):
    """ Given GPT config calculate total number of parameters """
    ffw_size = 4*d_model # in GPT the number of intermediate features is always 4*d_model
    # token and position embeddings
    embeddings = d_model * vocab_size + d_model * seq_len
    # transformer blocks
    attention = 3*d_model**2 + 3*d_model # weights and biases
    attproj = d_model**2 + d_model
    ffw = d_model*(ffw_size) + ffw_size
    ffwproj = ffw_size*d_model + d_model
    layernorms = 2*2*d_model
    # dense
    ln_f = 2*d_model
    dense = d_model*vocab_size # note: no bias here
    total_params = num_layers*(attention + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    # return total_params
    return total_params

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()
# Sbatch arguments
parser.add_argument('--sbatch_timeout', type=str, default='6:00:00')
parser.add_argument('--sbatch_nodes', type=int, default=1)
parser.add_argument('--sbatch_exclusive', action='store_true')
parser.add_argument('--n_gpus', type=int, default=2) # This will be nproc_per_node for torchrun
parser.add_argument('--cpus-per-task', type=int, default=8) # This will be cpus-per-task for srun
parser.add_argument('--sbatch_logging_dir', type=str, default=os.path.join(PROJECT_ROOT, 'mup_paper_experiments', 'slurm_logs'))
parser.add_argument('--sbatch_mem', type=int, default=32)  # Memory in GB
parser.add_argument('--partition', type=str, default='gpu')
parser.add_argument('--qos', type=str, default=None) #'lowprio')

# Model testbed arguments
parser.add_argument('--out_dir', type=str, default=os.path.join(PROJECT_ROOT, 'mup_paper_experiments', 'model_training', now))
parser.add_argument('--log_wandb', action='store_true')
parser.add_argument('--wandb_run_name', type=str, default='gpt')
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--backend', type=str, default='nccl')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--dtype', type=str, default='float16')
parser.add_argument('--compile', action='store_true')
parser.add_argument('--coord_check', action='store_true')

# Evaluation parameters
parser.add_argument('--eval_interval', type=int, default=10000000000)
parser.add_argument('--log_interval', type=int, default=1)
parser.add_argument('--avg_interval', type=int, default=30)
parser.add_argument('--eval_iters', type=int, default=300)
parser.add_argument('--eval_only', action='store_true')

# Initialization and dataset
parser.add_argument('--init_from', type=str, default='scratch')
parser.add_argument('--dataset', type=str, default='openwebtext')
parser.add_argument('--block_size', type=int, default=1024)

# Model dynamics arguments. By default max_iters is determined by the prespecified TPP
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument('--max_iters', type=int, default=None)
parser.add_argument('--decay_profile', type=str, default='cosine')  # ['cosine', 'wsd', 'wsd_cosine_tail']
parser.add_argument('--lr_decay_iters', type=int, default=None)
parser.add_argument('--cooldown_iters', type=int, default=1000)  # For WSD decay profile
parser.add_argument('--tpp', type=int, default=5)
parser.add_argument('--warmup_iters', type=int, default=None)
parser.add_argument('--anneal_wd', action='store_true', help='Enable weight decay annealing')
parser.add_argument('--min_wd', type=float, default=0.0, help='Minimum weight decay for annealing')
parser.add_argument('--wd_warmup_iters', type=int, default=1000, help='Weight decay warmup iterations')
parser.add_argument('--wd_anneal_iters', type=int, default=1000, help='Weight decay anneal iterations')
parser.add_argument('--adaptive_optimizer', action='store_true', help='Enable adaptive optimizer')
parser.add_argument('--use_fsdp', action='store_true', help='Enable FSDP (Fully Sharded Data Parallel)')

# Model training parameters
parser.add_argument('--n_layer', type=int, default=12)
parser.add_argument('--n_head', type=int, default=16)
parser.add_argument('--n_kv_head', type=int, default=16)
parser.add_argument('--n_embd', type=int, default=1024)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bias', action='store_true')
parser.add_argument('--init_std', type=float, default=0.02)
parser.add_argument('--learning_rate', type=float, default=0.000646)
parser.add_argument('--min_lr', type=float, default=0.0000646)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)

parser.add_argument('--mup', action='store_true', help='Enable muP', default=False)
parser.add_argument('--mup_multiplier', type=float, default=1)
parser.add_argument('--complete_p_layers', action='store_true', help='Enable complete P layers', default=False)

parser.add_argument('--normalization', type=str, default='RMSNorm')
parser.add_argument('--q_prelayer_normalization', type=str, default='NoNorm', help='Pre-layer normalization for query')
parser.add_argument('--k_prelayer_normalization', type=str, default='NoNorm', help='Pre-layer normalization for key')
parser.add_argument('--impl', type=str, default='tpv_left_impl')

# Optimizer parameters
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.95)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--decay_lr', action='store_true')

# FSDP
parser.add_argument('--enable_fsdp', action='store_true', help='Enable Fully Sharded Data Parallel (FSDP)', default=False)

# -------------------------
# Mixture-of-Experts (MoE) arguments
# -------------------------
# Enable MoE
parser.add_argument('--use_moe', nargs='?', const='true', default=False,
                    type=lambda s: str(s).lower() in ('true','1','yes','y','t'),
                    help='Enable mixture-of-experts')
# Core MoE knobs (used by model config)
parser.add_argument('--num_experts', type=int, default=0, help='number of experts (if 0 or 1 MoE is disabled)')
parser.add_argument('--moe_ffn_hidden_size', type=int, default=128, help='hidden size of each expert')
parser.add_argument('--router_topk', type=int, default=1, help='top-k experts to select')
parser.add_argument('--moe_seq_aux_loss_coeff', type=float, default=0.0, help='coefficient for MoE aux loss')
parser.add_argument('--moe_ffn_mup_multiplier', type=float, default=1.0, help='muP multiplier for MoE ffn hidden size')
parser.add_argument('--moe_null_expert_bias', type=float, default=0.0, help='bias added to null expert logits')
parser.add_argument('--moe_random_router', action='store_true', help='Use randomization in the MoE router', default=False)
# -------------------------

args = parser.parse_args()

file_dir = os.path.dirname(os.path.abspath(__file__))
if args.use_fsdp:
    raise Exception("use FSDP is depreciated, use `enable_fsdp` instead.")
else:
    TRAINING_SCRIPT = os.path.join(file_dir, 'slimpj_train_2.py')

os.makedirs(args.sbatch_logging_dir, exist_ok=True)
os.makedirs(args.out_dir, exist_ok=True)

if args.max_iters is None:
    n_params = gpt_params(
        seq_len=args.block_size,
        vocab_size=50257,
        d_model=args.n_embd,
        num_heads=args.n_head,
        num_layers=args.n_layer
    )
    # The batch size is assumed to be multiplied by gradient accumulation steps
    # but ideally the grad accumulation steps match the gpu numbers...
    args.max_iters = args.tpp * int(n_params / (args.batch_size * 979))

if args.lr_decay_iters is None:
    args.lr_decay_iters = args.max_iters

if args.warmup_iters is None:
    args.warmup_iters = min(int(0.1 * args.max_iters), 1000)


# exclusive flag grants exclusive access to the entire node. However
# requested GPUs are given exclusive access by default.
exclusive = '#SBATCH --exclusive' if args.sbatch_exclusive else ''

if args.n_gpus > 8:
    raise ValueError("n_gpus is a per-node value and cannot exceed 8.")

# Determine the command to run based on n_gpus
if args.n_gpus > 0:
    # For multi-GPU, use torchrun
    # SLURM_JOB_NODELIST and SLURM_JOBID are available in the SLURM job environment
    # We need to determine MASTER_ADDR and MASTER_PORT within the SLURM script
    # SLURM_NNODES and SLURM_NODEID are also provided by SLURM
    dist_args = f"""
# --- PyTorch DDP Specific Environment Variables ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"

NNODES=$SLURM_JOB_NUM_NODES

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export TRITON_CACHE_DIR="/tmp/triton-cache"

echo Node IP: $head_node_ip
echo $SLURM_JOB_NODELIST
export LOGLEVEL=INFO

DISTRIBUTED_ARGS=(
    --nproc_per_node={args.n_gpus} 
    --nnodes=$NNODES
    --rdzv_id $RANDOM-$USER 
    --rdzv_backend c10d 
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
)
"""
else:
    # For single GPU, just run python directly
    dist_args = f""

# Construct the full SLURM batch script
shell_script = f"""#!/bin/bash
#SBATCH --chdir={PROJECT_ROOT}
#SBATCH --time={args.sbatch_timeout}
#SBATCH --nodes={args.sbatch_nodes}
{exclusive}
#SBATCH --gres=gpu:{args.n_gpus}
#SBATCH --ntasks-per-node=1 # Important: srun launches one task (torchrun) per node, or one python script if n_gpus=1
#SBATCH --cpus-per-task={args.cpus_per_task}

#SBATCH --output={args.sbatch_logging_dir}/%j.out
#SBATCH --error={args.sbatch_logging_dir}/%j.err
#SBATCH --mem-per-gpu={args.sbatch_mem}GB
{f'#SBATCH --partition={args.partition}' if args.partition is not None else ''}
{f'#SBATCH --qos={args.qos}' if args.qos is not None else ''}
#SBATCH --distribution=pack

{dist_args}

TRAINING_ARGS=(
    --out_dir="{args.out_dir}"
    --wandb_log={args.log_wandb}
    --wandb_project='{args.wandb_project}'
    --wandb_run_name='{args.wandb_run_name}'
    --eval_interval={args.eval_interval}
    --log_interval={args.log_interval}
    --avg_interval={args.avg_interval}
    --eval_iters={args.eval_iters}
    --eval_only={args.eval_only}
    --init_from='{args.init_from}'
    --dataset='{args.dataset}'
    --gradient_accumulation_steps={args.gradient_accumulation_steps}
    --batch_size={args.batch_size}
    --block_size={args.block_size}
    --n_layer={args.n_layer}
    --n_head={args.n_head}
    --n_kv_head={args.n_kv_head}
    --n_embd={args.n_embd}
    --dropout={args.dropout}
    --bias={args.bias}
    --init_std={args.init_std}
    --learning_rate={args.learning_rate}
    --min_lr={args.min_lr}
    --max_iters={args.max_iters}
    --anneal_wd={args.anneal_wd}
    --min_wd={args.min_wd}
    --lr_decay_iters={args.lr_decay_iters}
    --warmup_iters={args.warmup_iters}
    --weight_decay={args.weight_decay}
    --beta1={args.beta1}
    --beta2={args.beta2}
    --grad_clip={args.grad_clip}
    --decay_lr={args.decay_lr}
    --complete_p_layers={args.complete_p_layers}
    --mup={args.mup}
    --mup_multiplier={args.mup_multiplier}
    --seed={args.seed}
    --backend='{args.backend}'
    --device='{args.device}'
    --dtype='{args.dtype}'
    --compile={args.compile}
    --enable_fsdp={args.enable_fsdp}
    --coord_check={args.coord_check}
    --normalization='{args.normalization}'
    --q_prelayer_normalization='{args.q_prelayer_normalization}'
    --k_prelayer_normalization='{args.k_prelayer_normalization}'
    --impl='{args.impl}'
    --decay_profile='{args.decay_profile}'
    --cooldown_iters={args.cooldown_iters}
    --slurm_job_id=$SLURM_JOB_ID
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID
    --wd_warmup_iters={args.wd_warmup_iters}
    --wd_anneal_iters={args.wd_anneal_iters}
    --adaptive_optimizer={args.adaptive_optimizer}
    --use_moe={args.use_moe}
    --num_experts={args.num_experts}
    --moe_ffn_hidden_size={args.moe_ffn_hidden_size}
    --router_topk={args.router_topk}
    --moe_seq_aux_loss_coeff={args.moe_seq_aux_loss_coeff}
    --moe_ffn_mup_multiplier={args.moe_ffn_mup_multiplier}
    --moe_null_expert_bias={args.moe_null_expert_bias}
    --moe_random_router={args.moe_random_router}
)

eval "$(conda shell.bash hook)"
conda activate nanogpt

srun --export=ALL,MASTER_ADDR,MASTER_PORT,WORKDIR,requirements \\
    torchrun "${{DISTRIBUTED_ARGS[@]}}" {TRAINING_SCRIPT} "${{TRAINING_ARGS[@]}}" &
    
SRUN_PID=$!
wait $SRUN_PID
"""

try:    
    with open(os.path.join(args.sbatch_logging_dir, f"sbatch_command.sh"), 'w') as f:
        f.write(shell_script)
        script_path = os.path.join(args.sbatch_logging_dir, f"sbatch_command.sh")
    # Use --wait so sbatch will not return until the submitted job completes.
    # This makes the Python submitter block, so the SLURM array task will
    # also wait and thus the array concurrency limit controls concurrent trainings.
    process = subprocess.run(['sbatch', '--wait', script_path], text=True, capture_output=True, check=True)
    print(f"Job submitted successfully: {process.stdout.strip()}", flush=True)
except subprocess.CalledProcessError as e:
    print("Return code:", e.returncode)
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
    raise