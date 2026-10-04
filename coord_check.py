"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import numpy as np
import pickle
import random
import torch
import wandb

from contextlib import nullcontext
from megatron.core.datasets.indexed_dataset import IndexedDataset

from model_moe_kyle import GPTConfig, GPT
from mup_implementations import (
    impl_dict
)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_kv_head = n_head
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
eps = 1e-8 
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 0 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'float32'
compile = True # use PyTorch 2.0 to compile the model to be faster
mup = True
mup_multiplier = 1.0
coord_check = True
# impl = tpv_left_impl
impl = 'mengxi_impl'
seed = 42
bias = False
init_std = 0.02
coord_check = True
normalization = "LayerNorm"
tag = "default"  # Tag for coordinate checking data
use_moe: bool = False
num_experts: int = 0
router_topk: int = 8
moe_ffn_hidden_size: int = 128
moe_seq_aux_loss_coeff: float = 1e-4
moe_ffn_mup_multiplier: float = 1.0
moe_random_router: bool = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
seed_everything(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ptimpl = impl_dict[impl] if isinstance(impl, str) else impl
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

if dataset != 'slim_pajama':
    # poor man's data loader
    data_dir = os.path.join(os.path.dirname(__file__), 'data', dataset)
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
else:
    path_prefix = "/mnt/sharefs/data/pretrain_tokenized/SlimPajama-627B_250k_tokenized/merged/slimpajama-train-chunk1"
    slimpj_dataset = IndexedDataset(path_prefix)

    data_buffer = []
    dataset_idx = 0

    def get_batch(_):
        global dataset_idx, data_buffer

        X_batch = []
        Y_batch = []

        while len(X_batch) < batch_size:
            while len(data_buffer) < (2 * block_size):
                new_sample = slimpj_dataset.get(dataset_idx)
                dataset_idx += 1

                if new_sample is None:
                    if not X_batch:
                        raise StopIteration("End of dataset reached and no full batches could be formed.")
                    else:
                        break
                
                data_buffer.extend(new_sample)

            while len(data_buffer) >= (block_size + 1) and len(X_batch) < batch_size:
                x = data_buffer[:block_size]
                y = data_buffer[1:block_size+1]

                X_batch.append(x)
                Y_batch.append(y)

                data_buffer = data_buffer[1:]

        x = torch.tensor(X_batch, dtype=torch.long)
        y = torch.tensor(Y_batch, dtype=torch.long)

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
if dataset != 'slim_pajama':
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
else:
    meta_vocab_size = 250221

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, mup=mup, 
                  mup_multiplier=mup_multiplier, init_std=init_std,
                  impl=ptimpl, normalization=normalization, use_moe=use_moe, 
                  num_experts=num_experts, router_topk=router_topk, 
                  moe_ffn_hidden_size=moe_ffn_hidden_size, 
                  moe_seq_aux_loss_coeff=moe_seq_aux_loss_coeff, 
                  moe_ffn_mup_multiplier=moe_ffn_mup_multiplier,
                  moe_random_router=moe_random_router) # start with model_args from command line

# init a new model from scratch
print("Initializing a new model from scratch")
# determine the vocab size we'll use for from-scratch training
if meta_vocab_size is None:
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
print("MODEL ARGS:", model_args)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

if coord_check:
    from coordinate_checking import get_hooks
    data = {}
    hooks = []

    m = model.config.mup_multiplier
    n = model.config.n_embd // model.config.n_head
    r = model.config.n_head // model.config.n_kv_head

    for i, layer in enumerate(model.transformer.h):
        if not use_moe:
            mlp = layer.ffn

            fc = mlp.c_fc
            forward_hook = get_hooks(data, f"fc.{i}")
            hook = fc.register_forward_hook(forward_hook)
            hooks.append(hook)

            c_proj = mlp.c_proj
            forward_hook = get_hooks(data, f"c_proj.{i}")
            hook = c_proj.register_forward_hook(forward_hook)
            hooks.append(hook)
        else:
            mlp = layer.ffn

            if hasattr(mlp, 'w_gating'):
                fc = mlp.w_gating
                forward_hook = get_hooks(data, f"w_gating.{i}")
                hook = fc.register_forward_hook(forward_hook)
                hooks.append(hook)
            else: # MLP layer
                fc = mlp.c_fc
                forward_hook = get_hooks(data, f"fc.{i}")
                hook = fc.register_forward_hook(forward_hook)
                hooks.append(hook)

                c_proj = mlp.c_proj
                forward_hook = get_hooks(data, f"c_proj.{i}")
                hook = c_proj.register_forward_hook(forward_hook)
                hooks.append(hook)

        attn = layer.attn
        # q: sn_key='q', m, n
        forward_hook = get_hooks(data, f"attn.c_q.{i}", sn_key='q', m=m, n=n)
        hook = attn.c_q.register_forward_hook(forward_hook)
        hooks.append(hook)

        # k: sn_key='k', m, n, r
        forward_hook = get_hooks(data, f"attn.c_kv.{i}", sn_key='k', m=m, n=n, r=r)
        hook = attn.c_kv.register_forward_hook(forward_hook)
        hooks.append(hook)

        forward_hook = get_hooks(data, f"attn.c_proj.{i}")
        hook = attn.c_proj.register_forward_hook(forward_hook)
        hooks.append(hook)

        if hasattr(attn, 'kv_capture'):
            forward_hook = get_hooks(data, f"attn.kv_capture.{i}")
            hook = attn.kv_capture.register_forward_hook(forward_hook)
            hooks.append(hook)
        else:
            raise ValueError(f"attn.kv_capture not found in layer {i}, please check the implementation")

    forward_hook = get_hooks(data, f"lm_head")
    hook = model.lm_head.register_forward_hook(forward_hook)
    hooks.append(hook)
    
    # forward_hook = get_hooks(data, "output")
    # hook = model.output.register_forward_hook(forward_hook)
    # hooks.append(hook)
        
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), eps, device_type)
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                if use_moe:
                    logits, loss, _ = model(X, Y)
                else:
                    logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model
running_mfu = -1.0
while True:
    # if iter_num % eval_interval == 0:
    #     losses = estimate_loss()
    #     print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    #     if wandb_log:
    #         wandb.log({
    #             "iter": iter_num,
    #             "train/loss": losses['train'],
    #             "val/loss": losses['val'],
    #             "mfu": running_mfu*100, # convert to percentage
    #         })
        # if losses['val'] < best_val_loss or always_save_checkpoint:
        #     best_val_loss = losses['val']
        #     if iter_num > 0:
        #         checkpoint = {
        #             'model': raw_model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'model_args': model_args,
        #             'iter_num': iter_num,
        #             'best_val_loss': best_val_loss,
        #             'config': config,
        #         }
        #         print(f"saving checkpoint to {out_dir}")
        #         torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    
    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            if use_moe:
                logits, loss, _ = model(X, Y)
            else:
                logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    abs_grad_norm = ( model.transformer.h[-1].attn.c_q.weight.grad.abs().mean().item() +
                      model.transformer.h[-1].attn.c_kv.weight.grad.abs().mean().item() ) / 2

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": loss.item(),
                    "mfu": running_mfu * 100, # convert to percentage
                    "abs_grad_norm": abs_grad_norm,
                })
    iter_num += 1
    local_iter_num += 1

    X, Y = get_batch('train') 

    # termination conditions
    if iter_num > max_iters:
        break

if coord_check:
    import datetime
    from coordinate_checking import dataframe_from_data
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    df = dataframe_from_data(data, width=config['n_embd'], depth=config['n_layer'], seed=seed, tag=config['tag'])
    df.to_csv(os.path.join(out_dir, f'coord_check_{now}.csv'), index=False)
    print(f"Attempted to save coordinate checking data to {os.path.join(out_dir, f'coord_check_{now}.csv')}")
