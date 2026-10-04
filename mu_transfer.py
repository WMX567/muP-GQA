"""Single-GPU muTransfer training; max_iters counts optimizer updates."""
import argparse
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
from contextlib import nullcontext

PROTOCOL_VERSION = 2


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false')


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir', default='mu_transfer_results/manual')
    p.add_argument('--dataset', default='openwebtext')
    p.add_argument('--data_dir', type=Path)
    for name, default in [('n_layer', 3), ('n_head', 4), ('n_kv_head', 2), ('n_embd', 512),
                          ('block_size', 1024), ('batch_size', 6), ('max_iters', 1160),
                          ('gradient_accumulation_steps', 6), ('seed', 0),
                          ('eval_interval', 100), ('eval_iters', 20), ('log_interval', 10),
                          ('checkpoint_interval', 100)]:
        p.add_argument('--' + name, type=int, default=default)
    for name, default in [('dropout', 0.0), ('init_std', 0.02), ('learning_rate', 0.00195),
                          ('weight_decay', 0.0), ('beta1', 0.9), ('beta2', 0.95),
                          ('eps', 1e-8), ('grad_clip', 1.0), ('base_width', 512.0)]:
        p.add_argument('--' + name, type=float, default=default)
    for name in ('bias', 'decay_lr', 'compile'):
        p.add_argument('--' + name, type=boolean, nargs='?', const=True, default=False)
    p.add_argument('--param_type', choices=['mup', 'sp'], default='mup')
    p.add_argument('--impl', choices=['mengxi_impl', 'standard_param_impl', 'kyle_impl'], default='mengxi_impl')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(argv)
    for name in ('n_layer', 'n_head', 'n_kv_head', 'n_embd', 'block_size', 'batch_size',
                 'max_iters', 'gradient_accumulation_steps', 'eval_interval', 'eval_iters',
                 'log_interval', 'checkpoint_interval', 'base_width', 'learning_rate'):
        if getattr(args, name) <= 0:
            p.error(f'{name} must be positive')
    if args.n_embd % args.n_head or args.n_head % args.n_kv_head:
        p.error('n_embd must divide by n_head, and n_head by n_kv_head')
    if args.bias:
        p.error('This experiment uses bias-free models for both SP and muP')
    return args


def normalized_config(args):
    data_dir = (args.data_dir or Path(__file__).resolve().parent / 'data' / args.dataset).resolve()
    config = {**vars(args), 'data_dir': str(data_dir), 'protocol_version': PROTOCOL_VERSION}
    for key in ('resume', 'out_dir', 'device'):
        config.pop(key)
    return config


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def main():
    args = get_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Use one GPU per experiment; the sweep runs independent SLURM jobs.')
    import numpy as np
    from mu_transfer_io import has_metrics, load_metrics, metrics_array, save_array
    import torch
    from model_moe_kyle import GPT, GPTConfig
    from mup_implementations import impl_dict

    data_dir = (args.data_dir or Path(__file__).resolve().parent / 'data' / args.dataset).resolve()
    data = {}
    for split in ('train', 'val'):
        path = data_dir / f'{split}.bin'
        if not path.exists():
            raise FileNotFoundError(f'{path}: prepare the dataset first (README.md).')
        data[split] = np.memmap(path, dtype=np.uint16, mode='r')
        if len(data[split]) <= args.block_size:
            raise ValueError(f'{path} must contain more than {args.block_size} tokens')
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = normalized_config(args)
    config_path = out / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Output directory belongs to a different configuration; use a new out_dir.')
    if has_metrics(out / 'metrics.npy') and not args.resume:
        raise ValueError('Results already exist; use --resume or a new out_dir.')
    write_json(config_path, config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    def autocast():
        return torch.amp.autocast(device.type, dtype=dtype) if device.type == 'cuda' and args.dtype != 'float32' else nullcontext()
    # Evaluation uses a private generator, reset each time, and cannot change training batches.
    train_rng = torch.Generator().manual_seed(args.seed)
    def batch(split, generator):
        indices = torch.randint(len(data[split]) - args.block_size, (args.batch_size,), generator=generator)
        x = torch.from_numpy(np.stack([data[split][i:i+args.block_size].astype(np.int64) for i in indices.tolist()]))
        y = torch.from_numpy(np.stack([data[split][i+1:i+args.block_size+1].astype(np.int64) for i in indices.tolist()]))
        return x.to(device), y.to(device)
    vocab_size = 50304
    if (data_dir / 'meta.pkl').exists():
        with (data_dir / 'meta.pkl').open('rb') as f:
            vocab_size = pickle.load(f)['vocab_size']
    impl_name = args.impl if args.param_type == 'mup' else 'standard_param_impl'
    if impl_name not in impl_dict:
        raise ValueError(f'Unknown impl {impl_name}; available: {list(impl_dict)}')
    model = GPT(GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
                          n_embd=args.n_embd, block_size=args.block_size, vocab_size=vocab_size,
                          dropout=args.dropout, bias=False, init_std=args.init_std,
                          mup=args.param_type == 'mup', mup_multiplier=args.n_embd / args.base_width if args.param_type == 'mup' else 1.0,
                          impl=impl_dict[impl_name], normalization='LayerNorm')).to(device)
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate,
                                           (args.beta1, args.beta2), args.eps, device.type)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda' and args.dtype == 'float16')
    start = 0
    checkpoint = out / 'checkpoint.pt'
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state['config'] != config:
            raise ValueError('Checkpoint configuration does not match')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        start = state['step']
        train_rng.set_state(state['train_rng'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda_rng']])
    elif args.resume and has_metrics(out / 'metrics.npy'):
        raise ValueError('Cannot resume metrics without a checkpoint')
    training_model = torch.compile(model) if args.compile else model
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(args.seed + 100000)
        total = 0.0
        with torch.no_grad():
            for _ in range(args.eval_iters):
                with autocast():
                    total += model(*batch('val', generator))[1].item()
        model.train()
        return total / args.eval_iters
    def save(step):
        temp = checkpoint.with_suffix('.tmp')
        torch.save({'config': config, 'step': step, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                    'train_rng': train_rng.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}, temp)
        temp.replace(checkpoint)
    metrics_path = out / 'metrics.npy'
    rows = []
    if has_metrics(metrics_path):
        saved = load_metrics(metrics_path)
        rows = [{name: row[name].item() for name in saved.dtype.names}
                for row in saved if row['step'] <= start]
    save_array(metrics_path, metrics_array(rows))
    model.train()
    for step in range(start + 1, args.max_iters + 1):
        t0 = time.monotonic()
        lr = args.learning_rate * (0.1 + 0.9 * (1 + math.cos(math.pi * (step-1) / args.max_iters)) / 2) if args.decay_lr else args.learning_rate
        for group in optimizer.param_groups:
            group['lr'] = lr * group['lr_scale']
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        for _ in range(args.gradient_accumulation_steps):
            with autocast():
                loss = training_model(*batch('train', train_rng))[1]
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Nonfinite loss at optimizer step {step}')
            loss_total += loss.detach().item() / args.gradient_accumulation_steps
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
        scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        val = evaluate() if step % args.eval_interval == 0 or step == args.max_iters else None
        if val is not None and not math.isfinite(val):
            raise FloatingPointError(f'Nonfinite validation loss at step {step}')
        if step % args.log_interval == 0 or val is not None:
            rows.append(dict(step=step, train_loss=loss_total, val_loss=val if val is not None else np.nan, lr=lr,
                                 tokens=step * args.batch_size * args.gradient_accumulation_steps * args.block_size,
                                 seconds=time.monotonic()-t0))
            save_array(metrics_path, metrics_array(rows))
            print(f'Update {step}/{args.max_iters}: train={loss_total:.4f}, val={val}', flush=True)
        if step % args.checkpoint_interval == 0 or step == args.max_iters:
            save(step)
    # A completion marker is written only after the final evaluation and checkpoint.
    if start >= args.max_iters:
        last = load_metrics(metrics_path)[-1]
        val = float(last['val_loss'])
    write_json(out / 'completed.json', {'protocol_version': PROTOCOL_VERSION, 'step': args.max_iters, 'val_loss': val})


if __name__ == '__main__':
    main()
