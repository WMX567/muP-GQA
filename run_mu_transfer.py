"""Audit, generate or run a reproducible width-transfer sweep without importing torch."""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time

from mu_transfer import PROTOCOL_VERSION, get_args, normalized_config
from mu_transfer_io import has_metrics, load_metrics, load_legacy

ROOT = Path(__file__).resolve().parent
WIDTH_STEPS = {512: 1160, 768: 1715, 1024: 2356, 2048: 4754}
LEARNING_RATES = [0.01563, 0.00781, 0.00391, 0.00195, 0.00098, 0.00049]


def training_args(config, out_dir):
    args = ['--out_dir', str(out_dir)]
    for key, value in config.items():
        if value is None:
            continue
        args.extend(['--' + key, str(value)])
    return args


def tasks(args):
    result = []
    for width, lr, seed, param_type, wd_mode in itertools.product(
            args.widths, args.learning_rates, args.seeds, args.param_types, args.wd_modes):
        steps = args.max_iters or WIDTH_STEPS[width]
        # Preserve the existing integrated-decay sweep; zero-WD is the cleaner control.
        wd = round(1 / steps / lr / 0.035, 5) if wd_mode == 'scaled' else 0.0
        config = dict(n_embd=width, n_head=width // 128, n_kv_head=2, n_layer=3,
                      learning_rate=lr, weight_decay=wd, seed=seed, param_type=param_type,
                      impl=args.impl, base_width=512.0, batch_size=args.batch_size,
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      max_iters=steps, block_size=args.block_size, dataset=args.dataset,
                      data_dir=str(args.data_dir.resolve()) if args.data_dir else None,
                      dtype=args.dtype, device=args.device, eval_iters=args.eval_iters,
                      eval_interval=args.eval_interval, checkpoint_interval=args.checkpoint_interval)
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        name = f'{param_type}_{wd_mode}_w{width}_lr{lr:g}_s{seed}_{signature}'
        result.append(dict(name=name, wd_mode=wd_mode, config=config, out_dir=str(args.results_dir.resolve()/name)))
    return result


def status(task):
    out = Path(task['out_dir'])
    if not out.exists():
        return 'missing'
    try:
        expected = normalized_config(get_args(training_args(task['config'], out)))
        actual = json.loads((out/'config.json').read_text())
        if expected != actual:
            return 'config_mismatch'
        marker = json.loads((out/'completed.json').read_text())
        rows = load_metrics(out/'metrics.npy')
        last = rows[-1]
        if (marker['protocol_version'] == PROTOCOL_VERSION and marker['step'] == expected['max_iters']
                and int(last['step']) == expected['max_iters'] and math.isfinite(float(last['val_loss']))
                and math.isfinite(float(last['train_loss'])) and math.isfinite(float(marker['val_loss']))
                and float(last['val_loss']) == float(marker['val_loss']) and (out/'checkpoint.pt').exists()):
            return 'complete'
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass
    return 'incomplete'


def legacy_audit(directory):
    rows = []
    for width, lr, seed in itertools.product(WIDTH_STEPS, LEARNING_RATES, [0, 1, 2]):
        steps = WIDTH_STEPS[width]
        wd = 1 / steps / lr / 0.035
        path = directory/f'width{width}_lr{lr:.5f}_wd{wd:.5f}_seed{seed}.npy'
        if not path.exists() and path.with_suffix('.csv').exists():
            path = path.with_suffix('.csv')
        state = 'missing'
        if path.exists():
            try:
                values = load_legacy(path)
                expected_rows = steps // 10
                state = 'legacy_complete' if len(values) == expected_rows and all(math.isfinite(float(x['loss'])) for x in values) else 'legacy_incomplete'
            except (OSError, ValueError, KeyError, EOFError):
                state = 'legacy_invalid'
        rows.append(dict(width=width, lr=lr, seed=seed, status=state, path=str(path)))
    return rows


def execute(task):
    out = Path(task['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"Already running: {task['name']}")
            return
        state = status(task)
        if state == 'complete':
            print(f"Skip completed: {task['name']}")
            return
        if state == 'config_mismatch':
            raise ValueError(f'{out}: configuration mismatch')
        command = [sys.executable, str(ROOT/'mu_transfer.py'), *training_args(task['config'], out)]
        if (out/'checkpoint.pt').exists():
            command.append('--resume')
        elif has_metrics(out/'metrics.npy'):
            # Preserve partial logs from failures before the first checkpoint.
            for suffix in ('.npy', '.csv'):
                path = out/f'metrics{suffix}'
                if path.exists():
                    path.rename(out/f'metrics_uncheckpointed_{time.time_ns()}{suffix}')
        print(shlex.join(command), flush=True)
        with (out/'run.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if status(task) != 'complete':
            raise RuntimeError(f'{out}: training exited without a valid completion marker')


def generate(args, pending):
    destination = args.scripts_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Immutable manifests keep a queued array independent of subsequent audits.
    data = json.dumps(pending, indent=2, sort_keys=True)+'\n'
    digest = hashlib.sha256(data.encode()).hexdigest()[:12]
    manifest = destination/f'tasks_{digest}.json'
    manifest.write_text(data)
    if not pending:
        print('No pending experiments; no array generated.')
        return None
    logs = destination/'logs'
    logs.mkdir(exist_ok=True)
    script = destination/f'submit_{digest}.sh'
    activate = ''
    if args.conda_env:
        activate = 'eval "$(conda shell.bash hook)"\nconda activate '+shlex.quote(args.conda_env)+'\n'
    command = ['python' if args.conda_env else sys.executable, str(ROOT/'run_mu_transfer.py'),
               '--mode', 'task', '--manifest', str(manifest)]
    script.write_text(f'''#!/bin/bash
#SBATCH --array=0-{len(pending)-1}%{args.max_concurrent}
#SBATCH --partition={args.partition}
#SBATCH --time={args.time_limit}
#SBATCH --gres=gpu:1
#SBATCH --mem={args.memory_gb}G
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --output={logs}/%A_%a.out
#SBATCH --error={logs}/%A_%a.err
#SBATCH --chdir={ROOT}
set -euo pipefail
{activate}{shlex.join(command)} --task-index "$SLURM_ARRAY_TASK_ID"
''')
    print(f'Generated {len(pending)} tasks: {script}')
    return script


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['report', 'generate', 'local', 'submit', 'task'], default='report')
    p.add_argument('--results-dir', type=Path, default=ROOT/'mu_transfer_results'/'v2')
    p.add_argument('--scripts-dir', type=Path, default=ROOT/'mu_transfer_jobs')
    p.add_argument('--legacy-dir', type=Path, default=ROOT/'mup')
    p.add_argument('--widths', nargs='+', type=int, choices=list(WIDTH_STEPS), default=list(WIDTH_STEPS))
    p.add_argument('--learning-rates', nargs='+', type=float, default=LEARNING_RATES)
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--param-types', nargs='+', choices=['mup', 'sp'], default=['mup'])
    p.add_argument('--wd-modes', nargs='+', choices=['scaled', 'zero'], default=['zero'])
    p.add_argument('--impl', choices=['mengxi_impl', 'standard_param_impl', 'kyle_impl'], default='mengxi_impl')
    p.add_argument('--dataset', default='openwebtext')
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    p.add_argument('--max-iters', type=int)
    for name, default in [('batch-size', 6), ('gradient-accumulation-steps', 6), ('block-size', 1024),
                          ('eval-iters', 20), ('eval-interval', 100), ('checkpoint-interval', 100),
                          ('max-concurrent', 4), ('memory-gb', 64), ('cpus-per-task', 4)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--partition', default='gpu')
    p.add_argument('--time-limit', default='12:00:00')
    p.add_argument('--conda-env')
    p.add_argument('--limit', type=int, help='Run/generate only the first N pending experiments')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--task-index', type=int)
    args = p.parse_args()
    if args.mode == 'task':
        if args.manifest is None or args.task_index is None:
            p.error('task mode requires --manifest and --task-index')
        selected = json.loads(args.manifest.read_text())
        if not 0 <= args.task_index < len(selected):
            p.error('task-index is outside the manifest')
        execute(selected[args.task_index])
        return
    if args.limit is not None and args.limit <= 0:
        p.error('limit must be positive')
    if args.max_concurrent <= 0:
        p.error('max-concurrent must be positive')
    all_tasks = tasks(args)
    for task in all_tasks:
        # Validate training options even in a audit without PyTorch.
        get_args(training_args(task['config'], task['out_dir']))
        task['status'] = status(task)
    pending = [t for t in all_tasks if t['status'] != 'complete']
    legacy = legacy_audit(args.legacy_dir)
    legacy_good = sum(t['status'] == 'legacy_complete' for t in legacy)
    print(f'Legacy grid: {legacy_good}/{len(legacy)} logs complete under the old protocol.')
    print(f'Protocol v{PROTOCOL_VERSION}: {len(all_tasks)-len(pending)}/{len(all_tasks)} complete; {len(pending)} pending.')
    args.scripts_dir.mkdir(parents=True, exist_ok=True)
    (args.scripts_dir/'audit.json').write_text(json.dumps({'protocol_version': PROTOCOL_VERSION, 'legacy': legacy, 'tasks': all_tasks},indent=2)+'\n')
    if args.mode == 'report':
        for param, wd in itertools.product(args.param_types, args.wd_modes):
            count = sum(t['config']['param_type']==param and t['wd_mode']==wd for t in pending)
            print(f'  {param}/{wd}: {count} pending')
        return
    if args.limit:
        pending = pending[:args.limit]
    if args.mode == 'local':
        for task in pending:
            execute(task)
    else:
        script = generate(args, pending)
        if args.mode == 'submit' and script:
            subprocess.run(['sbatch', str(script)], check=True)


if __name__ == '__main__':
    main()
