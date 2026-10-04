"""Summarize completed protocol-v2 runs and plot validation-loss transfer curves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from run_mu_transfer import status
from mu_transfer_io import load_metrics, save_array
import numpy as np

ROOT = Path(__file__).resolve().parent


def summarize(results_dir):
    groups = defaultdict(list)
    for config_path in sorted(results_dir.glob('*/config.json')):
        out = config_path.parent
        config = json.loads(config_path.read_text())
        # Reconstruct a task with the configuration recorded by training.
        flags = {k: v for k, v in config.items() if k != 'protocol_version'}
        if status({'out_dir': str(out), 'config': flags}) != 'complete':
            continue
        last = load_metrics(out/'metrics.npy')[-1]
        wd_mode = 'zero' if config['weight_decay'] == 0 else 'scaled'
        # Separate different protocols/budgets rather than averaging incompatible runs.
        common = {k: v for k, v in config.items() if k not in
                  ('n_embd', 'n_head', 'n_kv_head', 'seed', 'learning_rate', 'weight_decay', 'param_type', 'max_iters')}
        key = (json.dumps(common, sort_keys=True), config['param_type'], wd_mode, config['n_embd'],
               config['learning_rate'], config['weight_decay'], config['max_iters'], config['n_head'], config['n_kv_head'])
        groups[key].append((config['seed'], float(last['val_loss'])))
    rows = []
    for key, values in sorted(groups.items()):
        seeds = [seed for seed, _ in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f'Duplicate results for seeds {seeds}; use a results directory containing a single sweep.')
        losses = [loss for _, loss in values]
        protocol, param, wd_mode, width, lr, wd, steps, heads, kv_heads = key
        rows.append(dict(protocol=protocol, param_type=param, wd_mode=wd_mode, width=width, lr=lr, wd=wd,
                         max_iters=steps, n_head=heads, n_kv_head=kv_heads, n_seeds=len(values), seeds=','.join(map(str, sorted(seeds))),
                         val_loss=statistics.mean(losses), val_std=statistics.stdev(losses) if len(losses)>1 else 0.0))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir', type=Path, default=ROOT/'mu_transfer_results'/'v2')
    p.add_argument('--output', type=Path, default=ROOT/'mu_transfer_results'/'mu_transfer_v2.png')
    p.add_argument('--summary-only', action='store_true', help='Write NPY without requiring matplotlib')
    args = p.parse_args()
    rows = summarize(args.results_dir)
    if not rows:
        p.error('No completed v2 runs found; legacy CSVs are not mixed with the corrected protocol.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = args.output.with_suffix('.npy')
    integer_fields = {'width', 'max_iters', 'n_head', 'n_kv_head', 'n_seeds'}
    float_fields = {'lr', 'wd', 'val_loss', 'val_std'}
    dtype = [(name, '<i8' if name in integer_fields else '<f8' if name in float_fields
              else f'U{max(len(str(row[name])) for row in rows)}') for name in rows[0]]
    array = np.array([tuple(row.values()) for row in rows], dtype=dtype)
    save_array(summary, array)
    print(f'Summary: {summary}')
    if args.summary_only:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    protocols = sorted({r['protocol'] for r in rows})
    if len(protocols) != 1:
        p.error('Mixed sweep settings; plot each sweep in a separate results directory. NPY preserves the groups.')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for i, wd in enumerate(['zero', 'scaled']):
        for j, param in enumerate(['mup', 'sp']):
            ax = axes[i][j]
            selected = [r for r in rows if r['param_type']==param and r['wd_mode']==wd]
            for width in sorted({r['width'] for r in selected}):
                points = sorted([r for r in selected if r['width']==width], key=lambda r:r['lr'])
                if len({r['lr'] for r in points}) != len(points):
                    p.error('Mixed architecture or budgets; plot separate sweeps.')
                ax.errorbar([r['lr'] for r in points], [r['val_loss'] for r in points],
                            yerr=[r['val_std'] for r in points], marker='o', capsize=3,
                            label=f"width {width} (seeds {min(r['n_seeds'] for r in points)}–{max(r['n_seeds'] for r in points)})")
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Base learning rate')
            ax.set_ylabel('Final validation loss')
            ax.set_title(f'{param.upper()} / {wd} weight decay')
            ax.grid(alpha=0.3)
            if selected:
                ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f'Plot: {args.output}')


if __name__ == '__main__':
    main()
