# GQA-μP

GQA-μP with Mengxi μP learning-rate transfer experiments with grouped-query attention (GQA). The main entry point is [run_mu_transfer.py](run_mu_transfer.py).

## Setup

```sh
pip install torch numpy matplotlib datasets tiktoken tqdm transformers wandb
```

Training requires PyTorch with `torch.amp.GradScaler` and scaled dot-product attention support. Matplotlib is used for plotting; WandB is optional for the main sweep. Run all commands from the repository root.

## Dataset preparation

Data and preparation scripts are not bundled. Obtain the OpenWebText preparation script from upstream nanoGPT:

```sh
mkdir -p data/openwebtext
curl -fsSL https://raw.githubusercontent.com/karpathy/nanoGPT/master/data/openwebtext/prepare.py -o data/openwebtext/prepare.py
python data/openwebtext/prepare.py
```

Training expects `data/openwebtext/train.bin` and `val.bin` as raw uint16 token arrays. Use `--data-dir /path/to/tokenized/openwebtext` for an external dataset.

## Mengxi μP sweep

| Setting | Default |
| --- | --- |
| Implementation | `mengxi_impl` in [mup_implementations.py](mup_implementations.py) |
| Width / attention heads | 512/4, 768/6, 1024/8, 2048/16 |
| KV heads / depth / head size | 2 / 3 / 128 |
| Base learning rates | 0.01563, 0.00781, 0.00391, 0.00195, 0.00098, 0.00049 |
| Seeds / weight decay | 0, 1, 2 / zero |
| Optimizer updates by width | 1160, 1715, 2356, 4754 |
| Effective batch / context | 6 microbatch × 6 accumulation = 36 sequences / 1024 tokens |
| Validation | Every 100 updates and at completion; 20 batches |

The default grid contains 72 runs. Width-specific update counts use different token budgets; set `--max-iters N` for a fixed token budget. KV heads stay fixed, so the GQA ratio also changes with width. This sweep varies both width and GQA; MoE and depth sweeps are separate.

Start with a small GPU smoke test:

```sh
python run_mu_transfer.py --mode local --widths 512 --learning-rates 0.00195 --seeds 0 --max-iters 2 --block-size 64 --batch-size 1 --gradient-accumulation-steps 2 --eval-iters 2 --results-dir mu_transfer_results/smoke
```

Audit runs, generate pending jobs, then submit a SLURM array:

```sh
python run_mu_transfer.py --mode report
python run_mu_transfer.py --mode generate --max-concurrent 4 --conda-env nanogpt
python run_mu_transfer.py --mode submit --max-concurrent 4 --conda-env nanogpt --partition gpu
```

Generate jobs on the cluster, since scripts reference that checkout and environment. Without `--conda-env`, jobs use the Python executable that generated the manifest. Use `--limit N` to restrict pending runs and `--memory-gb`, `--time-limit`, or `--cpus-per-task` to adjust resources. Add `--wd-modes zero scaled` for the decay control, or `--param-types mup sp` for SP controls.

Repeating a command skips completed runs and resumes incomplete runs from their last checkpoint, restoring optimizer, scaler, and training RNG states. Each configuration has its own metrics, checkpoint, completion marker, and log. Audit details are saved in `mu_transfer_jobs/audit.json`.

## Results

```sh
python mu_transfer_plot.py
python mu_transfer_plot.py --summary-only
```

Plots summarize final validation loss across seeds, with sample standard deviation as error bars. Partial runs are excluded. Per-run `mu_transfer_results/v2/<run>/metrics.npy` contains the named fields `step`, `train_loss`, `val_loss`, `lr`, `tokens`, and `seconds`; load it with `np.load(path, allow_pickle=False)`. Unmeasured validation loss is `NaN`. The aggregate summary is `mu_transfer_results/mu_transfer_v2.npy`.

In protocol v2, `max_iters` counts optimizer updates. Data, results, and checkpoints are not stored in the repository.

## References

- **Kyle's μP implementation:** `kyle_impl` in [mup_implementations.py](mup_implementations.py), named “xLLM (muP) Kyle Candidate KV Scaling.” Retained as a comparison implementation; select it with `--impl kyle_impl`.
- **Upstream:** [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).
