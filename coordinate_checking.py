
"""
    Utilities for running and plotting coordinate checks.
"""
import math
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from pathlib import Path

# def spectral_norm_svd(A: torch.Tensor) -> float:
#     # return torch.linalg.matrix_norm(A, ord=2)
#     return torch.linalg.svdvals(A.to(torch.float32)).max()

# def natural_spectral_norm(A: torch.Tensor, return_list=False, transpose_weights=False, **kwargs) -> float | torch.Tensor:
#     # return torch.linalg.matrix_norm(A, ord='fro')
#     if len(A.shape) == 3:
#         norms = []
#         for matrix in A:
#             if transpose_weights:
#                 scale_factor = math.sqrt(matrix.size(0) / matrix.size(1))
#             else:
#                 scale_factor = math.sqrt(matrix.size(1) / matrix.size(0)) 
#             norms.append(spectral_norm_svd(matrix) * scale_factor)
#         if return_list:
#             return norms
#         else:
#             return torch.tensor(norms).mean()
#     if transpose_weights:
#         return spectral_norm_svd(A) * math.sqrt(A.size(0) / A.size(1))
#     else:
#         return spectral_norm_svd(A) * math.sqrt(A.size(1) / A.size(0))


spectral_norm_dict = {'q': lambda m, n: 1 / math.sqrt(m*n), 
                      'k': lambda m, n, r: math.sqrt(r / (m*n)), 
                      'v': lambda m, r: math.sqrt(r / m)}

def spectral_norm_svd(A: torch.Tensor) -> float:
    return torch.linalg.svdvals(A.to(torch.float32)).max()

def scale_factor_compute(A: torch.Tensor, key: str, transpose_weights: bool, **kwargs) -> float:
    if key == 'q':
        if not all(k in kwargs for k in ('m', 'n')):
            raise ValueError("spectral_norm_dict['q'] requires m and n (mup_multiplier, n_embd // n_head)")
        scale_factor = 1/spectral_norm_dict['q'](kwargs['m'], kwargs['n'])
    elif key == 'k':
        if not all(k in kwargs for k in ('m', 'n', 'r')):
            raise ValueError("spectral_norm_dict['k'] requires m, n, r (mup_multiplier, n_embd // n_head, n_head // n_kv_head)")
        scale_factor = 1/spectral_norm_dict['k'](kwargs['m'], kwargs['n'], kwargs['r'])
    elif key == 'v':
        if not all(k in kwargs for k in ('m', 'r')):
            raise ValueError("spectral_norm_dict['v'] requires m, r (mup_multiplier, n_head // n_kv_head)")
        scale_factor = 1/spectral_norm_dict['v'](kwargs['m'], kwargs['r'])
    elif transpose_weights:
        scale_factor = math.sqrt(A.size(0) / A.size(1))
    else:
        scale_factor = math.sqrt(A.size(1) / A.size(0))
    return scale_factor

def natural_spectral_norm(A: torch.Tensor, return_list=False, transpose_weights=False, key=None, **kwargs) -> float | torch.Tensor:

    scale_factor = 1.0
    
    if len(A.shape) == 3:
        norms = []
        for matrix in A:
            # scale_factor = scale_factor_compute(matrix, key, transpose_weights, **kwargs)
            norms.append(spectral_norm_svd(matrix) * scale_factor)

        if return_list:
            return norms
        else:
            return torch.tensor(norms).mean()
        
    # scale_factor = scale_factor_compute(A, key, transpose_weights, **kwargs)
    return spectral_norm_svd(A) * scale_factor

@dataclass
class DataRow:
    layer: int
    width: int 
    layer_type: str # ['input', 'weight', 'bias']
    fDf: str # ['f', 'Df']
    data_type: str # ['norm', 'mean', 'var']
    value: float
    iteration: int
    seed: int
    depth: int 
    tag: str

    def as_dict(self):
        return {
            "layer": self.layer,
            "width": self.width,
            "layer_type": self.layer_type,
            "fDf": self.fDf,
            "data_type": self.data_type,
            "value": self.value,
            "iteration": self.iteration,
            "seed": self.seed,
            "depth": self.depth,
            "tag": self.tag
        }


def get_hooks(data: dict, key: str, sn_key=None, m=None, n=None, r=None) -> tuple:

    hook_data = {
        "input_natural_norm": [],
        "input_mean": [],
        "input_var": [],
        "input_diff_natural_norm": [],
        "input_diff_mean": [],
        "input_diff_var": [],
        "weight_natural_norm": [],
        "weight_mean": [],
        "weight_var": [],
        "weight_diff_natural_norm": [],
        "weight_diff_mean": [],
        "weight_diff_var": [],
        "bias_natural_norm": [],
        "bias_mean": [],
        "bias_var": [],
        "bias_diff_natural_norm": [],
        "bias_diff_mean": [],
        "bias_diff_var": [],        
    }

    data[key] = hook_data

    # persistent variables should reduce computational load
    last_input = None
    last_weight = None
    last_bias = None

    def forward_hook(module, input, _):
        nonlocal last_input, last_weight, last_bias

        if type(input) is tuple:
            input = input[0]
        
        input_norm = input.norm(p=2, dim=-1).mean() / math.sqrt(input.size(-1))
        input_mean = input.abs().mean()
        input_var = input.var()
        hook_data["input_natural_norm"].append(input_norm.item())
        hook_data["input_mean"].append(input_mean.item())
        hook_data["input_var"].append(input_var.item())

        if hasattr(module, 'weight'):
            weight = module.weight
            # Use muP scaling if sn_key is provided
            if sn_key is not None:
                sn_kwargs = {}
                if m is not None:
                    sn_kwargs['m'] = m
                if n is not None:
                    sn_kwargs['n'] = n
                if r is not None:
                    sn_kwargs['r'] = r
                weight_norm = natural_spectral_norm(weight, key=sn_key, **sn_kwargs)
            else:
                weight_norm = natural_spectral_norm(weight)
            weight_mean = weight.abs().mean()
            weight_var = weight.var()
            hook_data["weight_natural_norm"].append(weight_norm.item())
            hook_data["weight_mean"].append(weight_mean.item())
            hook_data["weight_var"].append(weight_var.item())

        if last_input is not None:
            input_diff = input - last_input
            input_diff_norm = input_diff.norm(p=2) / math.sqrt(input.size(-1))
            input_diff_mean = input_diff.abs().mean()
            input_diff_var = input_diff.var()
            hook_data["input_diff_natural_norm"].append(input_diff_norm.item())
            hook_data["input_diff_mean"].append(input_diff_mean.item())
            hook_data["input_diff_var"].append(input_diff_var.item())
    
        last_input = input.clone()

        if last_weight is not None and hasattr(module, 'weight'):
            weight = module.weight
            weight_diff = weight - last_weight
            if sn_key is not None:
                sn_kwargs = {}
                if m is not None:
                    sn_kwargs['m'] = m
                if n is not None:
                    sn_kwargs['n'] = n
                if r is not None:
                    sn_kwargs['r'] = r
                weight_diff_norm = natural_spectral_norm(weight_diff, key=sn_key, **sn_kwargs)
            else:
                weight_diff_norm = natural_spectral_norm(weight_diff)
            weight_diff_mean = weight_diff.abs().mean()
            weight_diff_var = weight_diff.var()
            hook_data["weight_diff_natural_norm"].append(weight_diff_norm.item())
            hook_data["weight_diff_mean"].append(weight_diff_mean.item())
            hook_data["weight_diff_var"].append(weight_diff_var.item())

        if hasattr(module, 'weight'):
            last_weight = weight.clone()

    return forward_hook

def new_rows_from_layer_type(layer_data, layer_type: str, width: int, seed: int, layer: str, fDf: str, depth: int, tag: str) -> DataRow:
    new_rows = []
    for data_type in ["natural_norm", "mean", "var"]:
        for idx in range(len(layer_data[f"{layer_type}_{data_type}"])):
            layer_type_name = layer_type if not fDf == 'Df' else layer_type.split('_')[0]
            new_rows.append(
                DataRow(
                    layer=layer,
                    width=width,
                    seed=seed,
                    depth=depth,
                    layer_type=layer_type_name,
                    data_type="norm" if data_type == "natural_norm" else data_type,
                    fDf=fDf,
                    value=layer_data[f"{layer_type}_{data_type}"][idx],
                    iteration=idx,
                    tag=tag
                )
            )
    return new_rows

def dataframe_from_data(data: list, width: int, seed: int, depth: int, tag: str) -> pd.DataFrame:
    rows = []
    for key, layer_data in data.items():
        if 'fc1' not in key and 'fc2' not in key:
            rows.extend(new_rows_from_layer_type(layer_data, 'input', width, seed, key, 'f', depth, tag))
            rows.extend(new_rows_from_layer_type(layer_data, 'input_diff', width, seed, key, 'Df', depth, tag))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight', width, seed, key, 'f', depth, tag))
        rows.extend(new_rows_from_layer_type(layer_data, 'weight_diff', width, seed, key, 'Df', depth, tag))

    return pd.DataFrame(rows)


def read_coord_check_data(folder_path: str) -> pd.DataFrame:
    """Read coordinate check data from a folder containing CSV files."""
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    all_dataframes = []

    # Look for CSV files in the folder
    for csv_file in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
            continue

    if not all_dataframes:
        print(f"No valid data found in {folder_path}")
        return pd.DataFrame()

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df

# --- New: plot norm by group (qkv, emb, other) ---
def plot_norms_by_group(df: pd.DataFrame, save_prefix: str = None):
    """
    Plot input norm和weight norm，分三组：q/k/v, embedding/unembedding, 其它。
    每组单独画图。
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 分组定义
    qkv_keys = ['attn.c_q', 'attn.c_kv', 'attn.c_proj']
    emb_keys = ['wte', 'wpe', 'lm_head']
    # 1. q/k/v group
    def is_qkv(row):
        return any(k in str(row['layer']) or k in str(row['tag']) or k in str(row['layer_type']) for k in qkv_keys)
    def is_emb(row):
        return any(k in str(row['layer']) or k in str(row['tag']) or k in str(row['layer_type']) for k in emb_keys)
    # 2. embedding/unembedding group
    # 3. other group
    df['group'] = df.apply(lambda row: 'qkv' if is_qkv(row) else ('emb' if is_emb(row) else 'other'), axis=1)

    for group in ['qkv', 'emb', 'other']:
        for norm_type in ['input', 'weight']:
            plt.figure(figsize=(10, 6))
            sub = df[(df['group'] == group) & (df['layer_type'] == norm_type) & (df['data_type'] == 'norm') & (df['fDf'] == 'f')]
            if sub.empty:
                continue
            for layer in sorted(sub['layer'].unique()):
                layer_data = sub[sub['layer'] == layer]
                group_data = layer_data.groupby('width')['value'].mean().reset_index()
                marker = 'o' if norm_type == 'input' else 's'
                sns.lineplot(data=group_data, x='width', y='value', marker=marker, label=f'{norm_type.capitalize()} Norm (layer {layer})')
            plt.xlabel('Width')
            plt.ylabel('Norm')
            plt.title(f'{norm_type.capitalize()} Norm vs Width ({group})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if save_prefix:
                fname = f"{save_prefix}_{group}_{norm_type}_norm.png"
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                print(f"Plot saved to {fname}")
            plt.show()


def analyze_folder(folder_path: str = "2025-09-20_18-36-08"):
    """Complete analysis pipeline: read data and create plots."""
    print(f"Reading data from folder: {folder_path}")
    
    # Check if folder exists in current directory or coord-check-impl
    possible_paths = [
        folder_path,
        f"coord-check-impl/{folder_path}",
        f"./coord-check-impl/{folder_path}"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print(f"Could not find folder. Tried: {possible_paths}")
        return
    
    print(f"Found data at: {data_path}")
    
    # Read data
    df = read_coord_check_data(data_path)
    if df.empty:
        print("No data found or could not read data")
        return
    
    print(f"Loaded {len(df)} data points")
    print(f"Unique widths: {sorted(df['width'].unique())}")
    print(f"Unique seeds: {sorted(df['seed'].unique())}")
    print(f"Unique layers: {sorted(df['layer'].unique())}")
    
    # Create plots
    plot_norms_by_group(df, save_prefix=folder_path)
    
    return df

# Example usage:
df = analyze_folder("2025-09-22_17-39-39")