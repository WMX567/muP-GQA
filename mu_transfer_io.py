"""Typed NumPy storage for muTransfer metrics, with legacy CSV read support."""
import csv
from pathlib import Path

import numpy as np

METRICS_DTYPE = np.dtype([
    ('step', '<i8'), ('train_loss', '<f8'), ('val_loss', '<f8'),
    ('lr', '<f8'), ('tokens', '<i8'), ('seconds', '<f8'),
])


def save_array(path, array):
    """Replace a complete snapshot atomically; never serialize Python objects."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as f:
        np.save(f, array, allow_pickle=False)
    temp.replace(path)


def metrics_array(rows):
    return np.array([
        tuple(row[name] if row[name] is not None else np.nan for name in METRICS_DTYPE.names)
        for row in rows
    ], dtype=METRICS_DTYPE)


def load_metrics(path):
    """Read named numeric columns; existing v2 CSV logs remain resumable."""
    path = Path(path)
    if path.exists():
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1 or array.dtype != METRICS_DTYPE:
            raise ValueError(f'{path}: invalid metrics schema')
        return array
    legacy = path.with_suffix('.csv')
    with legacy.open(newline='') as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({name: int(row[name]) if name in ('step', 'tokens')
                         else float(row[name]) if row[name] else np.nan
                         for name in METRICS_DTYPE.names})
    return metrics_array(rows)


def has_metrics(path):
    path = Path(path)
    return path.exists() or path.with_suffix('.csv').exists()


LEGACY_DTYPE = np.dtype([
    ('step', '<i8'), ('loss', '<f8'), ('width', '<i8'), ('lr', '<f8'),
    ('wd', '<f8'), ('seed', '<i8'), ('param_type', '<U3'),
])


def load_legacy(path):
    """Read the original muTransfer schema without changing its step semantics."""
    path = Path(path)
    if path.suffix == '.npy':
        array = np.load(path, allow_pickle=False)
        if array.ndim != 1 or array.dtype != LEGACY_DTYPE:
            raise ValueError(f'{path}: invalid legacy schema')
        return array
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != LEGACY_DTYPE.names:
            raise ValueError(f'{path}: invalid legacy CSV columns')
        rows = []
        for row in reader:
            if len(row['param_type']) > 3:
                raise ValueError(f'{path}: unexpected param_type')
            rows.append(tuple(int(row[name]) if name in ('step', 'width', 'seed')
                              else row[name] if name == 'param_type' else float(row[name])
                              for name in LEGACY_DTYPE.names))
    return np.array(rows, dtype=LEGACY_DTYPE)
