"""Typed NumPy storage for muTransfer metrics."""
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
    """Read named numeric columns, validating the schema."""
    array = np.load(path, allow_pickle=False)
    if array.ndim != 1 or array.dtype != METRICS_DTYPE:
        raise ValueError(f'{path}: invalid metrics schema')
    return array
