"""Finite-dimensional geometry; NumPy only, independent of the training loop."""
from __future__ import annotations
import math
import numpy as np
from .spectral import estimate_spectral


def spectral_norm(matrix):
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or min(value.shape) == 0 or not np.isfinite(value).all():
        raise ValueError('Expected a finite, nonempty real matrix')
    return float(np.linalg.svd(value, compute_uv=False)[0])


def measure_direction(direction, *, w0_spectral=None, eta=None, spectral_options=None, w0_bounds=None):
    matrix = np.asarray(direction, dtype=np.float64)
    estimate = estimate_spectral(matrix, **(spectral_options or {}))
    spectral = estimate["value"]
    d, n = matrix.shape
    if d > n or n % d:
        raise ValueError('Distinct KV matrix must have shape (n/r, n) with integer r')
    frobenius = float(np.linalg.norm(matrix, 'fro'))
    rms = frobenius / math.sqrt(d * n)
    r = n // d
    stable_rank = (frobenius / spectral) ** 2 if spectral else None
    if stable_rank is not None and not (1-1e-9 <= stable_rank <= d+1e-9):
        raise ArithmeticError('Stable rank outside the mathematical bounds')
    if w0_spectral is not None and (not math.isfinite(w0_spectral) or w0_spectral <= 0):
        raise ValueError('Initialization spectral norm must be positive and finite')
    if eta is not None and (not math.isfinite(eta) or eta <= 0):
        raise ValueError('Actual learning rate must be positive and finite')
    lower,upper=estimate['lower'],estimate['upper']
    initial_lower,initial_upper=w0_bounds if w0_bounds is not None else (w0_spectral,w0_spectral)
    if initial_lower is not None and not (0<initial_lower<=initial_upper):
        raise ValueError('Initialization bounds must be positive and ordered')
    return {
        'n': n, 'd': d, 'r': r, 'status': 'ok' if spectral else 'zero_direction',
        'frobenius': frobenius, 'spectral': spectral, 'stable_rank': stable_rank,
        'rms_entry': rms,
        'q_over_sqrt_s': rms / math.sqrt(stable_rank) if spectral else 0.,
        'geometry_lr_factor': (1+math.sqrt(r))*math.sqrt(stable_rank)/(n*rms) if spectral else None,
        'coherent_lr_factor': (1+math.sqrt(r))/(2*n),
        'w0_spectral': w0_spectral,
        'exact_balance_eta': w0_spectral/spectral if w0_spectral is not None and spectral else None,
        'realized_balance': eta*spectral/w0_spectral if eta is not None and w0_spectral is not None else None,
        'spectral_method': estimate['method'],
        'spectral_lower': lower, 'spectral_upper': upper,
        'spectral_relative_gap': estimate['relative_gap'],
        'spectral_certified': estimate['certified'],
        'spectral_failure_probability': estimate['failure_probability'],
        'spectral_power': estimate['power'], 'spectral_probes': estimate['probes'],
        'stable_rank_lower': max(1.,(frobenius/upper)**2) if spectral else None,
        'stable_rank_upper': min(d,(frobenius/lower)**2) if spectral else None,
        'w0_spectral_lower': initial_lower, 'w0_spectral_upper': initial_upper,
        'balance_lower': eta*lower/initial_upper if eta is not None and initial_upper is not None else None,
        'balance_upper': eta*upper/initial_lower if eta is not None and initial_lower is not None else None,
        'estimate_kind': 'exact_numerical' if estimate['failure_probability']==0 else 'upper_bound_for_rank',

    }


def adam_direction(m, v, step, betas, epsilon, *, max_v=None):
    """Reconstruct the data direction from POST-step uncorrected Adam moments."""
    b1, b2 = map(float, betas)
    if not (0 <= b1 < 1 and 0 <= b2 < 1 and math.isfinite(epsilon) and epsilon >= 0):
        raise ValueError('Invalid Adam betas/epsilon')
    if isinstance(step, bool) or int(step) != step or step < 1:
        raise ValueError('Use the positive per-parameter optimizer step')
    m, v = np.asarray(m, dtype=np.float64), np.asarray(v, dtype=np.float64)
    variance = np.asarray(max_v, dtype=np.float64) if max_v is not None else v
    if m.shape != v.shape or variance.shape != m.shape:
        raise ValueError('Moments must have identical shapes')
    if not all(np.isfinite(x).all() for x in (m, v, variance)) or np.any(v < 0) or np.any(variance < 0):
        raise ValueError('Moments must be finite with nonnegative second moments')
    if max_v is not None and np.any(variance < v):
        raise ValueError('AMSGrad maximum must dominate current second moment')
    denominator = np.sqrt(variance/(1-b2**int(step))) + epsilon
    if np.any(denominator == 0):
        raise ValueError('Zero Adam denominator: export the actual implementation direction instead')
    return (m/(1-b1**int(step))) / denominator
