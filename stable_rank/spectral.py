"""Gaussian block power estimates with a conditional probabilistic upper bound.

No full SVD: multiply A and A.T by a small block of vectors. Point estimates
are lower bounds. The powered Gaussian-probe upper bound is checked with a
union allocation over stopping checkpoints, not inferred from iterate change.
"""
from __future__ import annotations
import math
import numpy as np


def gaussian_absolute_quantile(probability):
    if not 0<probability<1:
        raise ValueError('Probability must lie in (0,1)')
    low,high=0.,2*probability if probability<.1 else 10.
    for _ in range(100):
        middle=(low+high)/2
        if math.erf(middle/math.sqrt(2))<probability:low=middle
        else:high=middle
    return (low+high)/2


def estimate_spectral(matrix, *, method='svd', probes=8, max_power=32,
                      relative_tolerance=.05, failure_probability=1e-8, seed=2027,
                      fallback_exact=False):
    A=np.asarray(matrix,dtype=np.float64)
    if A.ndim!=2 or min(A.shape)==0 or not np.isfinite(A).all():
        raise ValueError('Expected a finite, nonempty real matrix')
    if method not in ('svd','power'):
        raise ValueError('Spectral method must be svd or power')
    if probes<1 or max_power<0 or relative_tolerance<=0 or not 0<failure_probability<1:
        raise ValueError('Invalid power-estimation settings')
    frobenius=float(np.linalg.norm(A,'fro'))
    if not math.isfinite(frobenius):
        raise ValueError('Matrix norm overflow; rescale the input')
    if frobenius==0:
        return dict(value=0.,lower=0.,upper=0.,relative_gap=0.,method='zero',
                    power=0,probes=0,failure_probability=0.,certified=True)
    if method=='svd':
        value=float(np.linalg.svd(A,compute_uv=False)[0])
        return dict(value=value,lower=value,upper=value,relative_gap=0.,method='float64_svd',
                    power=0,probes=0,failure_probability=0.,certified=True)
    # Scaling keeps matrix products safe; all norms are restored at the end.
    scaled=A/frobenius
    rng=np.random.default_rng(seed)
    vectors=rng.standard_normal((A.shape[1],probes))
    norms=np.linalg.norm(vectors,axis=0)
    log_scales=np.log(norms)
    vectors/=norms
    # A fixed set of possible stopping checks allows a valid union bound.
    checks=sorted({0,max_power,*[p for p in (1,2,4,8,16,32,64,128) if p<=max_power]})
    per_check=failure_probability/len(checks)
    threshold=gaussian_absolute_quantile(per_check**(1/probes))
    lower=1/math.sqrt(min(A.shape))  # deterministic ||A|| >= ||A||F / sqrt(rank bound)
    upper=1.
    for power in range(max_power+1):
        outputs=np.einsum('ij,jk->ik',scaled,vectors,optimize=False)
        norms=np.linalg.norm(outputs,axis=0)
        lower=max(lower,float(norms.max()))
        safe=np.where(norms>0,norms,1.)
        outputs/=safe
        with np.errstate(divide='ignore'):
            log_scales+=np.log(norms)
        if power in checks:
            # T=A(A.T A)^p has norm ||A||^(2p+1). For each original Gaussian
            # g, ||Tg|| >= ||T||*|<v1,g>|, and the latter is standard |N(0,1)|.
            probabilistic_upper=math.exp((float(log_scales.max())-math.log(threshold))/(2*power+1))
            upper=min(upper,probabilistic_upper)
            if upper<lower*(1-1e-10):
                # A rare probe-bound failure cannot be advertised as convergence.
                upper=1.
            upper=max(upper,lower)
            gap=upper/lower-1
            if gap<=relative_tolerance:
                break
        if power==max_power:break
        vectors=np.einsum('ij,ik->jk',scaled,outputs,optimize=False)
        norms=np.linalg.norm(vectors,axis=0)
        lower=max(lower,float(norms.max()))
        vectors/=np.where(norms>0,norms,1.)
        with np.errstate(divide='ignore'):
            log_scales+=np.log(norms)
    certified=upper/lower-1<=relative_tolerance
    if not certified and fallback_exact:
        exact=estimate_spectral(A,method='svd')
        exact['method']='float64_svd_fallback'
        return exact
    return dict(value=lower*frobenius,lower=lower*frobenius,upper=upper*frobenius,
                relative_gap=upper/lower-1,method='gaussian_block_power',power=power,
                probes=probes,failure_probability=failure_probability,certified=certified)
