"""Instrumentation for ordinary, unsharded PyTorch AdamW/zero-decay Adam."""
from __future__ import annotations
import csv
import hashlib
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np
import torch
from .geometry import adam_direction, measure_direction, spectral_norm
from .spectral import estimate_spectral


@dataclass
class Projection:
    name: str
    parameter: torch.nn.Parameter
    kind: str
    layer: int


def select_kv_projections(model, layers=None):
    """Separate nn.Linear k_proj/v_proj weights; fused/sharded layouts require an adapter."""
    result = []
    for name, parameter in model.named_parameters():
        match = re.search(r'(?:^|\.)(k_proj|v_proj)\.weight$', name)
        if not match:
            continue
        ids = re.findall(r'(?:layers|blocks)\.(\d+)', name)
        if not ids:
            raise ValueError(f'Cannot infer layer for {name}; pass explicit Projection objects')
        layer = int(ids[-1])
        if layers is None or layer in layers:
            result.append(Projection(name, parameter, 'K' if match[1]=='k_proj' else 'V', layer))
    if not result:
        raise ValueError('No separate k_proj/v_proj weights selected')
    return result


def as_array(tensor):
    # Copy avoids aliasing CPU optimizer state or pre-step parameters.
    return tensor.detach().to(device='cpu', dtype=torch.float64).numpy().copy()


def parameter_step(state):
    value = state.get('step', 0)
    return int(value.item() if torch.is_tensor(value) else value)


class GeometryCollector:
    """Call before_step AFTER LR scheduling/clipping, then after_step after optimizer.step.

    For AMP, call after_step even on skipped steps: unchanged optimizer counters are skipped.
    Distributed/full-matrix integration is deliberately explicit, never inferred from a shard.
    """
    def __init__(self, optimizer, projections, output, *, metadata, save_matrices=False, spectral_options=None, exact_check_steps=(), exact_check_layers=None):
        if type(optimizer) not in (torch.optim.AdamW, torch.optim.Adam):
            raise TypeError('Only standard torch.optim.AdamW/Adam are reconstructed')
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_world_size()>1:
            raise ValueError('Collector needs a full-matrix distributed adapter; do not measure local shards')
        self.optimizer = optimizer
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.metadata = dict(metadata)
        reserved = set(measure_direction(np.ones((1,1)))) | {
            'parameter','projection','layer','optimizer_step','global_step','tokens_seen','loss',
            'eta','weight_decay','epsilon','beta1','beta2','amsgrad','reconstruction_relative_error',
            'reconstruction_absolute_error','weight_dtype','direction_source'}
        if reserved.intersection(self.metadata):
            raise ValueError('Metadata must not override measured geometry or optimizer fields')
        json.dumps(self.metadata, allow_nan=False)
        self.spectral_options = dict(spectral_options or {})
        self.exact_check_steps = set(exact_check_steps)
        self.exact_check_layers = set(exact_check_layers) if exact_check_layers is not None else None
        self.initial_estimates = {}
        self.initial_exact = {}
        self.save_matrices = save_matrices
        self.projections = list(projections)
        if not self.projections:
            raise ValueError('Select at least one projection')
        if len({id(x.parameter) for x in self.projections}) != len(self.projections):
            raise ValueError('A parameter cannot be selected twice')
        self.groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        self.initial = {}
        self.initial_norms = {}
        for item in self.projections:
            if id(item.parameter) not in self.groups:
                raise ValueError(f'{item.name} is absent from optimizer')
            state = optimizer.state.get(item.parameter, {})
            if parameter_step(state):
                raise ValueError('Construct collector BEFORE the first optimizer step to capture W0')
            initial = as_array(item.parameter)
            d,n=initial.shape
            if d>n or n%d:
                raise ValueError('Expected distinct KV shape (n/r,n)')
            estimate=estimate_spectral(initial,**self._norm_options(item.name,'W0'))
            self.initial_estimates[item.name]=estimate
            self.initial_norms[item.name]=estimate['value']
            if self.exact_check_steps and (self.exact_check_layers is None or item.layer in self.exact_check_layers):
                self.initial_exact[item.name]=spectral_norm(initial)
            elif estimate['failure_probability']==0:
                self.initial_exact[item.name]=estimate['value']
            if self.initial_norms[item.name] <= 0:
                raise ValueError('Nonzero W0 required')
            if save_matrices:
                self.initial[item.name] = initial
            self._validate_group(self.groups[id(item.parameter)])
        self.pending = None
        self.handle = (self.output/'geometry.csv').open('x', newline='')
        self.writer = None

    def _norm_options(self,name,step):
        options=dict(self.spectral_options)
        base=int(options.pop('seed',2027))
        salt=int.from_bytes(hashlib.sha256(f'{name}:{step}'.encode()).digest()[:8],'little')
        options['seed']=(base+salt)%(2**63-1)
        return options

    def _validate_group(self, group):
        if type(self.optimizer) is torch.optim.Adam and float(group.get('weight_decay',0)) != 0:
            raise ValueError('L2-coupled Adam decay contaminates the data gradient; use zero decay or AdamW')
        if group.get('differentiable', False):
            raise ValueError('Differentiable optimizer instrumentation is not supported')

    def before_step(self):
        if self.pending is not None:
            raise RuntimeError('Previous measurement has not been completed')
        self.pending = {}
        for item in self.projections:
            p = item.parameter
            group = self.groups[id(p)]
            self._validate_group(group)
            lr = float(group['lr'])
            if lr <= 0:
                raise ValueError('Measurement checkpoints require positive actual LR')
            self.pending[item.name] = {
                'weight': as_array(p), 'previous_step': parameter_step(self.optimizer.state.get(p,{})),
                'eta': lr, 'weight_decay': float(group.get('weight_decay',0)),
                'epsilon': float(group['eps']), 'betas': tuple(float(x) for x in group['betas']),
                'amsgrad': bool(group.get('amsgrad',False)),
            }

    def after_step(self, *, global_step, tokens_seen, loss=None):
        if self.pending is None:
            raise RuntimeError('Call before_step before a sampled optimizer update')
        records = []
        try:
            for item in self.projections:
                saved = self.pending[item.name]
                state = self.optimizer.state.get(item.parameter,{})
                step = parameter_step(state)
                if step == saved['previous_step']:
                    continue  # No gradient or AMP overflow; not a new Adam update.
                if step != saved['previous_step']+1:
                    raise RuntimeError('More than one optimizer step occurred between collector calls')
                R = adam_direction(as_array(state['exp_avg']), as_array(state['exp_avg_sq']),
                                   step, saved['betas'], saved['epsilon'],
                                   max_v=as_array(state['max_exp_avg_sq']) if saved['amsgrad'] else None)
                initial=self.initial_estimates[item.name]
                row = measure_direction(R, w0_spectral=self.initial_norms[item.name], eta=saved['eta'],
                                        spectral_options=self._norm_options(item.name,step),
                                        w0_bounds=(initial['lower'],initial['upper']))
                exact=None
                verify=(global_step in self.exact_check_steps and
                        (self.exact_check_layers is None or item.layer in self.exact_check_layers))
                if verify:
                    exact=spectral_norm(R)
                elif row['spectral_failure_probability']==0:
                    exact=row['spectral']
                inside=(row['spectral_lower']-1e-9<=exact<=row['spectral_upper']+1e-9) if exact is not None else None
                row.update(w0_spectral_failure_probability=initial['failure_probability'],
                           spectral_exact_reference=exact,
                           spectral_reference_relative_error=(exact-row['spectral'])/exact if exact else None,
                           spectral_interval_contains_reference=inside,
                           stable_rank_exact_reference=(row['frobenius']/exact)**2 if exact else None,
                           balance_exact_reference=saved['eta']*exact/self.initial_exact[item.name]
                           if exact is not None and item.name in self.initial_exact else None)
                if inside is False:
                    row['spectral_certified']=False
                actual_delta = as_array(item.parameter) - saved['weight']
                reconstructed_delta = -saved['eta']*R-saved['eta']*saved['weight_decay']*saved['weight']
                denominator = saved['eta']*row['frobenius']
                error = float(np.linalg.norm(actual_delta-reconstructed_delta,'fro'))
                row.update(self.metadata)
                row.update(parameter=item.name, projection=item.kind, layer=item.layer,
                           optimizer_step=step, global_step=int(global_step), tokens_seen=int(tokens_seen),
                           loss=float(loss) if loss is not None else None,
                           eta=saved['eta'], weight_decay=saved['weight_decay'], epsilon=saved['epsilon'],
                           beta1=saved['betas'][0], beta2=saved['betas'][1], amsgrad=saved['amsgrad'],
                           reconstruction_relative_error=error/denominator if denominator else None,
                           reconstruction_absolute_error=error,
                           weight_dtype=str(item.parameter.dtype), direction_source='post_step_adam_moments')
                if self.save_matrices:
                    directory = self.output/'matrices'
                    directory.mkdir(exist_ok=True)
                    filename = f'{item.kind}_layer{item.layer}_global{global_step}_adam{step}.npz'
                    path = directory/filename
                    if path.exists():
                        raise FileExistsError(path)
                    np.savez_compressed(path, R=R, W0=self.initial[item.name], step=np.array(step),
                                        metadata=np.array(json.dumps(row, allow_nan=False)))
                if self.writer is None:
                    self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
                records.append(row)
            self.handle.flush()
            return records
        finally:
            self.pending = None

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
