"""Aggregate geometry by seed, fit descriptive scaling laws, and export figures."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import numpy as np

METRICS=['stable_rank','rms_entry','spectral','realized_balance','q_over_sqrt_s','reconstruction_relative_error',
         'stable_rank_lower','stable_rank_upper','balance_lower','balance_upper','spectral_relative_gap']
CELL=['rule','data_mode','projection','layer','global_step','n','r']


def load_rows(root):
    files=sorted(Path(root).rglob('geometry.csv'))
    if not files:
        raise ValueError('No geometry.csv files found')
    rows=[];identities=set();protocols=set()
    for path in files:
        # Exclude incomplete runner outputs; standalone integration has no config.json.
        if (path.parent/'config.json').exists() and not (path.parent/'complete.json').exists():
            raise ValueError(f'Incomplete run: {path.parent}')
        if (path.parent/'config.json').exists():
            config=json.loads((path.parent/'config.json').read_text())
            keys=['base_lr','reference_width','reference_r','warmup_steps','steps','depth','head_dim',
                  'vocab_size','seq_len','batch_size','sigma0','epsilon0','beta1','beta2','weight_decay',
                  'attention_scale','train_file','val_file','synthetic']
            protocols.add(json.dumps({key:config.get(key) for key in keys},sort_keys=True))
            if len(protocols)>1:
                raise ValueError('Mixed training protocols in one geometry fit; analyze each LR/budget/dataset separately')
        with path.open(newline='') as handle:
            for raw in csv.DictReader(handle):
                for field in ['seed','layer','global_step','optimizer_step','n','d','r']:
                    raw[field]=int(raw[field])
                for field in METRICS+['frobenius','eta','w0_spectral','geometry_lr_factor','exact_balance_eta']:
                    raw[field]=float(raw[field]) if raw.get(field) else None
                for field in ['stable_rank_lower','stable_rank_upper']:
                    if raw[field] is None: raw[field]=raw['stable_rank']
                for field in ['balance_lower','balance_upper']:
                    if raw[field] is None: raw[field]=raw['realized_balance']
                raw['spectral_certified']=raw.get('spectral_certified','True')=='True'
                for field in ['spectral_failure_probability','w0_spectral_failure_probability']:
                    raw[field]=float(raw.get(field) or 0)
                identity=(raw['run_id'],raw['parameter'],raw['global_step'])
                if identity in identities:
                    raise ValueError(f'Duplicate measurement {identity}')
                identities.add(identity)
                rows.append(raw)
    return rows


def summaries(rows,bootstrap=1000):
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='ok':
            groups[tuple(row[k] for k in CELL)].append(row)
    result=[]
    rng=np.random.default_rng(2027)
    for key,values in sorted(groups.items()):
        by_seed=defaultdict(list)
        for row in values:
            by_seed[row['seed']].append(row)
        # A cell contains one layer/projection/checkpoint; multiple records per seed are accidental replicates.
        if any(len(x)!=1 for x in by_seed.values()):
            raise ValueError(f'Multiple records for one seed in cell {key}; use separate output roots for LR sweeps')
        out=dict(zip(CELL,key));out['seed_count']=len(by_seed)
        for metric in METRICS:
            array=np.array([value[0][metric] for value in by_seed.values() if value[0][metric] is not None])
            out[metric+'_mean']=float(np.mean(array)) if len(array) else None
            if len(array)>1:
                sampled=array[rng.integers(0,len(array),size=(bootstrap,len(array)))].mean(axis=1)
                out[metric+'_low'],out[metric+'_high']=map(float,np.quantile(sampled,[.025,.975]))
            else:
                out[metric+'_low']=out[metric+'_high']=None
        result.append(out)
    return result


def fit_laws(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='ok' and row['rms_entry']>0:
            key=tuple(row[x] for x in ['rule','data_mode','projection','layer','global_step'])
            groups[key].append(row)
    fits=[]
    for key,values in sorted(groups.items()):
        # Cell mean log metric: each (n,r) contributes once, irrespective of seed count.
        cells=defaultdict(list)
        for row in values:
            cells[(row['n'],row['r'])].append(row)
        dims=sorted(cells)
        X=np.array([[1,math.log(n),math.log(r)] for n,r in dims])
        out=dict(zip(['rule','data_mode','projection','layer','global_step'],key))
        out.update(widths=sorted({n for n,r in dims}),repetitions=sorted({r for n,r in dims}),cell_count=len(dims))
        if len(dims)<3 or np.linalg.matrix_rank(X)<3:
            out['status']='not_identifiable';fits.append(out);continue
        if any(not row.get('spectral_certified',True) or (row.get('spectral_relative_gap') or 0)>.051 for row in values):
            out['status']='norm_estimates_not_tight';fits.append(out);continue
        out['status']='descriptive_fit' 
        for label,metric in [('s','stable_rank'),('q','rms_entry')]:
            y=np.array([np.mean([math.log(row[metric]) for row in cells[dim]]) for dim in dims])
            coefficients=np.linalg.lstsq(X,y,rcond=None)[0]
            residual=y-X@coefficients
            out[label]={'intercept':float(coefficients[0]),'alpha':float(coefficients[1]),'beta':float(coefficients[2]),
                        'log_rmse':float(np.sqrt(np.mean(residual**2)))}
        fits.append(out)
    return fits


def time_diagnostics(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[(row['run_id'],row['parameter'])].append(row)
    output=[]
    for (run,param),values in sorted(groups.items()):
        positive=[r for r in values if r['status']=='ok' and r['q_over_sqrt_s']>0]
        all_positive=len(positive)==len(values)
        u=[r['q_over_sqrt_s'] for r in positive]
        output.append({'run_id':run,'parameter':param,'checkpoints':len(values),
                       'all_directions_nonzero':all_positive,
                       'max_min_q_over_sqrt_s':max(u)/min(u) if all_positive and u else None,
                       'stable_rank_max_min':max(r['stable_rank'] for r in positive)/min(r['stable_rank'] for r in positive) if positive else None,
                       'max_reconstruction_relative_error':max((r['reconstruction_relative_error'] or 0) for r in values)})
    return output


def write_csv(path,rows):
    if not rows:
        path.write_text('');return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def figures(cells,rows,output):
    os.environ.setdefault('MPLCONFIGDIR',str(output/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    groups=defaultdict(list)
    for row in cells:
        groups[(row['rule'],row['data_mode'],row['projection'],row['layer'],row['global_step'])].append(row)
    destination=output/'figures';destination.mkdir(exist_ok=True)
    for (rule,data,projection,layer,step),values in groups.items():
        fig,axes=plt.subplots(1,3,figsize=(11,3.2),layout='constrained')
        for axis,metric,label in zip(axes,['stable_rank','rms_entry','realized_balance'],['Update stable rank s','RMS update energy q','Spectral balance B']):
            for r in sorted({x['r'] for x in values}):
                points=sorted((x for x in values if x['r']==r),key=lambda x:x['n'])
                x=np.array([v['n'] for v in points]);y=np.array([v[metric+'_mean'] for v in points])
                axis.plot(x,y,'o-',label=f'r={r}')
                if all(v[metric+'_low'] is not None for v in points):
                    axis.fill_between(x,[v[metric+'_low'] for v in points],[v[metric+'_high'] for v in points],alpha=.12)
            axis.set_xscale('log',base=2)
            all_y=[v[metric+'_mean'] for v in values]
            if metric=='rms_entry' and max(all_y)/min(all_y)<2:
                axis.set_ylim(0,1.15*max(all_y))
            else:
                axis.set_yscale('log')
            axis.set_xlabel('Model width n');axis.set_ylabel(label);axis.grid(alpha=.2)
        axes[0].legend();fig.suptitle(f'{data} | {rule} | {projection}, layer {layer}, step {step}')
        basename=f'{data}_{rule}_{projection}_layer{layer}_step{step}'
        fig.savefig(destination/(basename+'.pdf'),bbox_inches='tight',pad_inches=.12);fig.savefig(destination/(basename+'.png'),dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    # Seed curves: do not treat checkpoints or layers as independent replicates.
    temporal=defaultdict(list)
    for row in rows:
        if row['status']=='ok':
            temporal[(row['rule'],row['data_mode'],row['projection'],row['layer'],row['n'],row['r'])].append(row)
    for (rule,data,projection,layer,n,r),values in temporal.items():
        fig,axis=plt.subplots(figsize=(5.5,3.5),layout='constrained')
        for seed in sorted({x['seed'] for x in values}):
            seq=sorted((x for x in values if x['seed']==seed),key=lambda x:x['global_step'])
            axis.plot([x['global_step'] for x in seq],[x['q_over_sqrt_s'] for x in seq],'.-',label=f'seed={seed}')
        axis.set_xscale('log',base=2);axis.set_yscale('log');axis.set_xlabel('Optimizer global step');axis.set_ylabel('q / sqrt(s)');axis.legend();axis.grid(alpha=.2)
        axis.set_title(f'{data} | {rule} | {projection} L{layer} | n={n}, r={r}')
        fig.savefig(destination/f'temporal_{data}_{rule}_{projection}_L{layer}_n{n}_r{r}.pdf',bbox_inches='tight',pad_inches=.12);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true');p.add_argument('--bootstrap',type=int,default=1000)
    args=p.parse_args()
    if args.bootstrap<1:p.error('--bootstrap must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    rows=load_rows(args.input);cells=summaries(rows,args.bootstrap)
    write_csv(args.output/'summary.csv',cells);write_csv(args.output/'temporal.csv',time_diagnostics(rows))
    (args.output/'fits.json').write_text(json.dumps(fit_laws(rows),indent=2,allow_nan=False))
    notes={'row_count':len(rows),'zero_directions':sum(r['status']!='ok' for r in rows),
           'data_modes':sorted({r['data_mode'] for r in rows}),
           'uncertainty':'95% percentile seed-bootstrap interval; one seed has no interval; small seed counts are weak evidence',
           'fits':'Descriptive log-linear fits, not asymptotic laws or held-out validation',
           'temporal':'Only recorded checkpoints, not all training steps',
           'numerical_uncertainty':'Power point norm is a lower bound; point stable rank is an upper estimate; numerical intervals are distinct from seed CIs',
           'uncertified_rows':sum(not r.get('spectral_certified',True) for r in rows),
           'conservative_norm_failure_union_bound':min(1.,sum(r['spectral_failure_probability']+r['w0_spectral_failure_probability'] for r in rows))}
    (args.output/'analysis.json').write_text(json.dumps(notes,indent=2))
    if not args.no_plots and cells:
        figures(cells,rows,args.output)
    print(json.dumps(notes),flush=True)


if __name__=='__main__':main()
