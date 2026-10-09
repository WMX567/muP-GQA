"""Three research figures: s vs width, q vs width, and B vs progress.

No fake assets: figures are made only from logged geometry.csv. Synthetic mode
is printed on every plot and cannot be exported as the manuscript's main data.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
from .analyze import load_rows, summaries, write_csv


def slope_profiles(rows,bootstrap=1000):
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='ok':
            groups[(row['rule'],row['data_mode'],row['projection'],row['layer'],row['global_step'],row['r'])].append(row)
    results=[];rng=np.random.default_rng(419)
    for key,values in sorted(groups.items()):
        widths=sorted({x['n'] for x in values});seeds=sorted({x['seed'] for x in values})
        if len(widths)<2:continue
        by_pair={(x['seed'],x['n']):x for x in values}
        if len(by_pair)!=len(values):raise ValueError('Duplicate seed/width observations')
        if any((seed,n) not in by_pair for seed in seeds for n in widths):
            raise ValueError('Slope fits require a matched complete seed-by-width grid')
        x=np.log(widths);center=x-x.mean();weights=center/(center@center)
        log_s=np.array([[math.log(by_pair[(seed,n)]['stable_rank']) for n in widths] for seed in seeds])
        log_q=np.array([[math.log(by_pair[(seed,n)]['rms_entry']) for n in widths] for seed in seeds])
        alpha_s=float(log_s.mean(axis=0)@weights);alpha_q=float(log_q.mean(axis=0)@weights)
        lower=np.array([[math.log(by_pair[(seed,n)]['stable_rank_lower'] or by_pair[(seed,n)]['stable_rank']) for n in widths] for seed in seeds]).mean(axis=0)
        upper=np.array([[math.log(by_pair[(seed,n)]['stable_rank_upper'] or by_pair[(seed,n)]['stable_rank']) for n in widths] for seed in seeds]).mean(axis=0)
        numerical_low=float(np.where(weights>=0,lower,upper)@weights)
        numerical_high=float(np.where(weights>=0,upper,lower)@weights)
        row=dict(zip(['rule','data_mode','projection','layer','global_step','r'],key))
        row.update(width_count=len(widths),seed_count=len(seeds),alpha_s=alpha_s,alpha_q=alpha_q,
                   implied_eta_width_exponent=alpha_s/2-alpha_q-1,
                   alpha_s_numerical_low=numerical_low,alpha_s_numerical_high=numerical_high,
                   alpha_s_seed_low=None,alpha_s_seed_high=None,eta_exponent_seed_low=None,eta_exponent_seed_high=None)
        if len(seeds)>1:
            selection=rng.integers(0,len(seeds),size=(bootstrap,len(seeds)))
            a=log_s[selection].mean(axis=1)@weights
            q=log_q[selection].mean(axis=1)@weights
            row['alpha_s_seed_low'],row['alpha_s_seed_high']=map(float,np.quantile(a,[.025,.975]))
            row['eta_exponent_seed_low'],row['eta_exponent_seed_high']=map(float,np.quantile(a/2-q-1,[.025,.975]))
        results.append(row)
    return results


def export_paper_figures(rows,output,*,layer,steps,bootstrap=1000):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[x for x in rows if x['layer']==layer]
    if not rows:raise ValueError('Selected layer was not logged')
    cells=summaries(rows,bootstrap)
    data_modes={x['data_mode'] for x in cells}
    if len(data_modes)!=1:raise ValueError('Do not merge synthetic and token-file figures')
    data=next(iter(data_modes))
    available=sorted({x['global_step'] for x in cells})
    steps=steps or available
    if any(t not in available for t in steps):raise ValueError('Requested figure checkpoints were not logged')
    os.environ.setdefault('MPLCONFIGDIR',str(output/'mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    rules=sorted({x['rule'] for x in cells});projections=sorted({x['projection'] for x in cells})
    # Each row is one rule/projection; columns are matched physical checkpoints.
    panels=[(rule,projection) for rule in rules for projection in projections]
    for metric,title,filename,bounds in [
        ('stable_rank','Normalized-update stable rank','stable_rank_vs_width',('stable_rank_lower','stable_rank_upper')),
        ('rms_entry','Normalized-update RMS energy','energy_vs_width',None)]:
        fig,axes=plt.subplots(len(panels),len(steps),figsize=(3*len(steps),2.5*len(panels)),squeeze=False,layout='constrained')
        for i,(rule,projection) in enumerate(panels):
            for j,step in enumerate(steps):
                ax=axes[i,j];values=[v for v in cells if v['rule']==rule and v['projection']==projection and v['global_step']==step]
                for r in sorted({v['r'] for v in values}):
                    pts=sorted((v for v in values if v['r']==r),key=lambda v:v['n'])
                    x=[v['n'] for v in pts];y=[v[metric+'_mean'] for v in pts]
                    ax.plot(x,y,'o-',label=f'r={r}')
                    if all(v[metric+'_low'] is not None for v in pts):
                        ax.fill_between(x,[v[metric+'_low'] for v in pts],[v[metric+'_high'] for v in pts],alpha=.15)
                    if bounds and any((v[bounds[1]+'_mean']-v[bounds[0]+'_mean'])>1e-9 for v in pts):
                        ax.vlines(x,[v[bounds[0]+'_mean'] for v in pts],[v[bounds[1]+'_mean'] for v in pts],alpha=.6,linewidth=1)
                ax.set_xscale('log',base=2)
                if metric=='stable_rank':ax.set_yscale('log',base=2)
                elif values:ax.set_ylim(0,1.15*max(v[metric+'_high'] or v[metric+'_mean'] for v in values))
                ax.set_xlabel('Width n');ax.set_ylabel('s (log scale)' if metric=='stable_rank' else 'q')
                ax.set_title(f'{rule}, {projection}, step {step}');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f'{title} | layer {layer} | {data}\nSeed-bootstrap shading; norm-estimation bounds are thin vertical ranges')
        for ext in ['pdf','png']:fig.savefig(output/(filename+'.'+ext),dpi=180,bbox_inches='tight',pad_inches=.12)
        plt.close(fig)
    repetitions=sorted({x['r'] for x in cells})
    fig,axes=plt.subplots(len(panels),len(repetitions),figsize=(3*len(repetitions),2.5*len(panels)),squeeze=False,layout='constrained')
    for i,(rule,projection) in enumerate(panels):
        for j,r in enumerate(repetitions):
            ax=axes[i,j];values=[v for v in cells if v['rule']==rule and v['projection']==projection and v['r']==r]
            for n in sorted({v['n'] for v in values}):
                pts=sorted((v for v in values if v['n']==n),key=lambda v:v['global_step'])
                x=[v['global_step'] for v in pts];y=[v['realized_balance_mean'] for v in pts]
                ax.plot(x,y,'o-',label=f'n={n}')
                if all(v['realized_balance_low'] is not None for v in pts):
                    ax.fill_between(x,[v['realized_balance_low'] for v in pts],[v['realized_balance_high'] for v in pts],alpha=.15)
                if any(v['balance_upper_mean']-v['balance_lower_mean']>1e-9 for v in pts):
                    ax.vlines(x,[v['balance_lower_mean'] for v in pts],[v['balance_upper_mean'] for v in pts],alpha=.6,linewidth=1)
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Training step');ax.set_ylabel('Spectral ratio B')
            ax.set_title(f'{rule}, {projection}, r={r}');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle(f'Achieved spectral update balance | layer {layer} | {data}')
    for ext in ['pdf','png']:fig.savefig(output/('balance_vs_time.'+ext),dpi=180,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    profiles=slope_profiles(rows,bootstrap)
    write_csv(output/'width_exponents.csv',profiles)
    manifest={'data_mode':data,'layer':layer,'checkpoints':steps,'rules':rules,'row_count':len(rows),
              'manuscript_eligible':data=='token_file' and all(x.get('spectral_certified',True) and x.get('spectral_relative_gap',0)<=.051 for x in rows),
              'norm_intervals_tight':all(x.get('spectral_certified',True) and x.get('spectral_relative_gap',0)<=.051 for x in rows),
              'scope':'Finite-width descriptive slopes; q exponent is included; synthetic is not paper evidence'}
    (output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,default=0);p.add_argument('--steps',help='Comma-separated checkpoints; default all')
    p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    if a.bootstrap<1:p.error('--bootstrap must be positive')
    print(json.dumps(export_paper_figures(load_rows(a.input),a.output,layer=a.layer,
                                         steps=[int(x) for x in a.steps.split(',')] if a.steps else None,bootstrap=a.bootstrap)))


if __name__=='__main__':main()
