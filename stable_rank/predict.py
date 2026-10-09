"""Freeze proxy-only geometry extrapolations; no target measurements are accepted."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def multiplier(fit,n0,r0,n,r):
    def estimate(law,n,r):
        return math.exp(law['intercept']+law['alpha']*math.log(n)+law['beta']*math.log(r))
    s0,s=estimate(fit['s'],n0,r0),estimate(fit['s'],n,r)
    q0,q=estimate(fit['q'],n0,r0),estimate(fit['q'],n,r)
    if not (1<=s0<=n0/r0 and 1<=s<=n/r):
        raise ValueError('Extrapolated stable rank violates [1,d]; inspect the fit instead of silently clipping')
    a=(n0/n)*(1+math.sqrt(r))/(1+math.sqrt(r0))*math.sqrt(s/s0)*(q0/q)
    return dict(predicted_reference_s=s0,predicted_target_s=s,predicted_reference_q=q0,predicted_target_q=q,
                vanilla_multiplier=n0/n,coherent_multiplier=(n0/n)*(1+math.sqrt(r))/(1+math.sqrt(r0)),
                geometry_multiplier=a)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--rule',choices=['vanilla','coherent'],required=True)
    p.add_argument('--step',type=int,required=True)
    p.add_argument('--reference-width',type=int,required=True);p.add_argument('--reference-r',type=int,default=1)
    p.add_argument('--target-width',type=int,required=True);p.add_argument('--target-r',type=int,required=True)
    args=p.parse_args()
    n0,r0,n,r=args.reference_width,args.reference_r,args.target_width,args.target_r
    if min(n0,r0,n,r)<1 or n0%r0 or n%r:p.error('Positive dimensions and integer n/r required')
    source=args.fits.read_bytes();fits=json.loads(source)
    selected=[f for f in fits if f['rule']==args.rule and f['global_step']==args.step]
    if not selected:raise SystemExit('No matching proxy checkpoint/rule fits')
    if any(f['data_mode']!='token_file' for f in selected):
        raise SystemExit('Synthetic fits are pipeline checks and cannot be frozen as real-data predictions')
    for fit in selected:
        if fit['status']!='descriptive_fit':raise SystemExit('Proxy design does not identify both width/r exponents')
        if max(fit['widths'])>=n:raise SystemExit('Target width must exceed every fitted width; target geometry would leak into the fit')
        if n0 not in fit['widths'] or r0 not in fit['repetitions']:
            raise SystemExit('Reference width/r must be covered by the proxy design')
    records=[]
    for fit in selected:
        records.append(dict(layer=fit['layer'],projection=fit['projection'],**multiplier(fit,n0,r0,n,r)))
    # Explicit common-KV-rate aggregation, fixed before target evaluation.
    common=math.exp(sum(math.log(x['geometry_multiplier']) for x in records)/len(records))
    artifact={'mode':'proxy_only_extrapolation','fit_sha256':hashlib.sha256(source).hexdigest(),
              'reference_width':n0,'reference_r':r0,'target_width':n,'target_r':r,'rule':args.rule,'checkpoint':args.step,
              'aggregation':'geometric mean over selected K/V layers','common_kv_multiplier':common,'layers':records,
              'scope':'Prediction of a candidate spectral-balance multiplier, not loss-optimal LR; geometry depends on pilot history'}
    with args.output.open('x') as handle:json.dump(artifact,handle,indent=2,allow_nan=False)
    print(f'Frozen --rules custom --kv-multiplier {common:.12g}; file {args.output}')


if __name__=='__main__':main()
