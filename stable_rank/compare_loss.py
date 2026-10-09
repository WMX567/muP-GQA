"""Seed-level validation comparison of completed target candidates (no geometry refitting)."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    groups=defaultdict(list);identities=set();protocols=set()
    for path in sorted(args.input.rglob('complete.json')):
        config=json.loads((path.parent/'config.json').read_text());result=json.loads(path.read_text())
        # Across candidates, architecture width/r are explicit grouping variables; all other budget/data settings must match.
        keys=['depth','head_dim','vocab_size','seq_len','batch_size','steps','eval_batches','warmup_steps',
              'sigma0','epsilon0','beta1','beta2','weight_decay','attention_scale','train_file','val_file','synthetic',
              'reference_width','reference_r']
        protocols.add(json.dumps({key:config.get(key) for key in keys},sort_keys=True))
        if len(protocols)>1:raise ValueError('Loss comparison requires matching data, evaluation and training protocol')
        key=(config['width'],config['repetition'],config['rule'],config['base_lr'],config.get('kv_multiplier'))
        identity=key+(config['seed'],)
        if identity in identities:raise ValueError('Duplicate seed/candidate result')
        identities.add(identity);groups[key].append((config['seed'],result['validation_loss']))
    if not groups:raise ValueError('No completed target runs')
    # Matching seed sets required within each architecture.
    architecture=defaultdict(list)
    for key,values in groups.items():architecture[key[:2]].append({x[0] for x in values})
    if any(any(s!=sets[0] for s in sets) for sets in architecture.values()):
        raise ValueError('Candidate comparisons need the same seed set')
    rows=[];rng=np.random.default_rng(2027)
    for key,values in sorted(groups.items(),key=lambda x:str(x[0])):
        losses=np.array([v for seed,v in values],dtype=float)
        if not np.isfinite(losses).all():raise ValueError('Nonfinite losses')
        low=high=None
        if len(losses)>1:
            boot=losses[rng.integers(0,len(losses),size=(2000,len(losses)))].mean(axis=1)
            low,high=map(float,np.quantile(boot,[.025,.975]))
        rows.append(dict(width=key[0],r=key[1],rule=key[2],base_lr=key[3],kv_multiplier=key[4],
                         seed_count=len(losses),mean_validation_loss=float(losses.mean()),low=low,high=high))
    with args.output.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote',args.output,'; seed-bootstrap intervals do not establish optimality outside evaluated rates.')


if __name__=='__main__':main()
