"""Run matched-data, multi-seed GQA geometry audits; never download data implicitly."""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import platform
import random
import time
import numpy as np
import torch
from .collector import GeometryCollector, select_kv_projections
from .model import TinyGQALM, make_optimizer


def parse_ints(value):
    return [int(x) for x in value.split(',')]


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train-data',type=Path)
    p.add_argument('--val-data',type=Path)
    p.add_argument('--token-dtype',choices=['uint16','uint32','int32','int64'],default='uint16',help='Raw .bin dtype; .npy carries its own dtype')
    p.add_argument('--synthetic',action='store_true',help='Pipeline test only: uniform random tokens, not OpenWebText evidence')
    p.add_argument('--widths',type=parse_ints,default=[512,1024,2048])
    p.add_argument('--repetitions',type=parse_ints,default=[1,2,4,8])
    p.add_argument('--seeds',type=parse_ints,default=[1,2,3])
    p.add_argument('--rules',default='vanilla,coherent')
    p.add_argument('--depth',type=int,default=4)
    p.add_argument('--head-dim',type=int,default=64)
    p.add_argument('--vocab-size',type=int,default=50304)
    p.add_argument('--seq-len',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--steps',type=int,default=256)
    p.add_argument('--checkpoints',type=parse_ints,default=[1,8,32,128,256])
    p.add_argument('--progress-checkpoints',action='store_true',help='Also log step 1, end warmup, 25%, 50%, 100%')
    p.add_argument('--log-every',type=int,default=0,help='Additional measurement cadence; 1 measures every step')
    p.add_argument('--norm-method',choices=['svd','power'],default='svd')
    p.add_argument('--norm-probes',type=int,default=8)
    p.add_argument('--norm-max-power',type=int,default=32)
    p.add_argument('--norm-tolerance',type=float,default=.05)
    p.add_argument('--norm-failure-probability',type=float,default=1e-8)
    p.add_argument('--norm-seed',type=int,default=2027)
    p.add_argument('--norm-fallback-exact',action='store_true')
    p.add_argument('--exact-checkpoints',type=parse_ints,default=[])
    p.add_argument('--exact-check-layers',type=parse_ints,help='Layers with independent exact SVD audits')
    p.add_argument('--layers',type=parse_ints,help='Zero-based layers; default all')
    p.add_argument('--warmup-steps',type=int,default=16)
    p.add_argument('--eval-every',type=int,default=32)
    p.add_argument('--eval-batches',type=int,default=4)
    p.add_argument('--base-lr',type=float,default=1e-3)
    p.add_argument('--reference-width',type=int,default=512)
    p.add_argument('--reference-r',type=int,default=1)
    p.add_argument('--kv-multiplier',type=float,help='Frozen externally predicted KV LR multiplier, for rule=custom')
    p.add_argument('--prediction',type=Path,help='Frozen JSON from predict.py; requires one matching real-data target configuration')
    p.add_argument('--weight-decay',type=float,default=0.)
    p.add_argument('--epsilon0',type=float,default=1e-9)
    p.add_argument('--beta1',type=float,default=.9)
    p.add_argument('--beta2',type=float,default=.999)
    p.add_argument('--sigma0',type=float,default=1.)
    p.add_argument('--attention-scale',choices=['mup','standard'],default='mup')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--save-matrices',action='store_true')
    p.add_argument('--save-model',action='store_true')
    args=p.parse_args()
    args.rules=[x.strip() for x in args.rules.split(',')]
    if not args.rules or any(x not in ('vanilla','coherent','custom') for x in args.rules):
        p.error('--rules must use vanilla,coherent,custom')
    if args.prediction:
        frozen=json.loads(args.prediction.read_text())
        if args.synthetic or args.rules!=['custom'] or args.kv_multiplier is not None:
            p.error('--prediction requires real data, --rules custom, and no manual --kv-multiplier')
        if args.widths!=[frozen['target_width']] or args.repetitions!=[frozen['target_r']]:
            p.error('Target width/r must match the frozen artifact exactly')
        if args.reference_width!=frozen['reference_width'] or args.reference_r!=frozen['reference_r']:
            p.error('Reference width/r must match the frozen artifact')
        if frozen.get('mode')!='proxy_only_extrapolation':
            p.error('Unknown prediction artifact mode')
        args.kv_multiplier=float(frozen['common_kv_multiplier'])
    if 'custom' in args.rules and (args.kv_multiplier is None or not math.isfinite(args.kv_multiplier) or args.kv_multiplier<=0):
        p.error('custom requires --kv-multiplier from a frozen prediction')
    if args.synthetic and (args.train_data or args.val_data):
        p.error('--synthetic cannot be combined with real data paths')
    if not args.synthetic and (args.train_data is None or args.val_data is None):
        p.error('Provide both --train-data and --val-data, or explicitly use --synthetic')
    if args.train_data and args.train_data.resolve()==args.val_data.resolve():
        p.error('Train and validation files must be distinct')
    if any(x<1 for x in args.widths+args.repetitions+[args.depth,args.head_dim,args.vocab_size,args.seq_len,args.batch_size,args.steps,args.eval_every,args.eval_batches,args.reference_width,args.reference_r,args.threads]):
        p.error('Dimensions, counts and reference settings must be positive')
    if any(seed<0 for seed in args.seeds):
        p.error('Seeds must be nonnegative')
    if args.vocab_size<2 or args.warmup_steps<0 or args.warmup_steps>args.steps:
        p.error('Require vocab_size>=2 and 0<=warmup_steps<=steps')
    if any(n%args.head_dim or (n//args.head_dim)%r for n in args.widths for r in args.repetitions):
        p.error('Every repetition must divide each width/head_dim query-head count')
    if args.layers is not None and (not args.layers or min(args.layers)<0 or max(args.layers)>=args.depth):
        p.error('Selected layers must exist')
    if args.log_every<0 or args.norm_probes<1 or args.norm_max_power<0 or args.norm_tolerance<=0 or not 0<args.norm_failure_probability<1:
        p.error('Invalid logging/norm-estimation settings')
    if any(t<1 for t in args.checkpoints+args.exact_checkpoints):
        p.error('Checkpoints must be positive')
    if not (0<=args.beta1<1 and 0<=args.beta2<1 and args.epsilon0>0 and args.base_lr>0 and args.sigma0>0 and args.weight_decay>=0):
        p.error('Invalid optimizer or initialization parameters')
    for name in ['base_lr','sigma0','epsilon0','weight_decay','beta1','beta2']:
        if not math.isfinite(getattr(args,name)):
            p.error(f'{name} must be finite')
    combinations=list(itertools.product(args.widths,args.repetitions,args.seeds,args.rules))
    if len(combinations)!=len(set(combinations)):
        p.error('Grid contains duplicate configurations')
    return args,combinations


def measurement_steps(config):
    selected={t for t in config.checkpoints if 1<=t<=config.steps}|{config.steps}
    if config.progress_checkpoints:
        selected|={1,max(1,config.warmup_steps),max(1,math.ceil(config.steps*.25)),
                   max(1,math.ceil(config.steps*.5)),config.steps}
    if config.log_every:
        selected.update(range(config.log_every,config.steps+1,config.log_every))
    return selected|{t for t in config.exact_checkpoints if t<=config.steps}


def load_tokens(path,dtype):
    array=np.load(path,mmap_mode='r',allow_pickle=False) if path.suffix=='.npy' else np.memmap(path,dtype=dtype,mode='r')
    if array.ndim!=1 or not np.issubdtype(array.dtype,np.integer):
        raise ValueError('Token file must be a one-dimensional integer array')
    return array


def batch(tokens,generator,config,device):
    starts=generator.integers(0,len(tokens)-config.seq_len,size=config.batch_size)
    windows=np.stack([tokens[int(s):int(s)+config.seq_len+1] for s in starts]).astype(np.int64)
    if windows.min()<0 or windows.max()>=config.vocab_size:
        raise ValueError('Token IDs outside configured vocabulary')
    value=torch.from_numpy(windows).to(device)
    return value[:,:-1],value[:,1:]


def schedule(step,total,warmup):
    if warmup and step<=warmup:
        return step/warmup
    progress=(step-warmup)/max(1,total-warmup)
    return .1+.9*.5*(1+math.cos(math.pi*progress))


def path_metadata(path):
    if path is None:
        return None
    stat=path.stat()
    return {'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def train_one(config,n,r,seed,rule,train_tokens,val_tokens,device):
    run_id=f'n{n}_r{r}_seed{seed}_{rule}'
    directory=config.output/run_id
    directory.mkdir(parents=True,exist_ok=False)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model=TinyGQALM(width=n,depth=config.depth,head_dim=config.head_dim,repetition=r,
                    vocab_size=config.vocab_size,seq_len=config.seq_len,sigma0=config.sigma0,
                    attention_scale=config.attention_scale,init_seed=seed).to(device)
    optimizer=make_optimizer(model,width=n,repetition=r,rule=rule,base_lr=config.base_lr,
                             reference_width=config.reference_width,reference_r=config.reference_r,
                             weight_decay=config.weight_decay,epsilon0=config.epsilon0,
                             betas=(config.beta1,config.beta2),kv_multiplier=config.kv_multiplier)
    # Same seed produces exactly the same batches across widths/rules/repetitions.
    generator=np.random.default_rng(seed+10000)
    validation_seed=seed+20000
    metadata={'run_id':run_id,'seed':seed,'rule':rule,'data_mode':'synthetic' if config.synthetic else 'token_file',
              'depth':config.depth,'head_dim':config.head_dim,'batch_size':config.batch_size,
              'seq_len':config.seq_len,'attention_scale':config.attention_scale}
    saved={k:str(v) if isinstance(v,Path) else v for k,v in vars(config).items()}
    saved.update(metadata,width=n,repetition=r,device=str(device),torch_version=torch.__version__,
                 numpy_version=np.__version__,python_version=platform.python_version(),
                 parameter_count=sum(p.numel() for p in model.parameters()),
                 initialization_rng='parameter-local SHA256-derived CPU generator',
                 train_file=path_metadata(config.train_data),val_file=path_metadata(config.val_data),
                 frozen_prediction=json.loads(config.prediction.read_text()) if config.prediction else None)
    (directory/'config.json').write_text(json.dumps(saved,indent=2))
    checkpoints=measurement_steps(config)
    tokens_seen=0
    started=time.perf_counter()
    with GeometryCollector(optimizer,select_kv_projections(model,config.layers),directory,
                           metadata=metadata,save_matrices=config.save_matrices,
                           spectral_options=dict(method=config.norm_method,probes=config.norm_probes,max_power=config.norm_max_power,
                                                 relative_tolerance=config.norm_tolerance,failure_probability=config.norm_failure_probability,
                                                 seed=config.norm_seed,fallback_exact=config.norm_fallback_exact),
                           exact_check_steps=config.exact_checkpoints,exact_check_layers=config.exact_check_layers) as collector, (directory/'loss.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run_id','step','tokens_seen','train_loss','validation_loss','elapsed_seconds'])
        writer.writeheader()
        for step in range(1,config.steps+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            factor=schedule(step,config.steps,config.warmup_steps)
            for group in optimizer.param_groups:
                group['lr']=group['initial_lr']*factor
            x,y=batch(train_tokens,generator,config,device)
            loss=model(x,y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss at {run_id}/{step}')
            loss.backward()
            sample=step in checkpoints
            if sample:
                collector.before_step()
            optimizer.step()
            tokens_seen+=config.batch_size*config.seq_len
            train_loss=float(loss.detach().cpu())
            if sample:
                collector.after_step(global_step=step,tokens_seen=tokens_seen,loss=train_loss)
            val=None
            if step%config.eval_every==0 or step==config.steps:
                model.eval();values=[]
                val_generator=np.random.default_rng(validation_seed)
                with torch.no_grad():
                    for _ in range(config.eval_batches):
                        vx,vy=batch(val_tokens,val_generator,config,device)
                        values.append(float(model(vx,vy).cpu()))
                val=float(np.mean(values))
            writer.writerow(dict(run_id=run_id,step=step,tokens_seen=tokens_seen,train_loss=train_loss,
                                 validation_loss=val,elapsed_seconds=time.perf_counter()-started))
            handle.flush()
        if config.save_model:
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'config':saved},directory/'final.pt')
    (directory/'complete.json').write_text(json.dumps({'run_id':run_id,'steps':config.steps,'tokens_seen':tokens_seen,'validation_loss':val},indent=2))
    print(json.dumps({'completed':run_id,'validation_loss':val,'seconds':round(time.perf_counter()-started,2)}),flush=True)


def main():
    config,grid=arguments()
    torch.set_num_threads(config.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    if hasattr(torch.backends,'cudnn'):
        torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if config.device=='auto' and torch.cuda.is_available() else 'cpu' if config.device=='auto' else config.device)
    if config.synthetic:
        rng=np.random.default_rng(2027)
        count=max(8192,config.seq_len*32+1)
        train_tokens=rng.integers(0,config.vocab_size,size=count,dtype=np.int64)
        val_tokens=rng.integers(0,config.vocab_size,size=count,dtype=np.int64)
    else:
        train_tokens=load_tokens(config.train_data,config.token_dtype)
        val_tokens=load_tokens(config.val_data,config.token_dtype)
    if min(len(train_tokens),len(val_tokens))<=config.seq_len:
        raise ValueError('Token files require more than seq_len entries')
    for n,r,seed,rule in grid:
        train_one(config,n,r,seed,rule,train_tokens,val_tokens,device)


if __name__=='__main__':
    main()
