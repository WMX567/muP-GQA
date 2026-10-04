import ast
from contextlib import redirect_stderr
import io
import importlib.util
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mu_transfer import get_args, normalized_config
from run_mu_transfer import legacy_audit, status, training_args
from mu_transfer_io import load_metrics, metrics_array, save_array, load_legacy


class SweepTests(unittest.TestCase):
    def test_boolean_and_geometry(self):
        args = get_args(['--decay_lr=False', '--compile=False'])
        self.assertFalse(args.decay_lr)
        self.assertFalse(args.compile)
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            get_args(['--n_embd', '513'])

    def test_legacy_grid_complete(self):
        rows = legacy_audit(ROOT/'mup')
        self.assertEqual(len(rows), 72)
        self.assertTrue(all(row['status']=='legacy_complete' for row in rows))
        self.assertTrue(all(Path(row['path']).suffix=='.npy' for row in rows))
        self.assertFalse(list((ROOT/'mup').glob('*.csv')))
        self.assertEqual(sum(len(load_legacy(Path(row['path']))) for row in rows),17946)

    def test_completion_requires_matching_config_and_finite_final_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config = {'max_iters': 2}
            task = {'config': config, 'out_dir': str(out)}
            expected = normalized_config(get_args(training_args(config, out)))
            (out/'config.json').write_text(json.dumps(expected))
            (out/'checkpoint.pt').touch()
            (out/'completed.json').write_text(json.dumps({'protocol_version':2, 'step':2, 'val_loss':1.0}))
            def write_metrics(step, val):
                save_array(out/'metrics.npy', metrics_array([dict(step=step, train_loss=1.2, val_loss=val, lr=.001, tokens=72, seconds=.1)]))
            write_metrics(2, 1.0)
            self.assertEqual(status(task), 'complete')
            write_metrics(1, 1.0)
            self.assertEqual(status(task), 'incomplete')
            write_metrics(2, float('nan'))
            self.assertEqual(status(task), 'incomplete')
            expected['n_embd'] = 1024
            (out/'config.json').write_text(json.dumps(expected))
            self.assertEqual(status(task), 'config_mismatch')

    def test_numpy_metrics_roundtrip_and_csv_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metrics.npy'
            rows=[dict(step=1, train_loss=2., val_loss=None, lr=.001, tokens=2**55+1, seconds=.2)]
            save_array(path, metrics_array(rows))
            stored=np.load(path, allow_pickle=False)
            self.assertEqual(stored['tokens'][0], 2**55+1)
            self.assertTrue(np.isnan(stored['val_loss'][0]))
            self.assertEqual(stored.dtype['step'].kind, 'i')
            path.unlink()
            path.with_suffix('.csv').write_text('step,train_loss,val_loss,lr,tokens,seconds\n1,2.,,.001,32,.2\n')
            loaded=load_metrics(path)
            self.assertEqual(loaded['step'][0], 1)
            self.assertTrue(np.isnan(loaded['val_loss'][0]))

    def test_numpy_plot_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for seed,loss in [(0,1.), (1,2.), (2,3.)]:
                out=root/f'seed{seed}';out.mkdir()
                config=normalized_config(get_args(['--max_iters','2','--seed',str(seed)]))
                (out/'config.json').write_text(json.dumps(config))
                (out/'checkpoint.pt').touch()
                (out/'completed.json').write_text(json.dumps({'protocol_version':2,'step':2,'val_loss':loss}))
                save_array(out/'metrics.npy',metrics_array([dict(step=2,train_loss=1.,val_loss=loss,lr=.00195,tokens=72,seconds=.1)]))
            subprocess.run([sys.executable,str(ROOT/'mu_transfer_plot.py'),'--results-dir',str(root),
                            '--output',str(root/'summary.png'),'--summary-only'],check=True,capture_output=True)
            summary=np.load(root/'summary.npy',allow_pickle=False)
            self.assertEqual(len(summary),1)
            self.assertEqual(summary['n_seeds'][0],3)
            self.assertAlmostEqual(summary['val_loss'][0],2.)
            self.assertAlmostEqual(summary['val_std'][0],1.)
            self.assertFalse((root/'summary.csv').exists())

    def test_default_manifest_and_slurm_script(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable,str(ROOT/'run_mu_transfer.py'),'--mode','generate',
                            '--scripts-dir',directory],check=True,capture_output=True)
            files=list(Path(directory).glob('tasks_*.json'))
            tasks=json.loads(files[0].read_text())
            self.assertEqual(len(tasks),72)
            self.assertEqual({t['config']['impl'] for t in tasks},{'mengxi_impl'})
            self.assertEqual({t['config']['param_type'] for t in tasks},{'mup'})
            self.assertEqual({t['config']['weight_decay'] for t in tasks},{0.0})
            self.assertEqual(len({t['out_dir'] for t in tasks}),72)
            script=next(Path(directory).glob('submit_*.sh'))
            subprocess.run(['bash','-n',str(script)],check=True)
            self.assertIn('--array=0-71%4',script.read_text())

    def test_initialization_applies_qkv_scaling_after_hidden_scaling(self):
        # Execute the actual model method with an instrumented init function.
        tree=ast.parse((ROOT/'model_moe_kyle.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPT')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_init_weights')
        recorded={}
        def normal_(parameter, mean, std): recorded[parameter]=std
        fake_torch=types.SimpleNamespace(nn=types.SimpleNamespace(init=types.SimpleNamespace(normal_=normal_,zeros_=lambda p:None)))
        namespace={'torch':fake_torch}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'model_init','exec'),namespace)
        from mup_implementations import mengxi_impl
        attn=types.SimpleNamespace(c_q=types.SimpleNamespace(weight='q'),c_kv=types.SimpleNamespace(weight=Sliceable()),n_kv_reps=2)
        model=types.SimpleNamespace(config=types.SimpleNamespace(mup_multiplier=4,n_embd=2048,n_head=16,n_kv_head=8,init_std=.02),
                                   impl=mengxi_impl,transformer=types.SimpleNamespace(h=[types.SimpleNamespace(attn=attn)]),
                                   _get_weight_groups=lambda kv: (['emb'],['q','hidden'],['kv'],['out'],[],[],[],[]))
        namespace['_init_weights'](model,None)
        self.assertAlmostEqual(recorded['q'], .01)
        self.assertAlmostEqual(recorded['hidden'], .01)
        self.assertAlmostEqual(recorded['slice_0_1024'], .02)
        self.assertAlmostEqual(recorded['slice_1024_2048'], .02)


class Sliceable:
    def __getitem__(self,key):
        part=key[0]
        return f'slice_{part.start or 0}_{part.stop}'


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class TrainingTests(unittest.TestCase):
    def test_tiny_cpu_mup_sp_and_resume(self):
        import numpy as np
        import torch
        from model_moe_kyle import GPT,GPTConfig
        from mup_implementations import mengxi_impl
        model=GPT(GPTConfig(n_layer=1,n_head=2,n_kv_head=1,n_embd=16,block_size=8,vocab_size=16,
                            bias=False,mup=True,mup_multiplier=2,impl=mengxi_impl))
        optimizer=model.configure_optimizers(.2,.001,(.9,.95),1e-8,'cpu')
        for group in optimizer.param_groups:
            if group['weight_decay']:
                self.assertAlmostEqual(group['weight_decay'],.2*group['wd_scale'])
        self.assertEqual({id(p) for p in model.parameters() if p.requires_grad},
                         {id(p) for g in optimizer.param_groups for p in g['params']})
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'data';data.mkdir()
            for split in ['train','val']:
                (np.arange(128)%16).astype(np.uint16).tofile(data/f'{split}.bin')
            with (data/'meta.pkl').open('wb') as f:pickle.dump({'vocab_size':16},f)
            for param in ['mup','sp']:
                out=root/param
                command=[sys.executable,str(ROOT/'mu_transfer.py'),'--data_dir',str(data),'--out_dir',str(out),
                         '--device','cpu','--dtype','float32','--n_embd','16','--n_head','2','--n_kv_head','1',
                         '--n_layer','1','--block_size','8','--batch_size','1','--gradient_accumulation_steps','2',
                         '--max_iters','2','--eval_iters','1','--checkpoint_interval','1','--param_type',param]
                subprocess.run(command,check=True,capture_output=True)
                subprocess.run(command+['--resume'],check=True,capture_output=True)
                rows=load_metrics(out/'metrics.npy')
                self.assertFalse((out/'metrics.csv').exists())
                self.assertEqual(int(rows[-1]['step']),2)
                self.assertEqual(int(rows[-1]['tokens']),32)
                self.assertEqual(len(rows),1)
                self.assertEqual(json.loads((out/'completed.json').read_text())['step'],2)


if __name__=='__main__':
    unittest.main()
