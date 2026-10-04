"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:

    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]

        if key in globals():
              
            if key in ["slurm_job_id", "slurm_array_task_id"]:
                try:
                    attempt = int(attempt)
                except Exception as e:
                    print(f"[configurator.py] Could not force {key} to int before assertion: {attempt} ({e})")
            else:       
                try:
                    if type(globals()[key]) == int or type(globals()[key]) == float or type(globals()[key]) == str:
                        attempt = type(globals()[key])(val)
                    else:
                        attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    attempt = str(val)

            print(f"[configurator.py] Overriding key: {key}")
            print(f"  Original type: {type(globals()[key])}, value: {globals()[key]}")
            print(f"  Attempted type: {type(attempt)}, value: {attempt}")
            assert type(attempt) == type(globals()[key]), f"Type mismatch for {key}: {type(attempt)} != {type(globals()[key])}. " \
                    f"Please ensure you are passing the correct type for {key}. Key value is: {attempt}"
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt

        else:
            raise ValueError(f"Unknown config key: {key}")

