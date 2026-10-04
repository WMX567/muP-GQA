"""
    This file contains "implementations" of muP. There are different ways to implement
    the same paramaterization, and this file contains some examples. For example, we 
    may not want to initialize the parameters to be too small.
"""

# TODO: This is not the correct standard param implementation.
standard_param_impl = {
    'name':                     'SP',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 0.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    }, 
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0,
    },
    'attention_scale':          lambda d: 1 / d**(1/2),
    'depth_scale':              lambda L: 1.0,
}


mengxi_impl = {
    'name':                     'xLLM (muP) Mengxi Candidate KV Scaling',
    'embedding': {
        'init_std':             lambda m: 1.0,
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1.0/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'q_layer': {
        'init_std':             lambda m: 1.0 / m**(1.0/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'k_layer': {
        'init_std':             lambda m, r: r / m**(1.0/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m, r: 1.0 / r,
    },
    'v_layer': {
        'init_std':             lambda m, r: r / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m, r: 1.0 / r,
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m**(1.0/2),
        'lr_scale':             lambda m: 1.0,
        'wd_scale':             lambda m: 1.0,
        'output_multiplier':    lambda m: 1.0
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1.0 / d,
    'depth_scale':              lambda L: 1.0 / L,
}

kyle_impl = {
    'name':                     'xLLM (muP) Kyle Candidate KV Scaling',
    'embedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: m,
    },
    'hidden': {
        'init_std':             lambda m: 1.0 / m**(1/2),
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0
    },
    'kv_layer': {
        'init_std':             lambda m, r: (1 + r**(1/2)) / (2 * m**(1/2)),
        'lr_scale':             lambda m, r: 1 / m,
        'wd_scale':             lambda m, r: m,
        'output_multiplier':    lambda m, r: 2 / (1 + r**(1/2)),
    },
    'unembedding': {
        'init_std':             lambda m: 1.0 / m,
        'lr_scale':             lambda m: 1.0 / m,
        'wd_scale':             lambda m: m,
        'output_multiplier':    lambda m: 1.0,
    },
    'normalization': {
        'lr_scale':             lambda m: 1.0 / m,
    },
    'attention_scale':          lambda d: 1 / d,
    'depth_scale':              lambda L: 1.0 / L,
}


impl_dict = {
    'standard_param_impl': standard_param_impl,
    'mengxi_impl': mengxi_impl,
    'kyle_impl': kyle_impl,
}