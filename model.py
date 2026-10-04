"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.nn import functional as F

from mup_implementations import standard_param_impl

class Capture(nn.Module):
    def __init__(_):
        super().__init__()
    def forward(_, x: torch.Tensor):
        return x

def normalization(config):
    if config.normalization == "RMSNorm":
        return RMSNorm(config)
    elif config.normalization == "LayerNorm":
        return LayerNorm(config)
    else:
        raise ValueError(f"Unknown normalization type: {config.normalization}. Supported: 'RMSNorm', 'LayerNorm'")

class L2Norm(nn.Module):
    def __init__(self, config, eps=1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (x.norm(dim=-1, keepdim=True) + self.eps)

class RMSNorm(nn.Module):
    def __init__(self, config, eps=1e-3):
        super().__init__()
        ndim = config.n_embd
        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1
        self.weight = nn.Parameter(
            torch.ones(ndim) / self.mup_multiplier
        )
        self.eps = eps

    def forward(self, input):
        mean = (input**2).mean(dim=-1, keepdim=True)
        normed = input / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * self.weight * normed


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config, bias=None, shape=None):
        super().__init__()
        print(f"Initialized a Layer Norm", flush=True)
        ndim = config.n_embd
        if bias is None:
            bias = config.bias if hasattr(config, 'bias') else False

        self.mup_multiplier = config.mup_multiplier if hasattr(config, 'mup_multiplier') else 1

        # self.weight = nn.Parameter(
        #     torch.ones(ndim) / self.mup_multiplier
        # )
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(
            input, 
            self.weight.shape, 
            # self.mup_multiplier * self.weight, 
            self.weight,
            self.bias, 1e-5)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Lifted from xLLM: Credit Max Ma
    bsz, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, :, None, :].expand(bsz, slen, n_kv_heads, n_rep, head_dim)
    x = x.reshape(bsz, slen, n_kv_heads * n_rep, head_dim)
    return x

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, f"Expected config.n_head {config.n_head} to be divisible by config.n_kv_head {config.n_kv_head}"
        self.impl = config.impl
        self.n_kv_head = config.n_kv_head

        self.n_kv_reps = config.n_head // self.n_kv_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # n_embd / n_kv_reps = n_kv_head * head_size
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd // self.n_kv_reps, bias=config.bias)
        self.c_kv.kv = True

        self.kv_capture = Capture()

        self.q_prelayer_norm = None
        print(f"Config: q prelayer_norm: {config.q_prelayer_normalization}, k prelayer_norm: {config.k_prelayer_normalization}")
        if config.q_prelayer_normalization == 'LayerNorm':
            self.q_prelayer_norm = LayerNorm(config)
        elif config.q_prelayer_normalization == 'L2Norm':
            self.q_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.q_prelayer_norm = RMSNorm(config)
        elif config.q_prelayer_normalization == 'LayerNormWithBias':
            self.q_prelayer_norm = LayerNorm(config, bias=True)

        self.k_prelayer_norm = None
        if config.k_prelayer_normalization == 'LayerNorm':
            self.k_prelayer_norm = LayerNorm(config)
        elif config.k_prelayer_normalization == 'L2Norm':
            self.k_prelayer_norm = L2Norm(config)
        elif config.k_prelayer_normalization == 'L2NormScale':
            self.k_prelayer_norm = RMSNorm(config)
        elif config.k_prelayer_normalization == 'LayerNormWithBias':
            self.k_prelayer_norm = LayerNorm(config, bias=True)
            

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

        self.fc_mult = self.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.fc_mult * self.c_q(x)
        k, v = ( self.fc_mult * self.c_kv(x) ).split(self.n_embd // self.n_kv_reps, dim=2)

        if self.q_prelayer_norm is not None:
            q = self.q_prelayer_norm(q)
        if self.k_prelayer_norm is not None:
            k = self.k_prelayer_norm(k)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        k = k.view(B, T, self.n_kv_head, C // self.n_head) # (B, T, n_kvh, hs)
        v = v.view(B, T, self.n_kv_head, C // self.n_head) # (B, T, n_kvh, hs)

        k = repeat_kv(k, self.n_kv_reps).transpose(1, 2) # (B, nh, T, hs)
        v = repeat_kv(v, self.n_kv_reps).transpose(1, 2) # (B, nh, T, hs)

        if self.config.n_head != self.n_kv_head:
            head_size = self.config.n_embd // self.config.n_head
            H = head_size * self.config.n_kv_head
            m = self.config.mup_multiplier

            # k = k * (m**(1/2) / H**(1/2))
            # v = v * (m**(1/2) / H**(1/2))

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
                scale=self.impl['attention_scale'](k.size(-1))
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.impl['attention_scale'](k.size(-1)) # (B, nh, T, T)
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))

            att = att

            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

            y = self.kv_capture(y)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = y 

        # output projection
        y = self.resid_dropout( self.fc_mult * self.c_proj(y) )
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.fc_mult = config.impl['hidden']['output_multiplier'](config.mup_multiplier)

    def forward(self, x):
        x = self.fc_mult * self.c_fc(x)
        x = self.gelu(x)
        x = self.fc_mult * self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = normalization(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = normalization(config)
        self.mlp = MLP(config)

        if config.mup or config.complete_p_layers:
            self.depth_mult = 1.0 / config.n_layer
        else:
            self.depth_mult = 1.

    def forward(self, x):
        x = x + self.depth_mult * self.attn(self.ln_1(x))
        x = x + self.depth_mult * self.mlp(self.ln_2(x))
        return x

@dataclass                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 4
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    mup: bool = False
    mup_multiplier: float = 1.0
    init_std: float = 0.02
    impl: dict = field(default_factory=standard_param_impl) # implementation details, see standard_param_impl and tpv_left_impl
    normalization: str = "LayerNorm"
    q_prelayer_normalization: str = 'None'
    k_prelayer_normalization: str = 'None'
    complete_p_layers: bool = False # if True, the depth multiplier is 1/n_layer, otherwise 1.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.impl = config.impl if hasattr(config, 'impl') else standard_param_impl

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(reversed([Block(config) for _ in range(config.n_layer)])),
            ln_f = normalization(config),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.emb_mult = self.impl['embedding']['output_multiplier'](config.mup_multiplier)
        self.lm_mult = self.impl['unembedding']['output_multiplier'](config.mup_multiplier)

        # KYLE: Untie final layer weights
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # self._init_weights(None)
    
        # report number of parameters
        print("Total parameters: %.2fM" % (self.get_num_params(non_embedding=False)/1e6,))
        print("Total non-embedding parameters: %.2fM" % (self.get_num_params(non_embedding=True)/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.lm_head.weight.numel() 
        return n_params

    def _get_weight_groups(self, kv=False):
        # TODO: NO BIAS
        embedding_type = [self.transformer.wte.weight, 
                          self.transformer.wpe.weight]
        hidden_type = []
        kv_type = []
        unembedding_type = [self.lm_head.weight]
        layer_norms = []

        for block in reversed(self.transformer.h):
            hidden_type.append(block.attn.c_q.weight)
            kv_type.append(block.attn.c_kv.weight)
            hidden_type.append(block.attn.c_proj.weight)
            hidden_type.append(block.mlp.c_fc.weight)
            hidden_type.append(block.mlp.c_proj.weight)

        for n, p in self.named_parameters():
            if "bias" in n and self.config.mup:
                raise ValueError(f"Biases are not supported in {self.impl['name']} implementation, found {n}")

        if kv:
            return embedding_type, hidden_type, kv_type, unembedding_type
        else:
            return embedding_type, hidden_type + kv_type, unembedding_type
        
    def _init_weights(self, module):
        if not self.config.mup:
            return
            
        n0 = self.config.n_embd / self.config.mup_multiplier
        et, ht, kv, ut = self._get_weight_groups(kv=True)
        for p in et:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['embedding']['init_std'](self.config.mup_multiplier))

        for p in kv:
            target_mult = 1.0

            if self.config.n_head // self.config.n_kv_head > 1:
                # if we have kv heads we have to fiddle with the initialization distribution
                r = self.config.n_head // self.config.n_kv_head
                head_size = self.config.n_embd // self.config.n_head
                H = self.config.n_kv_head * head_size
                m = self.config.mup_multiplier
                n = self.config.n_embd
                n0 = n / m
                # target_mult = m**(1/2) * H**(1/2)**(m/n)**(1/2) / ( m**(1/2)*(m**(1/2) + H**(1/2)*(m/n)**(1/2)) )
                # target_mult = m**(1/2) / (r**(1/2)*(n**(1/2) + H**(1/2)))*(n0**(1/2) + H**(1/2))
                target_mult = m**(1/2) * ( 1 + (H / n0)**(1/2) ) / ( r**(1/2) * (m**(1/2) + (H / n0)**(1/2)) )
                # target_mult = r**(-1/2)
                # target_mult = m**(1/2) / (r**(1/2) * (n**(1/2) + H**(1/2)))
                # target_mult = H**(1/2) / ( m**(1/2) + H**(1/2) / n0**(1/2) )
                # target_mult = m**(1/2) * H**(1/2) * 1 / ( r**(1/2) * (n**(1/2) + H**(1/2)) )
                target_mult = H**(1/2) / r**(1/2)
                target_mult = m**(1/2) / ( r**(1/2) * ( m**(1/2) + (H / n0)**(1/2) ) )
                # target_mult = H**(1/2) / ( r**(1/2) * (m**(1/2) + (H / n0)**(1/2)) )
                # target_mult = 1 / ( r**(1/2) * (m**(1/2) + (H / n0)**(1/2)) )
                target_mult = m**(1/2) / ( r**(1/2) * ( m**(1/2) + (H / n0)**(1/2) ) )
                target_mult = m**(1/2) / ( r**(1/2) * ( n**(1/2) + (H)**(1/2) ) )
            
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier) * target_mult)

        for p in ht:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))

        for p in ut:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['unembedding']['init_std'](self.config.mup_multiplier))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop( self.emb_mult * (tok_emb + pos_emb) )
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_mult * self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_mult * self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:  
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps, device_type, adaptive_optimizer=False):
        # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = False # fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optim_groups = []
        embedding_type, hidden_type, kv_type, unembedding_type = self._get_weight_groups(kv=True)

        et_group = {}
        et_group['params'] = embedding_type
        et_group['lr_scale'] = self.impl['embedding']['lr_scale'](self.config.mup_multiplier)
        et_group['wd_scale'] = self.impl['embedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(et_group)

        ht_group = {}
        ht_group['params'] = hidden_type
        ht_group['lr_scale'] = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
        ht_group['wd_scale'] = self.impl['hidden']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ht_group)

        kv_group = {}
        kv_group['params'] = kv_type
        if self.config.n_head != self.config.n_kv_head:
            H = self.config.n_kv_head * (self.config.n_embd // self.config.n_head)
            r = self.config.n_head // self.config.n_kv_head
            kv_group['lr_scale'] = 1/(self.config.mup_multiplier**(1/2) * H**(1/2)) / r**(1/2)
            kv_group['wd_scale'] = (self.config.mup_multiplier**(1/2) * H**(1/2)) * r**(1/2)
        else:
            kv_group['lr_scale'] = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
            kv_group['wd_scale'] = self.impl['hidden']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(kv_group)

        ut_group = {}
        ut_group['params'] = unembedding_type
        ut_group['lr_scale'] = self.impl['unembedding']['lr_scale'](self.config.mup_multiplier)
        ut_group['wd_scale'] = self.impl['unembedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ut_group)

        layer_norms = [p for n, p in self.named_parameters() if 'ln_' in n]
        layer_norms_group = {
            'params': layer_norms,
            'weight_decay': 0.0,  # no weight decay for layer norms
            'lr_scale': self.impl['normalization']['lr_scale'](self.config.mup_multiplier),      # no scaling for layer norms
            'wd_scale': 1.0       # no scaling for layer norms
        }
        optim_groups.append(layer_norms_group)

        if adaptive_optimizer:
            # use the OWDA optimizer, which is an AdamW variant with dynamic weight decay
            from oawda import OWDAAdam
            print("Using OAWDA optimizer")
            optimizer = OWDAAdam(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
        else:
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, **extra_args)
            print(f"using fused AdamW: {use_fused}")

        for group in optimizer.param_groups:
            group['weight_decay'] = group['wd_scale'] * group['weight_decay'] * weight_decay
            group['lr'] = group['lr_scale'] * group['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx