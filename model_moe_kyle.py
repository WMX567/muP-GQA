"""
GQA GPT language model with configurable muP scaling.
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


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, config):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.n_embd))
        self.bias = nn.Parameter(torch.zeros(config.n_embd)) if config.bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

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
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0, f"Expected config.n_head {config.n_head} to be divisible by config.n_kv_head {config.n_kv_head}"
        self.impl = config.impl
        self.n_kv_head = config.n_kv_head
        self.n_kv_reps = config.n_head // self.n_kv_head
        m = config.mup_multiplier
        r = self.n_kv_reps

        # The constructor-time inits below are overwritten by GPT._init_weights, but they
        # advance the RNG and therefore determine the final initialization.
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        if 'q_layer' in self.impl:
            nn.init.normal_(self.c_q.weight, mean=0.0, std=self.impl['q_layer']['init_std'](m))

        # c_kv outputs 2 * (n_embd // n_kv_reps): [k, v]
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd // self.n_kv_reps, bias=config.bias)
        if 'k_layer' in self.impl and 'v_layer' in self.impl:
            kv_dim = config.n_embd // self.n_kv_reps
            nn.init.normal_(self.c_kv.weight[:kv_dim, :], mean=0.0, std=self.impl['k_layer']['init_std'](m, r))
            nn.init.normal_(self.c_kv.weight[kv_dim:, :], mean=0.0, std=self.impl['v_layer']['init_std'](m, r))

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.fc_mult = self.impl['hidden']['output_multiplier'](m)

        # muP multiplier for q, k, v layers
        hidden_mult = self.impl['hidden']['output_multiplier'](m)
        self.q_output_mult = self.impl['q_layer']['output_multiplier'](m) if 'q_layer' in self.impl else hidden_mult
        self.k_output_mult = self.impl['k_layer']['output_multiplier'](m, r) if 'k_layer' in self.impl else hidden_mult
        self.v_output_mult = self.impl['v_layer']['output_multiplier'](m, r) if 'v_layer' in self.impl else hidden_mult

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # --- Compute Q, K, V with independent muP scaling ---
        q = self.q_output_mult * self.c_q(x)
        k, v = self.c_kv(x).split(self.n_embd // self.n_kv_reps, dim=2)
        k = self.k_output_mult * k
        v = self.v_output_mult * v

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, C // self.n_head)
        v = v.view(B, T, self.n_kv_head, C // self.n_head)

        # Repeat kv heads as needed and transpose
        k = repeat_kv(k, self.n_kv_reps).transpose(1, 2)  # (B, nh, T, hs)
        v = repeat_kv(v, self.n_kv_reps).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
            scale=self.impl['attention_scale'](k.size(-1))
        )

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with muP scaling and dropout
        y = self.resid_dropout(self.fc_mult * self.c_proj(y))
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
        self.ln_1 = LayerNorm(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config)
        self.ffn = MLP(config)
        self.depth_mult = config.impl['depth_scale'](config.n_layer)

    def forward(self, x):
        x = x + self.depth_mult * self.attn(self.ln_1(x))
        return x + self.depth_mult * self.ffn(self.ln_2(x))

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
    impl: dict = field(default_factory=lambda: standard_param_impl) # see mup_implementations.py

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.impl = config.impl

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(reversed([Block(config) for _ in range(config.n_layer)])),
            ln_f = LayerNorm(config),
        ))
        # Untied input and output embeddings.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.emb_mult = self.impl['embedding']['output_multiplier'](config.mup_multiplier)
        self.lm_mult = self.impl['unembedding']['output_multiplier'](config.mup_multiplier)

        self._init_weights(None)

        # report number of parameters
        print("Total parameters: %.2fM" % (self.get_num_params(non_embedding=False)/1e6,))
        print("Total non-embedding parameters: %.2fM" % (self.get_num_params(non_embedding=True)/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings and the untied
        output embedding get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            n_params -= self.lm_head.weight.numel()
        return n_params

    def _get_weight_groups(self):
        embedding_type = [self.transformer.wte.weight,
                          self.transformer.wpe.weight]
        hidden_type = []
        kv_type = []
        unembedding_type = [self.lm_head.weight]

        for block in reversed(self.transformer.h):
            hidden_type.append(block.attn.c_q.weight)
            kv_type.append(block.attn.c_kv.weight)
            hidden_type.append(block.attn.c_proj.weight)
            hidden_type.append(block.ffn.c_fc.weight)
            hidden_type.append(block.ffn.c_proj.weight)

        for n, p in self.named_parameters():
            if "bias" in n and self.config.mup:
                raise ValueError(f"Biases are not supported in {self.impl['name']} implementation, found {n}")

        return embedding_type, hidden_type, kv_type, unembedding_type

    def _init_weights(self, module):
        m = self.config.mup_multiplier
        r = self.config.n_head // self.config.n_kv_head
        et, ht, kv, ut = self._get_weight_groups()
        for p in et:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['embedding']['init_std'](m))

        for p in ht + kv:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](m))

        # Separate muP initialization for q, k, v weights
        for block in self.transformer.h:
            attn = block.attn
            if 'q_layer' in self.impl:
                q_std = self.config.init_std * self.impl['q_layer']['init_std'](m)
                torch.nn.init.normal_(attn.c_q.weight, mean=0.0, std=q_std)
            if 'k_layer' in self.impl and 'v_layer' in self.impl:
                kv_dim = self.config.n_embd // attn.n_kv_reps
                k_std = self.config.init_std * self.impl['k_layer']['init_std'](m, r)
                v_std = self.config.init_std * self.impl['v_layer']['init_std'](m, r)
                torch.nn.init.normal_(attn.c_kv.weight[:kv_dim, :], mean=0.0, std=k_std)
                torch.nn.init.normal_(attn.c_kv.weight[kv_dim:2*kv_dim, :], mean=0.0, std=v_std)

        for p in ut:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['unembedding']['init_std'](m))

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

        if targets is None:
            return self.lm_mult * self.lm_head(x[:, [-1], :]), None
        logits = self.lm_mult * self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, eps):
        m = self.config.mup_multiplier
        embedding_type, hidden_type, kv_type, unembedding_type = self._get_weight_groups()

        def make_group(params, key, *args):
            return {'params': params,
                    'lr_scale': self.impl[key]['lr_scale'](m, *args),
                    'wd_scale': self.impl[key]['wd_scale'](m, *args)}

        optim_groups = [make_group(embedding_type, 'embedding')]
        if 'q_layer' in self.impl:
            # muP-compliant groups for q and kv (k scaling applies to the whole c_kv tensor)
            q_params = [block.attn.c_q.weight for block in self.transformer.h]
            kv_params = [block.attn.c_kv.weight for block in self.transformer.h]
            q_ids = {id(p) for p in q_params}
            optim_groups.append(make_group([p for p in hidden_type if id(p) not in q_ids], 'hidden'))
            optim_groups.append(make_group(q_params, 'q_layer'))
            optim_groups.append(make_group(kv_params, 'k_layer'))
        else:
            optim_groups.append(make_group(hidden_type + kv_type, 'hidden'))
        optim_groups.append(make_group(unembedding_type, 'unembedding'))
        optim_groups.append({
            'params': [p for n, p in self.named_parameters() if 'ln_' in n],
            'weight_decay': 0.0,  # no weight decay for layer norms
            'lr_scale': self.impl['normalization']['lr_scale'](m),
            'wd_scale': 1.0,
        })

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in optimizer.param_groups:
            group['weight_decay'] = group['wd_scale'] * group['weight_decay']
            group['lr'] = group['lr_scale'] * group['lr']

        return optimizer
