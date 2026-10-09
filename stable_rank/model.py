"""Small decoder with separate GQA projections for controlled geometry audits."""
from __future__ import annotations
import math
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class GQAAttention(nn.Module):
    def __init__(self, width, head_dim, repetition, attention_scale):
        super().__init__()
        if width % head_dim or (width//head_dim) % repetition:
            raise ValueError('r must divide the query head count, and width must divide into heads')
        self.head_dim = head_dim
        self.query_heads = width//head_dim
        self.kv_heads = self.query_heads//repetition
        self.repetition = repetition
        self.scale = 1/head_dim if attention_scale == 'mup' else 1/math.sqrt(head_dim)
        self.q_proj = nn.Linear(width,width,bias=False)
        self.k_proj = nn.Linear(width,width//repetition,bias=False)
        self.v_proj = nn.Linear(width,width//repetition,bias=False)
        self.o_proj = nn.Linear(width,width,bias=False)

    def forward(self, x):
        batch,length,_ = x.shape
        q = self.q_proj(x).reshape(batch,length,self.query_heads,self.head_dim).transpose(1,2)
        k = self.k_proj(x).reshape(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        v = self.v_proj(x).reshape(batch,length,self.kv_heads,self.head_dim).transpose(1,2)
        # Explicit repetition works on CPU and GPU, without experimental GQA kernels.
        k = k.repeat_interleave(self.repetition, dim=1)
        v = v.repeat_interleave(self.repetition, dim=1)
        output = F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.,scale=self.scale)
        return self.o_proj(output.transpose(1,2).contiguous().reshape(batch,length,-1))


class Block(nn.Module):
    def __init__(self, width, head_dim, repetition, attention_scale):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = GQAAttention(width,head_dim,repetition,attention_scale)
        self.ln2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width,4*width,bias=False),nn.GELU(),nn.Linear(4*width,width,bias=False))

    def forward(self,x):
        x = x+self.attn(self.ln1(x))
        return x+self.ffn(self.ln2(x))


class TinyGQALM(nn.Module):
    def __init__(self, *, width, depth, head_dim, repetition, vocab_size, seq_len, sigma0=1., attention_scale='mup', init_seed=None):
        super().__init__()
        self.width,self.seq_len = width,seq_len
        self.embedding = nn.Embedding(vocab_size,width)
        self.position = nn.Embedding(seq_len,width)
        self.blocks = nn.ModuleList([Block(width,head_dim,repetition,attention_scale) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(width)
        self.unembedding = nn.Linear(width,vocab_size,bias=False)
        # Parameter-local RNG keeps equal-shaped shared layers identical across r.
        base_seed=torch.initial_seed() if init_seed is None else init_seed
        for name,module in self.named_modules():
            generator=torch.Generator(device='cpu')
            salt=int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],'little')
            generator.manual_seed((base_seed+salt)%(2**63-1))
            if isinstance(module,nn.Linear):
                std=sigma0 if name=='unembedding' else sigma0/math.sqrt(module.in_features)
                nn.init.normal_(module.weight,std=std,generator=generator)
            elif isinstance(module,nn.Embedding):
                nn.init.normal_(module.weight,std=sigma0,generator=generator)

    def forward(self,tokens,targets=None):
        length = tokens.shape[1]
        if length>self.seq_len:
            raise ValueError('Sequence exceeds configured positional embeddings')
        x = self.embedding(tokens)+self.position(torch.arange(length,device=tokens.device))
        for block in self.blocks:
            x=block(x)
        logits=self.unembedding(self.final_norm(x))/self.width
        if targets is None:
            return logits
        return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1))


def make_optimizer(model, *, width, repetition, rule, base_lr, reference_width, reference_r,
                   weight_decay, epsilon0, betas, kv_multiplier=None):
    if rule not in ('vanilla','coherent','custom'):
        raise ValueError(rule)
    if rule=='custom' and (kv_multiplier is None or kv_multiplier<=0 or not math.isfinite(kv_multiplier)):
        raise ValueError('custom requires a frozen positive --kv-multiplier')
    groups=[]
    for name,p in model.named_parameters():
        if name.startswith(('embedding.','position.','unembedding.')) or p.ndim<2:
            a=1.
        else:
            a=reference_width/width
        is_kv=name.endswith(('k_proj.weight','v_proj.weight'))
        if is_kv and rule=='coherent':
            a *= (1+math.sqrt(repetition))/(1+math.sqrt(reference_r))
        if is_kv and rule=='custom':
            a=kv_multiplier
        groups.append(dict(params=[p],lr=base_lr*a,initial_lr=base_lr*a,
                           weight_decay=weight_decay/a if p.ndim>=2 else 0.,
                           eps=epsilon0*reference_width/width if p.ndim>=2 and not name.startswith(('embedding.','position.','unembedding.')) else epsilon0,
                           parameter_name=name))
    return torch.optim.AdamW(groups,betas=betas,foreach=False,fused=False)
