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
        mean = (input**2).mean(dim=-1, keepdim=True)
        normed = input / torch.sqrt(mean + self.eps)
        return self.mup_multiplier * normed

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
        m = config.mup_multiplier
        n = config.n_embd // config.n_head
        r = config.n_head // config.n_kv_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        if 'q_layer' in self.impl:
            q_init_std = self.impl['q_layer']['init_std'](m)
            nn.init.normal_(self.c_q.weight, mean=0.0, std=q_init_std)
        
        # c_kv outputs 2 * (n_embd // n_kv_reps): [k, v]
        self.c_kv = nn.Linear(config.n_embd, 2 * config.n_embd // self.n_kv_reps, bias=config.bias)
        self.c_kv.kv = True

        if 'k_layer' in self.impl and 'v_layer' in self.impl:
            # Split c_kv weight for k, v
            k_dim = config.n_embd // self.n_kv_reps
            v_dim = config.n_embd // self.n_kv_reps
            k_init_std = self.impl['k_layer']['init_std'](m, r)
            v_init_std = self.impl['v_layer']['init_std'](m, r)
            # k weight initialization
            nn.init.normal_(self.c_kv.weight[:k_dim, :], mean=0.0, std=k_init_std)
            # v weight initialization
            nn.init.normal_(self.c_kv.weight[k_dim:k_dim+v_dim, :], mean=0.0, std=v_init_std)

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

        # muP multiplier for q, k, v layers
        m = config.mup_multiplier
        n = config.n_embd // config.n_head
        r = config.n_head // config.n_kv_head
        if 'q_layer' in self.impl:
            self.q_output_mult = self.impl['q_layer']['output_multiplier'](m)
        else:
            self.q_output_mult = self.impl['hidden']['output_multiplier'](m)
        if 'k_layer' in self.impl:
            self.k_output_mult = self.impl['k_layer']['output_multiplier'](m,r)
        else:
            self.k_output_mult = self.impl['hidden']['output_multiplier'](m)
        if 'v_layer' in self.impl:
            self.v_output_mult = self.impl['v_layer']['output_multiplier'](m,r)
        else:
            self.v_output_mult = self.impl['hidden']['output_multiplier'](m)


    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # --- Compute Q, K, V with independent muP scaling ---
        q = self.c_q(x)
        k, v = self.c_kv(x).split(self.n_embd // self.n_kv_reps, dim=2)

        # Apply muP scaling for q, k, v
        q = self.q_output_mult * q
        k = self.k_output_mult * k
        v = self.v_output_mult * v

        # Optional pre-layer normalization
        if self.q_prelayer_norm is not None:
            q = self.q_prelayer_norm(q)
        if self.k_prelayer_norm is not None:
            k = self.k_prelayer_norm(k)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_kv_head, C // self.n_head)
        v = v.view(B, T, self.n_kv_head, C // self.n_head)

        # Repeat kv heads as needed and transpose
        k = repeat_kv(k, self.n_kv_reps).transpose(1, 2)  # (B, nh, T, hs)
        v = repeat_kv(v, self.n_kv_reps).transpose(1, 2)  # (B, nh, T, hs)

        # --- Attention computation ---
        if self.flash:
            # Efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True,
                scale=self.impl['attention_scale'](k.size(-1))
            )
        else:
            # Manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * self.impl['attention_scale'](k.size(-1))  # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = self.kv_capture(y)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with muP scaling and dropout
        y = self.resid_dropout(self.fc_mult * self.c_proj(y))
        return y

class _Expert(nn.Module):
    """Single FFN expert: GELU(FC)->FC"""
    def __init__(self, config):
        super().__init__()
        h = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, h,  bias=config.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h,            config.n_embd, bias=config.bias)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

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

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.topk = max(1, min(config.router_topk, self.num_experts))
        in_dim = config.n_embd
        hidden = config.moe_ffn_hidden_size
        E = self.num_experts
        self.config = config

        self.w_gating = nn.Linear(in_dim, E, bias=False)
        self.router_bias = nn.Parameter(torch.zeros(E))
        self.null_expert_bias = config.moe_null_expert_bias

        self.fc1_weight = nn.Parameter(torch.randn(E, in_dim, hidden))
        self.fc1_bias = nn.Parameter(torch.zeros(E, hidden)) if config.bias else None
        self.fc2_weight = nn.Parameter(torch.randn(E, hidden, in_dim))
        self.fc2_bias = nn.Parameter(torch.zeros(E, in_dim)) if config.bias else None

        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def _forward_sparse(self, x, topk_idx, gates):
        B, T, C = x.shape
        k = topk_idx.size(2)

        # Memory-efficient, group-by-expert + batched matmul approach.
        device = x.device
        dtype = x.dtype

        # Flatten token-expert selections and corresponding inputs
        flat_idx = topk_idx.reshape(-1)                  # (N,) where N = B*T*k
        N = flat_idx.numel()
        x_exp = x.unsqueeze(2).expand(-1, -1, k, -1).reshape(N, C)  # (N, C)

        # Find unique experts and inverse mapping (maps each position -> 0..U-1)
        unique_experts, inverse = torch.unique(flat_idx, return_inverse=True)
        U = unique_experts.numel()
        if U == 0:
            return torch.zeros(B, T, k, C, device=device, dtype=dtype)

        # Sort positions by expert to make contiguous groups
        perm = torch.argsort(inverse)
        inverse_sorted = inverse[perm]
        x_sorted = x_exp[perm]  # (N, C)

        # Counts per expert and grouping boundaries
        counts = torch.bincount(inverse_sorted, minlength=U)
        max_count = int(counts.max().item())
        starts = torch.cat((torch.tensor([0], device=device, dtype=counts.dtype), counts.cumsum(0)[:-1]))

        # Pack inputs into (U, max_count, C) with zero-padding
        x_grouped = torch.zeros(U, max_count, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            x_grouped[i, :cnt] = x_sorted[s:s+cnt]

        # Gather per-expert parameters once (U is small)
        weight1_u = self.fc1_weight[unique_experts]  # (U, C, H)
        if self.fc1_bias is not None:
            bias1_u = self.fc1_bias[unique_experts]      # (U, H)

        # Batched matmul: (U, max_count, C) x (U, C, H) -> (U, max_count, H)
        h_grouped = torch.bmm(x_grouped, weight1_u)
        h_grouped *= self.config.impl['moe_ffn_1']['output_multiplier'](self.config.mup_multiplier, self.config.moe_ffn_mup_multiplier, self.config.num_experts, self.topk)
        if self.fc1_bias is not None:
            h_grouped.add_(bias1_u.unsqueeze(1))  # (U, max_count, H)

        # Unpack grouped outputs back to sorted-flat layout
        h_sorted_flat = torch.empty(N, h_grouped.size(-1), device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            h_sorted_flat[s:s+cnt] = h_grouped[i, :cnt]

        # Unpermute to original flat order and reshape to (B, T, k, H)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device)
        h_flat = h_sorted_flat[inv_perm]
        H = h_flat.size(-1)
        h = h_flat.view(B, T, k, H)

        # Activation + dropout
        h = self.act(h)
        h = self.dropout(h)

        # Second projection: follow same grouped-batched strategy
        # Flatten h for second projection
        h_exp = h.reshape(N, H)
        h_sorted = h_exp[perm]

        weight2_u = self.fc2_weight[unique_experts]  # (U, H, C)
        if self.fc2_bias is not None:
            bias2_u = self.fc2_bias[unique_experts]      # (U, C)

        # Pack h_sorted into (U, max_count, H)
        h_grouped2 = torch.zeros(U, max_count, H, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            h_grouped2[i, :cnt] = h_sorted[s:s+cnt]

        # Batched matmul: (U, max_count, H) x (U, H, C) -> (U, max_count, C)
        out_grouped = torch.bmm(h_grouped2, weight2_u)
        out_grouped *= self.config.impl['moe_ffn_2']['output_multiplier'](self.config.mup_multiplier, self.config.moe_ffn_mup_multiplier, self.config.num_experts, self.topk)
        if self.fc2_bias is not None:
            out_grouped.add_(bias2_u.unsqueeze(1))  # (U, max_count, C)

        # Unpack back to flat sorted -> unpermute -> reshape
        out_sorted_flat = torch.empty(N, C, device=device, dtype=dtype)
        for i in range(U):
            cnt = int(counts[i].item())
            if cnt == 0:
                continue
            s = int(starts[i].item())
            out_sorted_flat[s:s+cnt] = out_grouped[i, :cnt]

        out_flat = out_sorted_flat[inv_perm]
        out_view = out_flat.view(B, T, k, C)

        # Multiply per-position outputs by corresponding gate values and scatter-add into final output y
        y = torch.zeros(B, T, C, device=device, dtype=out_view.dtype)
        gate_vals = gates.gather(2, topk_idx)               # (B, T, k)
        y = (out_view * gate_vals.unsqueeze(-1)).sum(dim=2)  # (B, T, C)

        return y

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        device = x.device
        E = self.num_experts

        logits = self.w_gating(x) + self.router_bias.view(1, 1, E)
        print(f"Gating logits: mean {logits.mean().item():.4f}, std {logits.std().item():.4f}, min {logits.min().item():.4f}, max {logits.max().item():.4f}")
        gates = F.softmax(logits, dim=-1)  # (B, T, E)

        # Compute aux loss
        importance = gates.sum(dim=(0, 1))  # (E,)
        importance_mean = importance / (B * T)
        if self.config.mup:
            aux = (E**2 * (importance_mean ** 2).sum()).to(device)
        else:
            aux = (E * (importance_mean ** 2).sum()).to(device)

        if self.config.moe_random_router:
            logits = torch.rand_like(logits)
            gates = logits

        topk_vals, topk_idx = gates.topk(self.topk, dim=-1)  # (B,T,topk)
        if self.config.mup:
            topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + self.null_expert_bias)
        sparse_gates = torch.zeros_like(gates)
        sparse_gates.scatter_(2, topk_idx, topk_vals)
        gates = sparse_gates

        y = self._forward_sparse(x, topk_idx, gates)

        return y, aux


class Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.ln_1 = normalization(config)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = normalization(config)

        # choose ffn implementation (MLP or MoE)
        # first layer (layer_idx=0) is always dense MLP, rest can be MoE
        self.use_moe = getattr(config, 'use_moe', False) and (layer_idx != 0)
        if self.use_moe:
            self.ffn = MoE(config)
        else:
            self.ffn = MLP(config)

        self.depth_mult = config.impl['depth_scale'](config.n_layer) if hasattr(config, 'impl') and 'depth_scale' in config.impl else 1.0

    def forward(self, x):
        x = x + self.depth_mult * self.attn(self.ln_1(x))
        ffn_out = self.ffn(self.ln_2(x))
        if isinstance(ffn_out, tuple):
            # MoE returns (output, aux_loss)
            y, aux = ffn_out
        else:
            y, aux = ffn_out, 0.0
        return x + self.depth_mult * y, aux
    
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
    impl: dict = field(default_factory=lambda: standard_param_impl) # implementation details, see standard_param_impl and tpv_left_impl
    normalization: str = "LayerNorm"
    q_prelayer_normalization: str = 'None'
    k_prelayer_normalization: str = 'None'
    complete_p_layers: bool = False # if True, the depth multiplier is 1/n_layer, otherwise 1.0
    use_moe: bool = False
    router_topk: int = 8
    num_experts: int = 64
    moe_seq_aux_loss_coeff: float = 1e-4
    moe_ffn_hidden_size: int = 128
    moe_ffn_mup_multiplier: float = 1.0
    moe_null_expert_bias: float = 0.0
    moe_random_router: bool = False
    
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
            h = nn.ModuleList(reversed([Block(config, layer_idx=config.n_layer - 1 - i) for i in range(config.n_layer)])),
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
        self._init_weights(None)
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
        router_type = []
        router_bias_type = []
        moe_ffn_type_1 = []
        moe_ffn_type_2 = []
        layer_norms = []

        for block in reversed(self.transformer.h):
            hidden_type.append(block.attn.c_q.weight)
            kv_type.append(block.attn.c_kv.weight)
            hidden_type.append(block.attn.c_proj.weight)
            # handle different FFN types (MLP or MoE)
            if hasattr(block, 'ffn'):
                ffn = block.ffn
                if isinstance(ffn, MLP):
                    hidden_type.append(ffn.c_fc.weight)
                    hidden_type.append(ffn.c_proj.weight)
                else:
                    # include gating network params
                    if hasattr(ffn, 'w_gating'):
                        router_type.append(ffn.w_gating.weight)
                        if hasattr(ffn.w_gating, 'bias') and ffn.w_gating.bias is not None:
                            router_bias_type.append(ffn.w_gating.bias)
                    # router bias
                    if hasattr(ffn, 'router_bias'):
                        router_bias_type.append(ffn.router_bias)
                    # expert weights (stacked tensors)
                    if hasattr(ffn, 'fc1_weight'):
                        moe_ffn_type_1.append(ffn.fc1_weight)
                    if hasattr(ffn, 'fc1_bias') and ffn.fc1_bias is not None:
                        print(ffn.fc1_bias)
                        raise ValueError("FC biases are not supported in MoE implementation")
                    if hasattr(ffn, 'fc2_weight'):
                        moe_ffn_type_2.append(ffn.fc2_weight)
                    if hasattr(ffn, 'fc2_bias') and ffn.fc2_bias is not None:
                        print(ffn.fc2_bias)
                        raise ValueError("FC biases are not supported in MoE implementation")
            else:
                # backwards compatibility with attribute name 'mlp'
                if hasattr(block, 'mlp'):
                    hidden_type.append(block.mlp.c_fc.weight)
                    hidden_type.append(block.mlp.c_proj.weight)

        for n, p in self.named_parameters():
            if ("bias" in n) and (self.config.mup) and \
                (p is not None) and \
                ('router' not in n) and \
                ('gating' not in n):
                raise ValueError(f"Biases are not supported in {self.impl['name']} implementation, found {n}")

        if kv:
            return embedding_type, hidden_type, kv_type, unembedding_type, router_type, router_bias_type, moe_ffn_type_1, moe_ffn_type_2
        else:
            return embedding_type, hidden_type + kv_type, unembedding_type, router_type, router_bias_type, moe_ffn_type_1, moe_ffn_type_2
        
    def _init_weights(self, module):
        m = self.config.mup_multiplier
        n = self.config.n_embd // self.config.n_head
        r = self.config.n_head // self.config.n_kv_head

        n0 = self.config.n_embd / self.config.mup_multiplier
        et, ht, kv, ut, rt, rbt, mft1, mft2 = self._get_weight_groups(kv=True)
        for p in et:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['embedding']['init_std'](self.config.mup_multiplier))

        for p in ht:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](self.config.mup_multiplier))

        for p in kv:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['hidden']['init_std'](m))

        # Improved muP initialization for q, k, v weights
        for block in self.transformer.h:
            attn = block.attn
            # Query
            if hasattr(attn, 'c_q') and 'q_layer' in self.impl:
                q_std = self.config.init_std * self.impl['q_layer']['init_std'](m)
                torch.nn.init.normal_(attn.c_q.weight, mean=0.0, std=q_std)
            # Key/Value
            if hasattr(attn, 'c_kv') and 'k_layer' in self.impl and 'v_layer' in self.impl:
                k_dim = self.config.n_embd // attn.n_kv_reps
                v_dim = self.config.n_embd // attn.n_kv_reps
                k_std = self.config.init_std * self.impl['k_layer']['init_std'](m, r)
                v_std = self.config.init_std * self.impl['v_layer']['init_std'](m, r)
                torch.nn.init.normal_(attn.c_kv.weight[:k_dim, :], mean=0.0, std=k_std)
                torch.nn.init.normal_(attn.c_kv.weight[k_dim:k_dim+v_dim, :], mean=0.0, std=v_std)

        for p in ut:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['unembedding']['init_std'](self.config.mup_multiplier))

        for p in rt:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['router']['init_std'](self.config.n_embd, self.config.num_experts))

        for p in rbt:
            torch.nn.init.zeros_(p)

        for p in mft1:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['moe_ffn_1']['init_std'](self.config.n_embd, self.config.moe_ffn_hidden_size, self.config.num_experts, self.config.router_topk))
        
        for p in mft2:
            torch.nn.init.normal_(p, mean=0.0, std=self.config.init_std * self.impl['moe_ffn_2']['init_std'](self.config.n_embd, self.config.moe_ffn_hidden_size, self.config.num_experts, self.config.router_topk))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop( self.emb_mult * (tok_emb + pos_emb) )

        # iterate through blocks and accumulate MoE aux losses if present
        moe_aux = torch.tensor(0.0, device=device)
        for block in self.transformer.h:
            out = block(x)
            if isinstance(out, tuple):
                x, aux = out
                if isinstance(aux, torch.Tensor):
                    moe_aux = moe_aux + aux
                else:
                    moe_aux = moe_aux + torch.tensor(float(aux), device=device)
            else:
                x = out
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits_all = self.lm_mult * self.lm_head(x)
            lm_loss = F.cross_entropy(logits_all.view(-1, logits_all.size(-1)), targets.view(-1), ignore_index=-1)
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_mult * self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            # combine language loss with MoE aux loss if MoE is enabled
            if getattr(self.config, 'use_moe', False):
                loss = lm_loss + self.config.moe_seq_aux_loss_coeff * moe_aux
            else:
                loss = lm_loss
        else:
            logits = self.lm_mult * self.lm_head(x[:, [-1], :])
            loss = None

        if self.config.use_moe:
            return logits, loss, moe_aux, lm_loss
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
        
        # Check if we need separate q, k, v groups
        has_separate_qkv = 'q_layer' in self.impl or 'k_layer' in self.impl or 'v_layer' in self.impl
        
        if has_separate_qkv:
            # Use kv=True to get separated groups, but we'll manually handle qkv parameters
            embedding_type, hidden_type, _, unembedding_type, router_type, router_bias_type, moe_ffn_type1, moe_ffn_type2 = self._get_weight_groups(kv=True)
            
            # Remove q and kv weights from hidden_type since we'll handle them separately
            filtered_hidden_type = []
            qkv_params_set = set()
            
            # Collect all q, k, v parameters that we want to handle separately
            for block in self.transformer.h:
                attn = block.attn
                if hasattr(attn, 'c_q'):
                    qkv_params_set.add(id(attn.c_q.weight))
                if hasattr(attn, 'c_kv'):
                    qkv_params_set.add(id(attn.c_kv.weight))
            
            # Filter out qkv params from hidden_type
            for param in hidden_type:
                if id(param) not in qkv_params_set:
                    filtered_hidden_type.append(param)
         
            q_params, kv_params = [], []
            for block in self.transformer.h:
                attn = block.attn
                if hasattr(attn, 'c_q'):
                    q_params.append(attn.c_q.weight)
                if hasattr(attn, 'c_kv'):
                    # We must pass the full c_kv.weight tensor to optimizer (can't slice)
                    kv_params.append(attn.c_kv.weight)
        else:
            # Use regular grouping without separate qkv
            embedding_type, hidden_type, unembedding_type, router_type, router_bias_type, moe_ffn_type1, moe_ffn_type2 = self._get_weight_groups(kv=False)
            filtered_hidden_type = hidden_type

        et_group = {}
        et_group['params'] = embedding_type
        et_group['lr_scale'] = self.impl['embedding']['lr_scale'](self.config.mup_multiplier)
        et_group['wd_scale'] = self.impl['embedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(et_group)

        ht_group = {}
        ht_group['params'] = filtered_hidden_type
        ht_group['lr_scale'] = self.impl['hidden']['lr_scale'](self.config.mup_multiplier)
        ht_group['wd_scale'] = self.impl['hidden']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ht_group)

        # --- muP-compliant optimizer groups for q, k, v ---
        if has_separate_qkv:
            # q_layer
            if 'q_layer' in self.impl and q_params:
                q_group = {'params': q_params,
                           'lr_scale': self.impl['q_layer']['lr_scale'](self.config.mup_multiplier),
                           'wd_scale': self.impl['q_layer']['wd_scale'](self.config.mup_multiplier)}
                optim_groups.append(q_group)
            # kv_layer (combined k and v since we can't split the tensor for optimizer)
            if ('k_layer' in self.impl or 'v_layer' in self.impl) and kv_params:
                # Use k_layer scaling if available, otherwise fall back to hidden scaling
                if 'k_layer' in self.impl:
                    kv_group = {'params': kv_params,
                               'lr_scale': self.impl['k_layer']['lr_scale'](self.config.mup_multiplier),
                               'wd_scale': self.impl['k_layer']['wd_scale'](self.config.mup_multiplier)}
                else:
                    # If only v_layer is defined, use v_layer scaling
                    kv_group = {'params': kv_params,
                               'lr_scale': self.impl['v_layer']['lr_scale'](self.config.mup_multiplier),
                               'wd_scale': self.impl['v_layer']['wd_scale'](self.config.mup_multiplier)}
                optim_groups.append(kv_group)

        ut_group = {}
        ut_group['params'] = unembedding_type
        ut_group['lr_scale'] = self.impl['unembedding']['lr_scale'](self.config.mup_multiplier)
        ut_group['wd_scale'] = self.impl['unembedding']['wd_scale'](self.config.mup_multiplier)
        optim_groups.append(ut_group)

        rt_group = {}
        if len(router_type) > 0:
            rt_group['params'] = router_type
            if 'router' in self.impl:
                router_group_dict = self.impl['router']
                rt_group['lr_scale'] = router_group_dict['lr_scale'](self.config.n_embd, self.config.num_experts)
                rt_group['wd_scale'] = router_group_dict['wd_scale'](self.config.n_embd, self.config.num_experts)
            else:
                router_group_dict = self.impl['hidden']
                rt_group['lr_scale'] = router_group_dict['lr_scale'](self.config.mup_multiplier)
                rt_group['wd_scale'] = router_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(rt_group)

        rbt_group = {}
        if len(router_bias_type) > 0:
            rbt_group['params'] = router_bias_type
            if 'router' in self.impl:
                router_bias_group_dict = self.impl['router_bias']
                rbt_group['lr_scale'] = router_bias_group_dict['lr_scale'](self.config.mup_multiplier, self.config.num_experts)
                rbt_group['wd_scale'] = router_bias_group_dict['wd_scale'](self.config.mup_multiplier, self.config.num_experts) 
            else:
                router_bias_group_dict = self.impl['hidden']    
                rbt_group['lr_scale'] = router_bias_group_dict['lr_scale'](self.config.mup_multiplier)
                rbt_group['wd_scale'] = router_bias_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(rbt_group)

        mft_group_1 = {}
        if len(moe_ffn_type1) > 0:
            mft_group_1['params'] = moe_ffn_type1
            if 'moe_ffn_1' in self.impl:
                print("using moe_ffn_1 param impl")
                moe_ffn_group_dict = self.impl['moe_ffn_1']
                mft_group_1['lr_scale'] = moe_ffn_group_dict['lr_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
                mft_group_1['wd_scale'] = moe_ffn_group_dict['wd_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
            else:
                moe_ffn_group_dict = self.impl['hidden']
                mft_group_1['lr_scale'] = moe_ffn_group_dict['lr_scale'](self.config.mup_multiplier)
                mft_group_1['wd_scale'] = moe_ffn_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(mft_group_1)

        mft_group_2 = {}
        if len(moe_ffn_type2) > 0:
            mft_group_2['params'] = moe_ffn_type2
            if 'moe_ffn_2' in self.impl:
                print(f"Using separate moe_ffn_2 param group with impl {self.impl['name']}")
                moe_ffn_group_dict = self.impl['moe_ffn_2']
                mft_group_2['lr_scale'] = moe_ffn_group_dict['lr_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
                mft_group_2['wd_scale'] = moe_ffn_group_dict['wd_scale'](
                    self.config.mup_multiplier, 
                    self.config.moe_ffn_mup_multiplier,
                    self.config.num_experts,
                    self.config.router_topk
                )
            else:
                moe_ffn_group_dict = self.impl['hidden']
                mft_group_2['lr_scale'] = moe_ffn_group_dict['lr_scale'](self.config.mup_multiplier)
                mft_group_2['wd_scale'] = moe_ffn_group_dict['wd_scale'](self.config.mup_multiplier)
            optim_groups.append(mft_group_2)

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
            # optimizer = torch.optim.SGD(optim_groups, lr=learning_rate, weight_decay=weight_decay, momentum=0.9, nesterov=True)

        for group in optimizer.param_groups:
            group['weight_decay'] = group['wd_scale'] * group['weight_decay']
            group['lr'] = group['lr_scale'] * group['lr']

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        This version provides a different estimate when using a Mixture-of-Experts (MoE):
        - For non-MoE models we keep the original heuristic.
        - For MoE models we replace the standard FFN cost with the cost of executing
          `router_topk` experts per token (each expert does two linear projections).
        """
        # basic sizes
        N = self.get_num_params()
        cfg = self.config
        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        T = cfg.block_size

        # original coarse PaLM-style heuristic
        flops_per_token_orig = 6 * N + 12 * L * H * Q * T

        use_moe = getattr(cfg, 'use_moe', False)
        if not use_moe:
            flops_per_fwdbwd = flops_per_token_orig * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        else:
            # Estimate attention + other non-FFN work by subtracting a heuristic
            # standard FFN cost from the original term, then add MoE cost.
            # Standard MLP hidden size is typically 4 * n_embd.
            n_embd = cfg.n_embd
            std_ffn_hidden = 4 * n_embd
            # heuristic standard FFN flops per token per layer (two matmuls: n_embd->hidden and hidden->n_embd)
            std_ffn_flops_per_token_per_layer = 2 * n_embd * std_ffn_hidden
            std_ffn_total = L * std_ffn_flops_per_token_per_layer

            # approximate the non-FFN component contained in the original heuristic
            attention_and_other = max(flops_per_token_orig - std_ffn_total, 0.0)

            # MoE cost: each selected expert does two matmuls of sizes (n_embd x moe_hidden) and (moe_hidden x n_embd)
            topk = max(1, min(getattr(cfg, 'router_topk', 1), getattr(cfg, 'num_experts', 1)))
            moe_hidden = getattr(cfg, 'moe_ffn_hidden_size', std_ffn_hidden)
            moe_ffn_flops_per_token_per_layer = topk * (2 * n_embd * moe_hidden)
            moe_ffn_total = L * moe_ffn_flops_per_token_per_layer

            # final per-token cost: non-FFN work + MoE-executed FFN work + parameter-related term (6*N)
            flops_per_token = 6 * N + attention_and_other + moe_ffn_total
            flops_per_fwdbwd = flops_per_token * T
            flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # express our flops throughput as ratio of H200 bf16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # H200 bf16 peak FLOPS
        flops_promised = 1979e12
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
            # sample from 