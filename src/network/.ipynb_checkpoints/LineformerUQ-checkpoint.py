import math
import warnings
from abc import abstractmethod

import torch as th
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Init helpers
# ============================================================

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with th.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def zero_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


# ============================================================
# Timestep embedding
# ============================================================

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ============================================================
# Base interfaces
# ============================================================

class TimestepBlock(nn.Module):
    """
    Any module whose forward accepts (x, emb).
    """
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Like the diffusion U-Net version: passes emb only to modules that need it.
    """
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


# ============================================================
# Small helpers
# ============================================================

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def ray_partition(x, line_size):
    """
    x: [N, C]
    returns: [N // line_size, line_size, C]
    """
    n, c = x.shape
    if n % line_size != 0:
        raise ValueError(f"Input length {n} must be divisible by line_size={line_size}.")
    return x.view(n // line_size, line_size, c)


def ray_merge(x):
    """
    x: [B_lines, line_size, C]
    returns: [B_lines * line_size, C]
    """
    b_lines, line_size, c = x.shape
    return x.reshape(b_lines * line_size, c)


# ============================================================
# Core line attention
# ============================================================

class LineAttention(nn.Module):
    def __init__(
        self,
        dim,
        line_size=24,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        inner_dim = dim_head * heads

        self.pos_emb = nn.Parameter(th.empty(1, heads, line_size, line_size))
        trunc_normal_(self.pos_emb)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        """
        x: [N, C]
        returns: [N, C]
        """
        x_inp = ray_partition(x, self.line_size)  # [B_lines, L, C]

        q = self.to_q(x_inp)                      # [B_lines, L, H*D]
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)

        def split_heads(t):
            b, l, hd = t.shape
            return t.view(b, l, self.heads, self.dim_head).permute(0, 2, 1, 3).contiguous()

        q = split_heads(q)  # [B_lines, H, L, D]
        k = split_heads(k)
        v = split_heads(v)

        q = q * self.scale

        sim = th.einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        out = th.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(x_inp.shape[0], x_inp.shape[1], -1)
        out = self.to_out(out)

        return ray_merge(out)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner, bias=False),
            GELU(),
            nn.Linear(inner, inner, bias=False),
            GELU(),
            nn.Linear(inner, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Timestep-conditioned residual wrappers
# ============================================================

class TimeConditionedResidualBlock(TimestepBlock):
    """
    Residual wrapper around an arbitrary function fn: [N, C] -> [N, C],
    with timestep conditioning similar to diffusion ResBlock.

    Structure:
        h = norm(x)
        h = h * (1 + scale) + shift     if use_scale_shift_norm
            OR
        h = h + emb_out                 otherwise
        h = fn(h)
        return x + h
    """
    def __init__(
        self,
        dim,
        fn,
        emb_channels,
        dropout=0.0,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.use_scale_shift_norm = use_scale_shift_norm

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * dim if use_scale_shift_norm else dim,
            ),
        )

        self.out_layers = nn.Sequential(
            nn.Identity(),  # placeholder to mirror U-Net structure conceptually
            nn.Dropout(dropout),
        )

    def forward(self, x, emb):
        """
        x:   [N, C]
        emb: [N, emb_channels] OR [B_lines, emb_channels] that has already been expanded
        """
        h = self.norm(x)
        emb_out = self.emb_layers(emb).type(h.dtype)

        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=-1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out

        h = self.fn(h)
        h = self.out_layers(h)
        return x + h


# ============================================================
# Timestep-conditioned line-attention block
# ============================================================

class LineAttentionBlock(TimestepBlock):
    def __init__(
        self,
        dim,
        emb_channels,
        line_size=24,
        dim_head=32,
        heads=8,
        num_blocks=1,
        ffn_mult=4,
        dropout=0.0,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        for _ in range(num_blocks):
            attn = LineAttention(
                dim=dim,
                line_size=line_size,
                dim_head=dim_head,
                heads=heads,
            )
            ff = FFN(dim=dim, mult=ffn_mult)

            self.blocks.append(
                nn.ModuleList(
                    [
                        TimeConditionedResidualBlock(
                            dim=dim,
                            fn=attn,
                            emb_channels=emb_channels,
                            dropout=dropout,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ),
                        TimeConditionedResidualBlock(
                            dim=dim,
                            fn=ff,
                            emb_channels=emb_channels,
                            dropout=dropout,
                            use_scale_shift_norm=use_scale_shift_norm,
                        ),
                    ]
                )
            )

    def forward(self, x, emb):
        for attn_block, ff_block in self.blocks:
            x = attn_block(x, emb)
            x = ff_block(x, emb)
        return x


# ============================================================
# Main model without encoder
# ============================================================

class TimeConditionedLineformerNoEncoder(nn.Module):
    def __init__(
        self,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=(4,),
        out_dim=1,
        last_activation="sigmoid",
        line_size=32,
        dim_head=32,
        heads=8,
        num_blocks=1,
        input_dim=32,
        time_embed_dim=None,
        dropout=0.0,
        use_scale_shift_norm=True,
        ffn_mult=4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = set(skips)
        self.bound = bound
        self.in_dim = input_dim

        if time_embed_dim is None:
            time_embed_dim = hidden_dim * 4
        self.time_embed_dim = time_embed_dim

        # same style as diffusion U-Net
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        # input projection
        self.layers.append(nn.Linear(self.in_dim, hidden_dim))
        self.activations.append(nn.LeakyReLU())

        # hidden trunk
        for i in range(1, num_layers - 1):
            if i in self.skips:
                self.layers.append(nn.Linear(hidden_dim + self.in_dim, hidden_dim))
            else:
                self.layers.append(
                    LineAttentionBlock(
                        dim=hidden_dim,
                        emb_channels=time_embed_dim,
                        line_size=line_size,
                        dim_head=dim_head,
                        heads=heads,
                        num_blocks=num_blocks,
                        ffn_mult=ffn_mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
            self.activations.append(nn.LeakyReLU())

        # output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "identity":
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError(f"Unknown last_activation: {last_activation}")

    def forward(self, x, timesteps):
        """
        x: [N_rays * N_samples, input_dim]
        timesteps:
            either [N_rays] or [N_rays * N_samples]

        If timesteps is [N_rays], it will be repeated per sample automatically
        only if N is divisible accordingly and line_size-style grouping is consistent.
        """
        n = x.shape[0]

        # Make timestep batch match x batch.
        if timesteps.dim() != 1:
            timesteps = timesteps.view(-1)

        if timesteps.shape[0] != n:
            raise ValueError(
                f"timesteps must have shape [{n}] to match x, but got {tuple(timesteps.shape)}. "
                "Pass one timestep per point after flattening."
            )

        emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))

        input_pts = x

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            if i in self.skips:
                x = th.cat([input_pts, x], dim=-1)

            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)

            x = activation(x)

        return x


# ============================================================
# Main model with encoder
# ============================================================

class ConditionedLineformer(nn.Module):
    def __init__(
        self,
        encoder,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=(4,),
        out_dim=1,
        last_activation="sigmoid",
        line_size=16,
        dim_head=32,
        heads=8,
        num_blocks=1,
        cond_embed_dim=None,
        dropout=0.0,
        use_scale_shift_norm=True,
        ffn_mult=4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = set(skips)
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        if cond_embed_dim is None:
            cond_embed_dim = hidden_dim * 4
            
        self.cond_embed_dim = cond_embed_dim

        self.cond_embed = nn.Sequential(
            nn.Linear(hidden_dim, cond_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, cond_embed_dim),
        )

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.layers.append(nn.Linear(self.in_dim, hidden_dim))
        self.activations.append(nn.LeakyReLU())

        for i in range(1, num_layers - 1):
            if i in self.skips:
                self.layers.append(nn.Linear(hidden_dim + self.in_dim, hidden_dim))
            else:
                self.layers.append(
                    LineAttentionBlock(
                        dim=hidden_dim,
                        emb_channels=cond_embed_dim,
                        line_size=line_size,
                        dim_head=dim_head,
                        heads=heads,
                        num_blocks=num_blocks,
                        ffn_mult=ffn_mult,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
            self.activations.append(nn.LeakyReLU())

        self.layers.append(nn.Linear(hidden_dim, out_dim))
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "identity":
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError(f"Unknown last_activation: {last_activation}")

    def forward(self, x, condition):
        """
        x: [N_rays * N_samples, coord_dim]
        condition: [N_rays * N_samples]
        """
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        n = x.shape[0]
        if condition.dim() != 1:
            condition = condition.view(-1)
        if condition.shape[0] != n:
            raise ValueError(
                f"condition must have shape [{n}] to match encoded x, but got {tuple(timesteps.shape)}."
            )

        emb = self.cond_embed(timestep_embedding(condition, self.hidden_dim))

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            if i in self.skips:
                x = th.cat([input_pts, x], dim=-1)

            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)

            x = activation(x)

        return x