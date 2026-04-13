# uq_lineformer.py

import math
import warnings
from typing import Dict, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


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

    with torch.no_grad():
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


# ============================================================
# Original helpers
# ============================================================

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def ray_partition(x, line_size):
    """
    x: [N_ray * N_samples, C]
    returns: [N_ray * N_samples // line_size, line_size, C]
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
    return x.view(b_lines * line_size, c)


# ============================================================
# Original LineAttention stack
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
        self.scale = dim_head ** -0.5
        self.line_size = line_size

        seq_l = line_size
        self.pos_emb = nn.Parameter(torch.Tensor(1, heads, seq_l, seq_l))
        trunc_normal_(self.pos_emb)

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        """
        x: [N, C]
        returns: [N, C]
        """
        x_inp = ray_partition(x, self.line_size)

        q = self.to_q(x_inp)
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)

        q, k, v = map(
            lambda t: t.contiguous()
            .view(t.shape[0], t.shape[1], self.heads, t.shape[2] // self.heads)
            .permute(0, 2, 1, 3),
            (q, k, v),
        )

        q *= self.scale
        sim = einsum("b h i d, b h j d -> b h i j", q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(out.shape[0], out.shape[2], -1)
        out = self.to_out(out)
        out = ray_merge(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim * mult, bias=False),
            GELU(),
            nn.Linear(dim * mult, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class Line_Attention_Blcok(nn.Module):
    def __init__(
        self,
        dim,
        line_size=24,
        dim_head=32,
        heads=8,
        num_blocks=1,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            LineAttention(
                                dim=dim,
                                line_size=line_size,
                                dim_head=dim_head,
                                heads=heads,
                            ),
                        ),
                        PreNorm(dim, FFN(dim=dim)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        return x


# ============================================================
# Quantile embedding
# ============================================================

def scalar_condition_embedding(vals, dim, max_period=10000):
    """
    vals: [N] scalar conditions, here quantiles in (0, 1)
    returns: [N, dim]
    """
    if vals.dim() != 1:
        vals = vals.view(-1)

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=vals.device)
        / max(half, 1)
    )
    args = vals[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


# ============================================================
# UQ / Quantile-conditioned models
# ============================================================

class UQLineformerNoEncoder(nn.Module):
    """
    Quantile-conditioned version of Lineformer_no_encoder.

    Design goal:
    - preserve original backbone structure
    - allow clean loading from deterministic checkpoint where possible
    - inject quantile conditioning additively into hidden states
    """

    def __init__(
        self,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=(4,),
        out_dim=1,
        last_activation="identity",
        line_size=32,
        dim_head=32,
        heads=8,
        num_blocks=1,
        input_dim=32,
        quantile_embed_dim=None,
    ):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = list(skips)
        self.bound = bound
        self.in_dim = input_dim

        if quantile_embed_dim is None:
            quantile_embed_dim = hidden_dim

        self.quantile_embed = nn.Sequential(
            nn.Linear(hidden_dim, quantile_embed_dim),
            nn.SiLU(),
            nn.Linear(quantile_embed_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)]
            + [
                Line_Attention_Blcok(
                    dim=hidden_dim,
                    line_size=line_size,
                    dim_head=dim_head,
                    heads=heads,
                    num_blocks=num_blocks,
                )
                if i not in self.skips
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                for i in range(1, num_layers - 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.activations = nn.ModuleList(
            [nn.LeakyReLU() for _ in range(0, num_layers - 1)]
        )
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "identity":
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError(f"Unknown last activation: {last_activation}")

    def forward(self, x, quantiles):
        """
        x: [N, input_dim]
        quantiles: [N] in (0, 1)
        """
        input_pts = x

        quantiles = quantiles.view(-1)
        if quantiles.shape[0] != x.shape[0]:
            raise ValueError(
                f"quantiles must have shape [{x.shape[0]}], got {tuple(quantiles.shape)}"
            )

        q_emb = self.quantile_embed(
            scalar_condition_embedding(quantiles, self.hidden_dim)
        )

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], dim=-1)

            x = layer(x)

            # Inject conditioning only into hidden layers, not final output layer.
            if i < len(self.layers) - 1 and x.shape[-1] == self.hidden_dim:
                x = x + q_emb

            x = activation(x)

        return x


class UQLineformer(nn.Module):
    """
    Quantile-conditioned version of the original encoder-based Lineformer.
    This is the one you likely want for loading your pretrained checkpoint.
    """

    def __init__(
        self,
        encoder,
        bound=0.2,
        num_layers=8,
        hidden_dim=256,
        skips=(4,),
        out_dim=1,
        last_activation="identity",
        line_size=16,
        dim_head=32,
        heads=8,
        num_blocks=1,
        quantile_embed_dim=None,
    ):
        super().__init__()
        self.nunm_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = list(skips)
        self.bound = bound
        self.encoder = encoder
        self.in_dim = encoder.output_dim

        if quantile_embed_dim is None:
            quantile_embed_dim = hidden_dim

        self.quantile_embed = nn.Sequential(
            nn.Linear(hidden_dim, quantile_embed_dim),
            nn.SiLU(),
            nn.Linear(quantile_embed_dim, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [nn.Linear(self.in_dim, hidden_dim)]
            + [
                Line_Attention_Blcok(
                    dim=hidden_dim,
                    line_size=line_size,
                    dim_head=dim_head,
                    heads=heads,
                    num_blocks=num_blocks,
                )
                if i not in self.skips
                else nn.Linear(hidden_dim + self.in_dim, hidden_dim)
                for i in range(1, num_layers - 1)
            ]
        )
        self.layers.append(nn.Linear(hidden_dim, out_dim))

        self.activations = nn.ModuleList(
            [nn.LeakyReLU() for _ in range(0, num_layers - 1)]
        )
        if last_activation == "sigmoid":
            self.activations.append(nn.Sigmoid())
        elif last_activation == "relu":
            self.activations.append(nn.LeakyReLU())
        elif last_activation == "identity":
            self.activations.append(nn.Identity())
        else:
            raise NotImplementedError(f"Unknown last activation: {last_activation}")

    def forward(self, x, quantiles):
        """
        x: [N_rays * N_samples, coord_dim]
        quantiles: [N_rays * N_samples] in (0, 1)
        """
        x = self.encoder(x, self.bound)
        input_pts = x[..., :self.in_dim]

        quantiles = quantiles.view(-1)
        if quantiles.shape[0] != x.shape[0]:
            raise ValueError(
                f"quantiles must have shape [{x.shape[0]}], got {tuple(quantiles.shape)}"
            )

        q_emb = self.quantile_embed(
            scalar_condition_embedding(quantiles, self.hidden_dim)
        )

        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = self.activations[i]

            if i in self.skips:
                x = torch.cat([input_pts, x], dim=-1)

            x = layer(x)

            if i < len(self.layers) - 1 and x.shape[-1] == self.hidden_dim:
                x = x + q_emb

            x = activation(x)

        return x


# ============================================================
# Losses + inference helpers
# ============================================================

def pinball_loss(pred, target, quantiles):
    """
    pred: [N, ...]
    target: [N, ...]
    quantiles: [N]
    """
    if pred.shape != target.shape:
        raise ValueError(f"pred shape {pred.shape} != target shape {target.shape}")

    q = quantiles
    while q.ndim < pred.ndim:
        q = q.unsqueeze(-1)

    err = target - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


@torch.no_grad()
def predict_quantiles(model, x, quantile_values):
    """
    quantile_values: iterable like [0.05, 0.5, 0.95]
    returns dict[q] = prediction
    """
    preds = {}
    n = x.shape[0]
    device = x.device

    for qv in quantile_values:
        q = torch.full((n,), float(qv), device=device, dtype=torch.float32)
        preds[float(qv)] = model(x, q)

    return preds


# ============================================================
# Pretrained loading
# ============================================================

def load_pretrained_lineformer_weights(
    uq_model: nn.Module,
    ckpt_path: str,
    strict: bool = False,
    key: str = "network",
    verbose: bool = True,
) -> Tuple[Iterable[str], Iterable[str]]:
    """
    Loads deterministic LineFormer weights into the UQ model.
    Works cleanly when the UQ model preserves the original backbone names.

    Expected behavior:
    - backbone weights load
    - quantile_embed.* remain missing
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt[key] if key in ckpt else ckpt

    missing, unexpected = uq_model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"Loaded checkpoint: {ckpt_path}")
        print("\nMissing keys:")
        for k in missing:
            print("  ", k)
        print("\nUnexpected keys:")
        for k in unexpected:
            print("  ", k)

    return missing, unexpected


def freeze_backbone_except_quantile(uq_model: nn.Module):
    """
    Optional warm start:
    freeze everything except the quantile embedding MLP.
    """
    for name, p in uq_model.named_parameters():
        p.requires_grad = name.startswith("quantile_embed.")


def unfreeze_all(uq_model: nn.Module):
    for p in uq_model.parameters():
        p.requires_grad = True


# ============================================================
# Example training-step utility
# ============================================================

def calculate_uq_output_and_loss(
    model,
    x,
    target,
    mode="random",
    quantiles_fixed=(0.05, 0.5, 0.95),
):
    """
    x:      [N, C]
    target: [N, 1] or [N, ...]
    mode:
      - "random": sample one random quantile per point
      - "fixed": average pinball loss over fixed quantiles
    """
    device = x.device
    n = x.shape[0]

    if mode == "random":
        curr_quantiles = torch.rand(n, device=device, dtype=torch.float32)
        curr_quantiles = curr_quantiles.clamp_(1e-7, 1.0 - 1e-7)
        pred = model(x, curr_quantiles)
        loss = pinball_loss(pred, target, curr_quantiles)
        return pred, loss

    if mode == "fixed":
        preds = []
        losses = []
        for qv in quantiles_fixed:
            q = torch.full((n,), float(qv), device=device, dtype=torch.float32)
            pred = model(x, q)
            preds.append(pred)
            losses.append(pinball_loss(pred, target, q))
        loss = sum(losses) / len(losses)
        return preds, loss

    raise ValueError(f"Unknown mode: {mode}")


# ============================================================
# Minimal example
# ============================================================

if __name__ == "__main__":
    # Demo only for no-encoder version.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UQLineformerNoEncoder(
        num_layers=4,
        hidden_dim=32,
        skips=(2,),
        out_dim=1,
        last_activation="identity",
        bound=0.3,
        input_dim=32,
        line_size=32,
        dim_head=32,
        heads=8,
        num_blocks=1,
    ).to(device)

    x = torch.randn((1024 * 192, 32), device=device)
    y = torch.randn((1024 * 192, 1), device=device)

    pred, loss = calculate_uq_output_and_loss(model, x, y, mode="random")
    print("pred shape:", pred.shape)
    print("loss:", float(loss))