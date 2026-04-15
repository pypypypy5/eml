from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GrokkingConfig:
    p: int = 113
    d_model: int = 128
    d_mlp: int = 512
    n_heads: int = 4
    d_head: int = 32
    n_ctx: int = 3


class OneLayerModularTransformer(nn.Module):
    """Minimal TransformerLens-compatible forward pass for the grokking model.

    The public checkpoint used in the notebooks is a single-layer causal
    transformer with no layer norm. This class intentionally implements only the
    pieces needed to replay that exact checkpoint.
    """

    def __init__(self, cfg: GrokkingConfig):
        super().__init__()
        self.cfg = cfg
        vocab = cfg.p + 1

        # This public checkpoint uses the older ARENA/TransformerLens layout:
        # embedding and unembedding are stored as [d_model, d_vocab].
        self.W_E = nn.Parameter(torch.empty(cfg.d_model, vocab))
        self.W_pos = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))

        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        self.W_O = nn.Parameter(torch.empty(cfg.d_model, cfg.n_heads * cfg.d_head))

        self.W_in = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        self.b_in = nn.Parameter(torch.empty(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        self.b_out = nn.Parameter(torch.empty(cfg.d_model))

        self.W_U = nn.Parameter(torch.empty(cfg.d_model, vocab))

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> "OneLayerModularTransformer":
        cfg = infer_config(state_dict)
        model = cls(cfg)
        remapped = remap_transformer_lens_state_dict(state_dict)
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint load mismatch: missing={missing}, unexpected={unexpected}")
        model.eval()
        return model

    def forward(self, tokens: torch.Tensor, return_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cfg = self.cfg
        resid = self.W_E.T[tokens] + self.W_pos[: tokens.shape[1]]

        q = torch.einsum("hfd,bpd->bphf", self.W_Q, resid)
        k = torch.einsum("hfd,bpd->bphf", self.W_K, resid)
        v = torch.einsum("hfd,bpd->bphf", self.W_V, resid)
        attn_scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (cfg.d_head**0.5)

        pos = tokens.shape[1]
        causal_mask = torch.triu(torch.ones(pos, pos, dtype=torch.bool, device=tokens.device), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1.0e9)
        pattern = F.softmax(attn_scores, dim=-1)
        z = torch.einsum("bhqk,bkhd->bqhd", pattern, v)
        z_flat = z.reshape(z.shape[0], z.shape[1], cfg.n_heads * cfg.d_head)
        attn_out = torch.einsum("df,bpf->bpd", self.W_O, z_flat)
        resid = resid + attn_out

        mlp_pre = torch.einsum("md,bpd->bpm", self.W_in, resid) + self.b_in
        mlp_post = F.relu(mlp_pre)
        mlp_out = torch.einsum("dm,bpm->bpd", self.W_out, mlp_post) + self.b_out
        resid = resid + mlp_out
        logits = torch.einsum("bpd,dv->bpv", resid, self.W_U)

        if not return_cache:
            return logits
        cache = {
            "resid_embed": (self.W_E.T[tokens] + self.W_pos[: tokens.shape[1]]).detach(),
            "q": q.detach(),
            "k": k.detach(),
            "v": v.detach(),
            "attn_scores": attn_scores.detach(),
            "attn_pattern": pattern.detach(),
            "z": z.detach(),
            "attn_out": attn_out.detach(),
            "resid_post_attn": (resid - mlp_out).detach(),
            "mlp_pre": mlp_pre.detach(),
            "mlp_post": mlp_post.detach(),
            "mlp_out": mlp_out.detach(),
            "resid_final": resid.detach(),
            "logits": logits.detach(),
        }
        return logits, cache


def infer_config(state_dict: dict[str, torch.Tensor]) -> GrokkingConfig:
    w_e = state_dict["embed.W_E"]
    w_q = state_dict["blocks.0.attn.W_Q"]
    w_in = state_dict["blocks.0.mlp.W_in"]
    w_pos = state_dict["pos_embed.W_pos"]
    p = int(w_e.shape[1] - 1)
    return GrokkingConfig(
        p=p,
        d_model=int(w_e.shape[0]),
        d_mlp=int(w_in.shape[0]),
        n_heads=int(w_q.shape[0]),
        d_head=int(w_q.shape[1]),
        n_ctx=int(w_pos.shape[0]),
    )


def remap_transformer_lens_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    key_map = {
        "embed.W_E": "W_E",
        "pos_embed.W_pos": "W_pos",
        "blocks.0.attn.W_Q": "W_Q",
        "blocks.0.attn.W_K": "W_K",
        "blocks.0.attn.W_V": "W_V",
        "blocks.0.attn.W_O": "W_O",
        "blocks.0.mlp.W_in": "W_in",
        "blocks.0.mlp.b_in": "b_in",
        "blocks.0.mlp.W_out": "W_out",
        "blocks.0.mlp.b_out": "b_out",
        "unembed.W_U": "W_U",
    }
    return {new: state_dict[old].detach().clone() for old, new in key_map.items() if old in state_dict}


def load_full_run_data(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def make_all_mod_add_tokens(p: int, device: str | torch.device = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    xs = torch.arange(p, device=device)
    ys = torch.arange(p, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    eq = torch.full_like(xx, p)
    tokens = torch.stack([xx.reshape(-1), yy.reshape(-1), eq.reshape(-1)], dim=1)
    labels = (tokens[:, 0] + tokens[:, 1]) % p
    return tokens.long(), labels.long()


def cross_entropy_and_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    loss = F.cross_entropy(logits.double(), labels)
    acc = (logits.argmax(dim=-1) == labels).double().mean()
    return float(loss.item()), float(acc.item())
