from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeDataset:
    name: str
    inputs: torch.Tensor
    target: torch.Tensor
    description: str


def _norm_token(t: torch.Tensor, p: int) -> torch.Tensor:
    return (2.0 * t.to(torch.float64) / float(p - 1)) - 1.0


def build_default_probe_datasets(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    logits: torch.Tensor,
    cache: dict[str, torch.Tensor],
    p: int,
    max_mlp_neurons: int = 4,
) -> list[ProbeDataset]:
    """Create scalar graph-node datasets for EML probing.

    The graph tensors are high-dimensional, so each probe is one scalar slice of
    one local operation. Inputs are the actual immediate inputs to that operation
    rather than external Fourier or modular coordinates.
    """
    probes: list[ProbeDataset] = []

    # Local softmax nodes: attention scores for one query/head -> one attention
    # probability. This asks whether a small EML tree can learn a scalar slice of
    # softmax from the three local score inputs.
    attn_scores = cache["attn_scores"].to(torch.float64)
    attn_pattern = cache["attn_pattern"].to(torch.float64)
    for head in range(min(2, cache["attn_pattern"].shape[1])):
        for key_pos in range(3):
            probes.append(
                ProbeDataset(
                    name=f"attn_softmax_h{head}_q2_k{key_pos}",
                    inputs=attn_scores[:, head, 2, :3],
                    target=attn_pattern[:, head, 2, key_pos],
                    description=f"Local softmax node: scores[h={head}, q=2, k=0..2] -> pattern[k={key_pos}].",
                )
            )

    # Local attention weighted-sum nodes:
    # z[q,h,d] = sum_k pattern[q,k] * v[k,h,d]. Each scalar probe receives the
    # three pattern weights and the three value scalars for a selected dimension.
    v = cache["v"].to(torch.float64)
    z = cache["z"].to(torch.float64)
    for head in range(min(2, z.shape[2])):
        variances = z[:, 2, head, :].var(dim=0)
        top_dims = torch.topk(variances, k=min(2, z.shape[-1])).indices.tolist()
        for dim in top_dims:
            inputs = torch.cat(
                [
                    attn_pattern[:, head, 2, :3],
                    v[:, :3, head, dim],
                ],
                dim=1,
            )
            probes.append(
                ProbeDataset(
                    name=f"attn_weighted_sum_h{head}_d{dim}",
                    inputs=inputs,
                    target=z[:, 2, head, dim],
                    description=f"Local attention weighted sum: pattern[3], v_dim[3] -> z[h={head}, d={dim}].",
                )
            )

    # Local ReLU nodes: one MLP pre-activation scalar -> its post-ReLU scalar.
    mlp_pre = cache["mlp_pre"][:, 2, :].to(torch.float64)
    variances = mlp_pre.var(dim=0)
    top_neurons = torch.topk(variances, k=min(max_mlp_neurons, mlp_pre.shape[1])).indices.tolist()
    for neuron in top_neurons:
        pre = mlp_pre[:, neuron]
        probes.append(
            ProbeDataset(
                name=f"mlp_relu_n{neuron}_local",
                inputs=pre.unsqueeze(1),
                target=cache["mlp_post"][:, 2, neuron].to(torch.float64),
                description=f"Local ReLU node for MLP neuron {neuron}: input is that neuron's pre-activation.",
            )
        )
    return probes


def split_probe_dataset(
    dataset: ProbeDataset,
    train_n: int,
    test_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = dataset.target.numel()
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    need = min(n, train_n + test_n)
    idx = perm[:need]
    train_idx = idx[: min(train_n, need)]
    test_idx = idx[min(train_n, need) : need]
    if test_idx.numel() == 0:
        test_idx = train_idx
    return (
        dataset.inputs[train_idx].to(torch.float64),
        dataset.target[train_idx].to(torch.float64),
        dataset.inputs[test_idx].to(torch.float64),
        dataset.target[test_idx].to(torch.float64),
    )
