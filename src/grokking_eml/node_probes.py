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

    The graph tensors are high-dimensional, so each probe is one scalar slice or
    one local scalar operation. This makes the question measurable: how much of
    that node's scalar behavior can a small EML tree capture?
    """
    x_raw = tokens[:, 0].to(torch.float64)
    y_raw = tokens[:, 1].to(torch.float64)
    x = _norm_token(tokens[:, 0], p)
    y = _norm_token(tokens[:, 1], p)
    xy = torch.stack([x, y], dim=1)
    probes: list[ProbeDataset] = []

    probes.append(
        ProbeDataset(
            name="logit_correct_from_xy",
            inputs=xy,
            target=logits[torch.arange(labels.numel()), labels].to(torch.float64),
            description="Final correct-class logit as a function of normalized input tokens x,y.",
        )
    )
    probes.append(
        ProbeDataset(
            name="logit_zero_from_xy",
            inputs=xy,
            target=logits[:, 0].to(torch.float64),
            description="Final logit for candidate z=0 as a function of normalized input tokens x,y.",
        )
    )

    for head in range(min(2, cache["attn_pattern"].shape[1])):
        for key_pos in range(2):
            probes.append(
                ProbeDataset(
                    name=f"attn_h{head}_q2_k{key_pos}_from_xy",
                    inputs=xy,
                    target=cache["attn_pattern"][:, head, 2, key_pos].to(torch.float64),
                    description=f"Attention pattern head {head}, query position 2, key position {key_pos}.",
                )
            )

    mlp_pre = cache["mlp_pre"][:, 2, :].to(torch.float64)
    variances = mlp_pre.var(dim=0)
    top_neurons = torch.topk(variances, k=min(max_mlp_neurons, mlp_pre.shape[1])).indices.tolist()
    for neuron in top_neurons:
        probes.append(
            ProbeDataset(
                name=f"mlp_pre_n{neuron}_from_xy",
                inputs=xy,
                target=mlp_pre[:, neuron],
                description=f"MLP pre-activation neuron {neuron} at final token position.",
            )
        )
        pre = mlp_pre[:, neuron]
        probes.append(
            ProbeDataset(
                name=f"mlp_relu_n{neuron}_local",
                inputs=pre.unsqueeze(1),
                target=cache["mlp_post"][:, 2, neuron].to(torch.float64),
                description=f"Local ReLU node for MLP neuron {neuron}: input is that neuron's pre-activation.",
            )
        )

    # A direct periodic score probe: sample all candidate logits and use
    # d=(x+y-z) mod p as the single input. This is the most EML-friendly view of
    # the modular-addition circuit.
    z = torch.arange(p, dtype=torch.float64).repeat(tokens.shape[0])
    d = ((x_raw.repeat_interleave(p) + y_raw.repeat_interleave(p) - z) % p).to(torch.float64)
    d_norm = (2.0 * d / float(p - 1)) - 1.0
    probes.append(
        ProbeDataset(
            name="candidate_logit_from_modular_difference",
            inputs=d_norm.unsqueeze(1),
            target=logits[:, :p].to(torch.float64).reshape(-1),
            description="All candidate logits as a function of d=(x+y-z) mod p.",
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
