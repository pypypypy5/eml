from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DTYPE = torch.complex128
REAL_DTYPE = torch.float64


@dataclass(frozen=True)
class MatrixProbeDataset:
    name: str
    inputs: torch.Tensor
    target: torch.Tensor
    description: str


@dataclass
class MatrixProbeResult:
    name: str
    depth: int
    n_inputs: int
    n_outputs: int
    train_n: int
    test_n: int
    steps: int
    train_rmse_norm: float
    test_rmse_norm: float
    test_rmse: float
    baseline_rmse: float
    rmse_baseline_ratio: float
    mean_output_r2: float
    median_output_r2: float
    min_output_r2: float
    max_abs_error: float
    param_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class VectorEMLTree(nn.Module):
    """Vector-valued EML tree for one matrix/module-level probe.

    The module is one EML tree object producing a full output vector. Each
    output coordinate has its own soft terminal choices and gates, because a
    dense matrix maps different coordinates to different input directions. This
    keeps the probe module-level while respecting that EML itself is scalar at
    each coordinate.
    """

    def __init__(self, depth: int, n_inputs: int, n_outputs: int, init_scale: float = 0.25):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_leaves = 2**depth
        self.n_internal = self.n_leaves - 1
        self.leaf_logits = nn.Parameter(
            torch.randn(self.n_leaves, n_outputs, n_inputs + 1, dtype=REAL_DTYPE) * init_scale
        )
        self.gate_logits = nn.Parameter(
            torch.randn(self.n_internal, n_outputs, 2, dtype=REAL_DTYPE) * init_scale + 2.0
        )
        self.out_scale = nn.Parameter(torch.full((n_outputs,), 0.1, dtype=REAL_DTYPE))
        self.out_bias = nn.Parameter(torch.zeros(n_outputs, dtype=REAL_DTYPE))

    @staticmethod
    def eml(left: torch.Tensor, right: torch.Tensor, clamp: float) -> torch.Tensor:
        out = torch.exp(left) - torch.log(right)
        return torch.complex(
            torch.nan_to_num(out.real, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
            torch.nan_to_num(out.imag, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
        )

    def forward(self, inputs: torch.Tensor, tau_leaf: float = 1.0, tau_gate: float = 1.0, clamp: float = 1.0e6):
        batch = inputs.shape[0]
        ones = torch.ones(batch, 1, dtype=REAL_DTYPE, device=inputs.device)
        candidates = torch.cat([ones, inputs.to(REAL_DTYPE)], dim=1).to(DTYPE)
        leaf_probs = F.softmax(self.leaf_logits / tau_leaf, dim=-1)
        current = torch.einsum("bc,loc->blo", candidates, leaf_probs.to(DTYPE))

        gate_probs_all = []
        node_idx = 0
        while current.shape[1] > 1:
            n_pairs = current.shape[1] // 2
            left_child = current[:, 0::2, :]
            right_child = current[:, 1::2, :]
            gates = torch.sigmoid(self.gate_logits[node_idx : node_idx + n_pairs] / tau_gate)
            gate_probs_all.append(gates)
            s_left = gates[:, :, 0].unsqueeze(0).to(DTYPE)
            s_right = gates[:, :, 1].unsqueeze(0).to(DTYPE)
            left = s_left + (1.0 - s_left) * left_child
            right = s_right + (1.0 - s_right) * right_child
            current = self.eml(left, right, clamp=clamp)
            node_idx += n_pairs

        raw = current[:, 0, :]
        out = raw * self.out_scale.to(DTYPE).unsqueeze(0) + self.out_bias.to(DTYPE).unsqueeze(0)
        return out, leaf_probs, torch.cat(gate_probs_all, dim=0)


def standardize_matrix_train_test(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
):
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1.0e-8)
    y_mean = y_train.mean(dim=0, keepdim=True)
    y_std = y_train.std(dim=0, keepdim=True).clamp_min(1.0e-8)
    return (
        (x_train - x_mean) / x_std,
        (y_train - y_mean) / y_std,
        (x_test - x_mean) / x_std,
        (y_test - y_mean) / y_std,
        x_mean,
        x_std,
        y_mean,
        y_std,
    )


def train_matrix_probe(
    dataset: MatrixProbeDataset,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    depth: int,
    steps: int,
    lr: float,
    seed: int,
    imag_penalty: float = 1.0e-2,
    binarity_penalty: float = 1.0e-4,
    clamp: float = 1.0e6,
    out_state_path: str | Path | None = None,
) -> MatrixProbeResult:
    torch.manual_seed(seed)
    x_train = x_train.to(REAL_DTYPE)
    y_train = y_train.to(REAL_DTYPE)
    x_test = x_test.to(REAL_DTYPE)
    y_test = y_test.to(REAL_DTYPE)
    x_train_s, y_train_s, x_test_s, y_test_s, _, _, y_mean, y_std = standardize_matrix_train_test(
        x_train, y_train, x_test, y_test
    )

    tree = VectorEMLTree(depth=depth, n_inputs=x_train.shape[1], n_outputs=y_train.shape[1])
    opt = torch.optim.Adam(tree.parameters(), lr=lr)
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in tree.state_dict().items()}

    for step in range(steps):
        t = step / max(1, steps - 1)
        tau = 2.0 * (0.2 / 2.0) ** t
        opt.zero_grad()
        pred, leaf_probs, gate_probs = tree(x_train_s, tau_leaf=tau, tau_gate=tau, clamp=clamp)
        mse = torch.mean((pred.real - y_train_s) ** 2)
        imag = torch.mean(pred.imag**2)
        leaf_unc = torch.mean(1.0 - leaf_probs.max(dim=-1).values)
        gate_unc = torch.mean(gate_probs * (1.0 - gate_probs))
        loss = mse + imag_penalty * imag + binarity_penalty * (leaf_unc + gate_unc)
        if not torch.isfinite(loss):
            break
        if float(mse.item()) < best_loss:
            best_loss = float(mse.item())
            best_state = {k: v.detach().clone() for k, v in tree.state_dict().items()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

    tree.load_state_dict(best_state)
    if out_state_path is not None:
        torch.save(tree.state_dict(), out_state_path)

    with torch.no_grad():
        train_pred = tree(x_train_s, tau_leaf=0.05, tau_gate=0.05, clamp=clamp)[0].real
        test_pred = tree(x_test_s, tau_leaf=0.05, tau_gate=0.05, clamp=clamp)[0].real
        train_rmse_norm = torch.sqrt(torch.mean((train_pred - y_train_s) ** 2)).item()
        test_rmse_norm = torch.sqrt(torch.mean((test_pred - y_test_s) ** 2)).item()
        test_pred_orig = test_pred * y_std + y_mean
        test_rmse = torch.sqrt(torch.mean((test_pred_orig - y_test) ** 2)).item()
        baseline = y_train.mean(dim=0, keepdim=True).expand_as(y_test)
        baseline_rmse = torch.sqrt(torch.mean((baseline - y_test) ** 2)).item()
        rmse_ratio = test_rmse / max(baseline_rmse, 1.0e-12)

        ss_res = torch.sum((test_pred_orig - y_test) ** 2, dim=0)
        ss_tot = torch.sum((y_test - y_test.mean(dim=0, keepdim=True)) ** 2, dim=0).clamp_min(1.0e-12)
        r2 = 1.0 - ss_res / ss_tot
        max_abs_error = torch.max(torch.abs(test_pred_orig - y_test)).item()

    return MatrixProbeResult(
        name=dataset.name,
        depth=depth,
        n_inputs=int(x_train.shape[1]),
        n_outputs=int(y_train.shape[1]),
        train_n=int(x_train.shape[0]),
        test_n=int(x_test.shape[0]),
        steps=steps,
        train_rmse_norm=float(train_rmse_norm),
        test_rmse_norm=float(test_rmse_norm),
        test_rmse=float(test_rmse),
        baseline_rmse=float(baseline_rmse),
        rmse_baseline_ratio=float(rmse_ratio),
        mean_output_r2=float(r2.mean().item()),
        median_output_r2=float(r2.median().item()),
        min_output_r2=float(r2.min().item()),
        max_abs_error=float(max_abs_error),
        param_count=sum(p.numel() for p in tree.parameters()),
    )


def split_matrix_dataset(
    dataset: MatrixProbeDataset,
    train_n: int,
    test_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = dataset.target.shape[0]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    need = min(n, train_n + test_n)
    idx = perm[:need]
    train_idx = idx[: min(train_n, need)]
    test_idx = idx[min(train_n, need) : need]
    if test_idx.numel() == 0:
        test_idx = train_idx
    return (
        dataset.inputs[train_idx].to(REAL_DTYPE),
        dataset.target[train_idx].to(REAL_DTYPE),
        dataset.inputs[test_idx].to(REAL_DTYPE),
        dataset.target[test_idx].to(REAL_DTYPE),
    )


def build_matrix_probe_datasets(
    logits: torch.Tensor,
    cache: dict[str, torch.Tensor],
    p: int,
    max_heads: int = 2,
) -> list[MatrixProbeDataset]:
    probes: list[MatrixProbeDataset] = []
    final_pos = 2

    resid_embed_all = cache["resid_embed"].reshape(-1, cache["resid_embed"].shape[-1]).to(REAL_DTYPE)
    for head in range(min(max_heads, cache["q"].shape[2])):
        probes.append(
            MatrixProbeDataset(
                name=f"W_Q_head{head}_allpos",
                inputs=resid_embed_all,
                target=cache["q"][:, :, head, :].reshape(-1, cache["q"].shape[-1]).to(REAL_DTYPE),
                description=f"Matrix probe for W_Q head {head}: resid_embed at all positions -> q_head.",
            )
        )
        probes.append(
            MatrixProbeDataset(
                name=f"W_K_head{head}_allpos",
                inputs=resid_embed_all,
                target=cache["k"][:, :, head, :].reshape(-1, cache["k"].shape[-1]).to(REAL_DTYPE),
                description=f"Matrix probe for W_K head {head}: resid_embed at all positions -> k_head.",
            )
        )
        probes.append(
            MatrixProbeDataset(
                name=f"W_V_head{head}_allpos",
                inputs=resid_embed_all,
                target=cache["v"][:, :, head, :].reshape(-1, cache["v"].shape[-1]).to(REAL_DTYPE),
                description=f"Matrix probe for W_V head {head}: resid_embed at all positions -> v_head.",
            )
        )

    z_flat = cache["z"][:, final_pos, :, :].reshape(cache["z"].shape[0], -1).to(REAL_DTYPE)
    probes.append(
        MatrixProbeDataset(
            name="W_O_final",
            inputs=z_flat,
            target=cache["attn_out"][:, final_pos, :].to(REAL_DTYPE),
            description="Matrix probe for W_O at final position: flattened attention z -> attn_out.",
        )
    )
    probes.append(
        MatrixProbeDataset(
            name="W_in_final",
            inputs=cache["resid_post_attn"][:, final_pos, :].to(REAL_DTYPE),
            target=cache["mlp_pre"][:, final_pos, :].to(REAL_DTYPE),
            description="Matrix probe for W_in+b_in at final position: resid_post_attn -> mlp_pre.",
        )
    )
    probes.append(
        MatrixProbeDataset(
            name="W_out_final",
            inputs=cache["mlp_post"][:, final_pos, :].to(REAL_DTYPE),
            target=cache["mlp_out"][:, final_pos, :].to(REAL_DTYPE),
            description="Matrix probe for W_out+b_out at final position: mlp_post -> mlp_out.",
        )
    )
    probes.append(
        MatrixProbeDataset(
            name="W_U_final",
            inputs=cache["resid_final"][:, final_pos, :].to(REAL_DTYPE),
            target=logits[:, :p].to(REAL_DTYPE),
            description="Matrix probe for unembedding at final position: resid_final -> class logits.",
        )
    )
    return probes
