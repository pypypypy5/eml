from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DTYPE = torch.complex128
REAL_DTYPE = torch.float64


@dataclass
class EMLProbeResult:
    name: str
    depth: int
    n_inputs: int
    train_n: int
    test_n: int
    steps: int
    target_mean: float
    target_std: float
    train_rmse_norm: float
    test_rmse_norm: float
    test_rmse: float
    baseline_rmse: float
    rmse_baseline_ratio: float
    test_r2: float
    max_abs_error: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


class SoftEMLTree(nn.Module):
    """Small differentiable EML tree for scalar node probes.

    Leaves choose among constant 1 and `n_inputs` real scalar inputs via
    softmax. Each internal EML node can bypass either child with constant 1 via
    sigmoid gates, following the public EML PyTorch trainer's parameterization.
    """

    def __init__(self, depth: int, n_inputs: int, init_scale: float = 0.5):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if n_inputs < 1:
            raise ValueError("n_inputs must be >= 1")
        self.depth = depth
        self.n_inputs = n_inputs
        self.n_leaves = 2**depth
        self.n_internal = self.n_leaves - 1
        self.leaf_logits = nn.Parameter(
            torch.randn(self.n_leaves, n_inputs + 1, dtype=REAL_DTYPE) * init_scale
        )
        # Positive init follows the public EML trainer and keeps early values
        # finite by preferring the constant-1 bypass.
        self.gate_logits = nn.Parameter(torch.randn(self.n_internal, 2, dtype=REAL_DTYPE) * init_scale + 2.0)
        self.out_scale = nn.Parameter(torch.tensor(0.1, dtype=REAL_DTYPE))
        self.out_bias = nn.Parameter(torch.tensor(0.0, dtype=REAL_DTYPE))

    @staticmethod
    def eml(left: torch.Tensor, right: torch.Tensor, clamp: float) -> torch.Tensor:
        out = torch.exp(left) - torch.log(right)
        return torch.complex(
            torch.nan_to_num(out.real, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
            torch.nan_to_num(out.imag, nan=0.0, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp),
        )

    def forward(self, inputs: torch.Tensor, tau_leaf: float = 1.0, tau_gate: float = 1.0, clamp: float = 1.0e6):
        inputs = inputs.to(REAL_DTYPE)
        batch = inputs.shape[0]
        ones = torch.ones(batch, 1, dtype=REAL_DTYPE, device=inputs.device)
        candidates = torch.cat([ones, inputs], dim=1).to(DTYPE)

        leaf_probs = F.softmax(self.leaf_logits / tau_leaf, dim=1)
        current = candidates @ leaf_probs.T.to(DTYPE)

        node_idx = 0
        gate_probs_all = []
        while current.shape[1] > 1:
            n_pairs = current.shape[1] // 2
            left_child = current[:, 0::2]
            right_child = current[:, 1::2]
            gates = torch.sigmoid(self.gate_logits[node_idx : node_idx + n_pairs] / tau_gate)
            gate_probs_all.append(gates)

            s_left = gates[:, 0].unsqueeze(0).to(DTYPE)
            s_right = gates[:, 1].unsqueeze(0).to(DTYPE)
            left = s_left + (1.0 - s_left) * left_child
            right = s_right + (1.0 - s_right) * right_child
            current = self.eml(left, right, clamp=clamp)
            node_idx += n_pairs

        raw = current[:, 0]
        calibrated = raw * self.out_scale.to(DTYPE) + self.out_bias.to(DTYPE)
        return calibrated, leaf_probs, torch.cat(gate_probs_all, dim=0)

    def hard_expression(self, input_names: list[str] | None = None) -> str:
        names = ["1"] + (input_names if input_names else [f"x{i}" for i in range(self.n_inputs)])
        with torch.no_grad():
            leaf_choice = torch.argmax(self.leaf_logits, dim=1).cpu().tolist()
            gate_choice = (torch.sigmoid(self.gate_logits) >= 0.5).cpu().tolist()

        level = [names[i] for i in leaf_choice]
        node_idx = 0
        while len(level) > 1:
            next_level = []
            for pair_idx in range(len(level) // 2):
                left = "1" if gate_choice[node_idx + pair_idx][0] else level[2 * pair_idx]
                right = "1" if gate_choice[node_idx + pair_idx][1] else level[2 * pair_idx + 1]
                next_level.append(f"EML[{left},{right}]")
            node_idx += len(level) // 2
            level = next_level
        return level[0]


def standardize_train_test(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1.0e-8)
    y_mean = y_train.mean()
    y_std = y_train.std().clamp_min(1.0e-8)
    return (
        (x_train - x_mean) / x_std,
        (y_train - y_mean) / y_std,
        (x_test - x_mean) / x_std,
        (y_test - y_mean) / y_std,
        y_mean,
        y_std,
    )


def train_eml_probe(
    name: str,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    depth: int = 2,
    steps: int = 400,
    lr: float = 1.0e-2,
    seed: int = 0,
    imag_penalty: float = 1.0e-2,
    binarity_penalty: float = 1.0e-3,
    clamp: float = 1.0e6,
    out_expr_path: str | Path | None = None,
) -> EMLProbeResult:
    torch.manual_seed(seed)
    x_train_s, y_train_s, x_test_s, y_test_s, y_mean, y_std = standardize_train_test(
        x_train.to(REAL_DTYPE),
        y_train.to(REAL_DTYPE),
        x_test.to(REAL_DTYPE),
        y_test.to(REAL_DTYPE),
    )
    tree = SoftEMLTree(depth=depth, n_inputs=x_train_s.shape[1])
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
        leaf_unc = torch.mean(1.0 - leaf_probs.max(dim=1).values)
        gate_unc = torch.mean(gate_probs * (1.0 - gate_probs))
        loss = mse + imag_penalty * imag + binarity_penalty * (leaf_unc + gate_unc)
        if not torch.isfinite(loss):
            break
        loss_value = float(mse.item())
        if loss_value < best_loss:
            best_loss = loss_value
            best_state = {k: v.detach().clone() for k, v in tree.state_dict().items()}
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tree.parameters(), 1.0)
        opt.step()

    tree.load_state_dict(best_state)
    with torch.no_grad():
        train_pred = tree(x_train_s, tau_leaf=0.05, tau_gate=0.05, clamp=clamp)[0].real
        test_pred = tree(x_test_s, tau_leaf=0.05, tau_gate=0.05, clamp=clamp)[0].real
        train_rmse_norm = torch.sqrt(torch.mean((train_pred - y_train_s) ** 2)).item()
        test_rmse_norm = torch.sqrt(torch.mean((test_pred - y_test_s) ** 2)).item()
        test_pred_orig = test_pred * y_std + y_mean
        test_rmse = torch.sqrt(torch.mean((test_pred_orig - y_test) ** 2)).item()
        baseline = torch.full_like(y_test, y_train.mean())
        baseline_rmse = torch.sqrt(torch.mean((baseline - y_test) ** 2)).item()
        rmse_baseline_ratio = test_rmse / max(baseline_rmse, 1.0e-12)
        ss_res = torch.sum((test_pred_orig - y_test) ** 2)
        ss_tot = torch.sum((y_test - y_test.mean()) ** 2).clamp_min(1.0e-12)
        r2 = (1.0 - ss_res / ss_tot).item()
        max_abs_error = torch.max(torch.abs(test_pred_orig - y_test)).item()

    if out_expr_path is not None:
        Path(out_expr_path).write_text(tree.hard_expression(["u", "v", "w"][: x_train.shape[1]]) + "\n", encoding="utf-8")

    return EMLProbeResult(
        name=name,
        depth=depth,
        n_inputs=int(x_train.shape[1]),
        train_n=int(x_train.shape[0]),
        test_n=int(x_test.shape[0]),
        steps=steps,
        target_mean=float(y_mean.item()),
        target_std=float(y_std.item()),
        train_rmse_norm=float(train_rmse_norm),
        test_rmse_norm=float(test_rmse_norm),
        test_rmse=float(test_rmse),
        baseline_rmse=float(baseline_rmse),
        rmse_baseline_ratio=float(rmse_baseline_ratio),
        test_r2=float(r2),
        max_abs_error=float(max_abs_error),
    )
