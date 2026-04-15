from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import cross_entropy_and_accuracy


@dataclass(frozen=True)
class FourierKernelFit:
    p: int
    c0: float
    a: dict[int, float]
    b: dict[int, float]
    selected_freqs: list[int]
    kernel: np.ndarray

    def formula(self, variable: str = "d", max_terms: int | None = None) -> str:
        freqs = self.selected_freqs if max_terms is None else self.selected_freqs[:max_terms]
        parts = [f"{self.c0:.10g}"]
        for f in freqs:
            angle = f"(2*Pi*{f}*{variable})/{self.p}"
            if abs(self.a[f]) > 1e-12:
                parts.append(f"({self.a[f]:.10g})*Cos[{angle}]")
            if abs(self.b[f]) > 1e-12:
                parts.append(f"({self.b[f]:.10g})*Sin[{angle}]")
        return " + ".join(parts)

    def parameterized_formula(self, variable: str = "d", max_terms: int | None = None) -> str:
        freqs = self.selected_freqs if max_terms is None else self.selected_freqs[:max_terms]
        parts = ["c0"]
        for f in freqs:
            angle = f"(2*Pi*{f}*{variable})/{self.p}"
            parts.append(f"a{f}*Cos[{angle}]")
            parts.append(f"b{f}*Sin[{angle}]")
        return " + ".join(parts)


def translation_invariant_kernel(logits: torch.Tensor, p: int) -> np.ndarray:
    """Average model logits by d=(x+y-z) mod p."""
    logits_np = logits.detach().cpu().double().numpy()
    kernel = np.zeros(p, dtype=np.float64)
    counts = np.zeros(p, dtype=np.int64)
    for x in range(p):
        for y in range(p):
            row = logits_np[x * p + y]
            for z in range(p):
                d = (x + y - z) % p
                kernel[d] += row[z]
                counts[d] += 1
    return kernel / counts


def logits_from_kernel(kernel: np.ndarray, p: int) -> torch.Tensor:
    out = np.empty((p * p, p), dtype=np.float64)
    for x in range(p):
        for y in range(p):
            row = x * p + y
            for z in range(p):
                out[row, z] = kernel[(x + y - z) % p]
    return torch.tensor(out, dtype=torch.float64)


def fit_fourier_kernel(kernel: np.ndarray, top_k: int = 5) -> FourierKernelFit:
    p = int(kernel.shape[0])
    centered = kernel - kernel.mean()
    spectrum = np.fft.rfft(centered)
    amplitudes = np.abs(spectrum)
    # Ignore DC. For odd p=113, rfft bins 1..56 are the independent frequencies.
    candidate_freqs = list(range(1, len(amplitudes)))
    selected = sorted(candidate_freqs, key=lambda f: amplitudes[f], reverse=True)[:top_k]
    selected = sorted(selected)

    d = np.arange(p, dtype=np.float64)
    cols = [np.ones_like(d)]
    names: list[tuple[str, int]] = [("c0", 0)]
    for f in selected:
        angle = 2.0 * np.pi * f * d / p
        cols.append(np.cos(angle))
        names.append(("a", f))
        cols.append(np.sin(angle))
        names.append(("b", f))

    design = np.stack(cols, axis=1)
    coeffs, *_ = np.linalg.lstsq(design, kernel, rcond=None)
    recon = design @ coeffs

    a: dict[int, float] = {}
    b: dict[int, float] = {}
    for value, (kind, freq) in zip(coeffs, names, strict=True):
        if kind == "a":
            a[freq] = float(value)
        elif kind == "b":
            b[freq] = float(value)

    # Keep the reconstructed top-k kernel, since this is the actual formula we evaluate.
    return FourierKernelFit(
        p=p,
        c0=float(coeffs[0]),
        a=a,
        b=b,
        selected_freqs=selected,
        kernel=recon,
    )


def evaluate_kernel_formula(fit: FourierKernelFit, labels: torch.Tensor) -> dict[str, float | list[int]]:
    formula_logits = logits_from_kernel(fit.kernel, fit.p)
    loss, acc = cross_entropy_and_accuracy(formula_logits, labels.cpu())
    margins = formula_logits[torch.arange(labels.numel()), labels.cpu()] - formula_logits.masked_fill(
        torch.nn.functional.one_hot(labels.cpu(), fit.p).bool(), -torch.inf
    ).max(dim=-1).values
    return {
        "top_freqs": fit.selected_freqs,
        "loss": loss,
        "accuracy": acc,
        "min_correct_margin": float(margins.min().item()),
    }


def kernel_approximation_error(original_logits: torch.Tensor, fit: FourierKernelFit) -> dict[str, float]:
    formula_logits = logits_from_kernel(fit.kernel, fit.p).to(original_logits.device)
    diff = (formula_logits - original_logits.double()).detach().cpu()
    return {
        "mse": float((diff**2).mean().item()),
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
    }
