#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from grokking_eml.eml_probe import predict_with_artifacts, train_eml_probe
from grokking_eml.fourier_formula import fit_fourier_kernel, logits_from_kernel, translation_invariant_kernel
from grokking_eml.model import OneLayerModularTransformer, load_full_run_data, make_all_mod_add_tokens


IDENTITY_DEPTH4 = "EML[1,EML[EML[1,EML[x0,1]],1]]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EML-tree Fourier-feature ensemble and test modular addition.")
    parser.add_argument("--repo-id", default="callummcdougall/grokking_full_run_data")
    parser.add_argument("--filename", default="full_run_data.pth")
    parser.add_argument("--checkpoint-index", type=int, default=400)
    parser.add_argument("--data-dir", default="large_files")
    parser.add_argument("--out-dir", default="runs/eml_fourier_ensemble")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--init-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def classifier_metrics(kernel: np.ndarray, labels: torch.Tensor, p: int) -> dict[str, float]:
    logits = logits_from_kernel(kernel, p)
    loss = F.cross_entropy(logits.double(), labels.cpu())
    acc = (logits.argmax(dim=-1) == labels.cpu()).double().mean()
    margins = logits[torch.arange(labels.numel()), labels.cpu()] - logits.masked_fill(
        F.one_hot(labels.cpu(), p).bool(), -torch.inf
    ).max(dim=-1).values
    return {
        "accuracy": float(acc.item()),
        "cross_entropy": float(loss.item()),
        "min_correct_margin": float(margins.min().item()),
    }


def main() -> None:
    args = parse_args()
    if args.depth != 4:
        raise ValueError("This ensemble currently uses the depth-4 exact EML identity initialization; pass --depth 4.")

    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    expr_dir = out_dir / "expressions"
    data_dir.mkdir(parents=True, exist_ok=True)
    expr_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=data_dir)
    full_run_data = load_full_run_data(checkpoint_path, map_location=args.device)
    model = OneLayerModularTransformer.from_state_dict(full_run_data["state_dicts"][args.checkpoint_index]).to(args.device)

    tokens, labels = make_all_mod_add_tokens(model.cfg.p, device=args.device)
    with torch.no_grad():
        logits_full = model(tokens)
        logits = logits_full[:, -1, : model.cfg.p]

    kernel = translation_invariant_kernel(logits.cpu(), model.cfg.p)
    fit = fit_fourier_kernel(kernel, top_k=args.top_k)

    d = np.arange(model.cfg.p, dtype=np.float64)
    eml_kernel = np.full(model.cfg.p, fit.c0, dtype=np.float64)
    exact_kernel = fit.kernel
    component_results = []

    for f_idx, freq in enumerate(fit.selected_freqs):
        angle = 2.0 * np.pi * freq * d / model.cfg.p
        features = {
            "cos": np.cos(angle),
            "sin": np.sin(angle),
        }
        for kind, values in features.items():
            coef = fit.a[freq] if kind == "cos" else fit.b[freq]
            x = torch.tensor(values, dtype=torch.float64).unsqueeze(1)
            y = torch.tensor(values, dtype=torch.float64)
            expr_path = expr_dir / f"freq_{freq}_{kind}_identity.eml"
            result, tree, x_mean, x_std, y_mean, y_std = train_eml_probe(
                name=f"freq_{freq}_{kind}_identity",
                x_train=x,
                y_train=y,
                x_test=x,
                y_test=y,
                depth=args.depth,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed + f_idx * 17 + (0 if kind == "cos" else 1),
                init_expr=IDENTITY_DEPTH4,
                init_noise=args.init_noise,
                out_expr_path=expr_path,
                return_artifacts=True,
            )
            pred = predict_with_artifacts(tree, x, x_mean, x_std, y_mean, y_std).cpu().numpy()
            eml_kernel += coef * pred
            component_results.append(
                {
                    **result.to_dict(),
                    "frequency": freq,
                    "basis": kind,
                    "coefficient": float(coef),
                    "max_basis_abs_error": float(np.max(np.abs(pred - values))),
                    "expression_file": str(expr_path),
                }
            )

    exact_metrics = classifier_metrics(exact_kernel, labels, model.cfg.p)
    eml_metrics = classifier_metrics(eml_kernel, labels, model.cfg.p)
    kernel_error = eml_kernel - exact_kernel

    summary = {
        "checkpoint": {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "checkpoint_index": args.checkpoint_index,
            "path": str(checkpoint_path),
        },
        "fit_config": {
            "top_k": args.top_k,
            "depth": args.depth,
            "steps": args.steps,
            "lr": args.lr,
            "init_noise": args.init_noise,
            "seed": args.seed,
            "identity_init": IDENTITY_DEPTH4,
        },
        "selected_freqs": fit.selected_freqs,
        "exact_fourier_classifier": exact_metrics,
        "eml_ensemble_classifier": eml_metrics,
        "kernel_error_vs_exact_fourier": {
            "mse": float(np.mean(kernel_error**2)),
            "mae": float(np.mean(np.abs(kernel_error))),
            "max_abs": float(np.max(np.abs(kernel_error))),
        },
        "components": component_results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save(
        {
            "exact_kernel": torch.tensor(exact_kernel),
            "eml_kernel": torch.tensor(eml_kernel),
            "selected_freqs": fit.selected_freqs,
            "coefficients": {"c0": fit.c0, "a": fit.a, "b": fit.b},
        },
        out_dir / "eml_fourier_ensemble.pt",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
