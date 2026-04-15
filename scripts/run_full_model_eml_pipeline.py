#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from grokking_eml.eml_probe import predict_with_artifacts, train_eml_probe
from grokking_eml.fourier_formula import (
    evaluate_kernel_formula,
    fit_fourier_kernel,
    kernel_approximation_error,
    translation_invariant_kernel,
)
from grokking_eml.model import (
    OneLayerModularTransformer,
    cross_entropy_and_accuracy,
    load_full_run_data,
    make_all_mod_add_tokens,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build whole-model IO data, fit Fourier structure, and train a direct EML tree on x,y,z -> logit."
    )
    parser.add_argument("--repo-id", default="callummcdougall/grokking_full_run_data")
    parser.add_argument("--filename", default="full_run_data.pth")
    parser.add_argument("--checkpoint-index", type=int, default=400)
    parser.add_argument("--data-dir", default="large_files")
    parser.add_argument("--out-dir", default="runs/full_model_eml")
    parser.add_argument("--dataset-cache", default="", help="Directory containing cached whole-model IO tensors.")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--train-n", type=int, default=4096)
    parser.add_argument("--test-n", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-full-grid", action="store_true", help="Evaluate the trained EML tree on all p^3 candidate logits.")
    parser.add_argument("--eval-batch-size", type=int, default=32768)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def save_full_model_io(cache_dir: Path, payload: dict, metadata: dict) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "full_model_io.pt"
    torch.save(payload, path)
    manifest = {**metadata, "path": str(path)}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_full_model_io(cache_dir: Path) -> tuple[dict, dict] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = torch.load(manifest["path"], map_location="cpu", weights_only=False)
    return payload, manifest


def build_or_load_full_model_io(args: argparse.Namespace, root: Path, cache_dir: Path) -> tuple[dict, dict, bool]:
    cached = None if args.refresh_cache else load_full_model_io(cache_dir)
    if cached is not None:
        payload, manifest = cached
        print(f"Loaded whole-model IO cache from {manifest['path']}")
        return payload, manifest, True

    data_dir = root / args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=data_dir)
    full_run_data = load_full_run_data(checkpoint_path, map_location=args.device)
    model = OneLayerModularTransformer.from_state_dict(full_run_data["state_dicts"][args.checkpoint_index]).to(args.device)
    tokens, labels = make_all_mod_add_tokens(model.cfg.p, device=args.device)
    with torch.no_grad():
        logits_full = model(tokens)
        logits = logits_full[:, -1, : model.cfg.p].cpu()

    payload = {
        "p": model.cfg.p,
        "tokens": tokens.cpu(),
        "labels": labels.cpu(),
        "logits": logits,
    }
    metadata = {
        "checkpoint": {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "checkpoint_index": args.checkpoint_index,
            "path": str(checkpoint_path),
        },
        "p": model.cfg.p,
        "n_inputs": int(tokens.shape[0]),
        "logit_shape": list(logits.shape),
    }
    path = save_full_model_io(cache_dir, payload, metadata)
    metadata["path"] = str(path)
    print(f"Saved whole-model IO cache to {path}")
    return payload, metadata, False


def sample_candidate_io(
    logits: torch.Tensor,
    p: int,
    train_n: int,
    test_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = p * p * p
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(total, generator=generator)[: min(total, train_n + test_n)]
    x = idx // (p * p)
    rem = idx % (p * p)
    y = rem // p
    z = rem % p
    inputs = torch.stack([x, y, z], dim=1).to(torch.float64)
    targets = logits[(x * p + y).long(), z.long()].to(torch.float64)
    train_count = min(train_n, idx.numel())
    return (
        inputs[:train_count],
        targets[:train_count],
        inputs[train_count:],
        targets[train_count:],
    )


def evaluate_eml_tree_full_grid(
    tree,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    labels: torch.Tensor,
    p: int,
    batch_size: int,
) -> dict[str, float]:
    eml_logits = torch.empty(p * p, p, dtype=torch.float64)
    total = p * p * p
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        idx = torch.arange(start, end)
        x = idx // (p * p)
        rem = idx % (p * p)
        y = rem // p
        z = rem % p
        inputs = torch.stack([x, y, z], dim=1).to(torch.float64)
        pred = predict_with_artifacts(tree, inputs, x_mean, x_std, y_mean, y_std)
        eml_logits[(x * p + y).long(), z.long()] = pred
    loss, acc = cross_entropy_and_accuracy(eml_logits, labels)
    margins = eml_logits[torch.arange(labels.numel()), labels] - eml_logits.masked_fill(
        torch.nn.functional.one_hot(labels, p).bool(), -torch.inf
    ).max(dim=-1).values
    return {
        "accuracy": acc,
        "cross_entropy": loss,
        "min_correct_margin": float(margins.min().item()),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = root / args.dataset_cache if args.dataset_cache else out_dir / "datasets"

    payload, manifest, loaded_from_cache = build_or_load_full_model_io(args, root, cache_dir)
    p = int(payload["p"])
    logits = payload["logits"].to(torch.float64)
    labels = payload["labels"].long()

    model_loss, model_acc = cross_entropy_and_accuracy(logits, labels)
    kernel = translation_invariant_kernel(logits, p)
    fit = fit_fourier_kernel(kernel, top_k=args.top_k)
    fourier_metrics = evaluate_kernel_formula(fit, labels)
    approx_metrics = kernel_approximation_error(logits, fit)

    formula = fit.formula(variable="d")
    (out_dir / "fourier_formula_from_full_io.wl").write_text(formula + "\n", encoding="utf-8")
    torch.save(
        {
            "kernel": torch.tensor(kernel),
            "reconstructed_kernel": torch.tensor(fit.kernel),
            "selected_freqs": fit.selected_freqs,
            "c0": fit.c0,
            "a": fit.a,
            "b": fit.b,
        },
        out_dir / "fourier_fit_from_full_io.pt",
    )

    x_train, y_train, x_test, y_test = sample_candidate_io(
        logits=logits,
        p=p,
        train_n=args.train_n,
        test_n=args.test_n,
        seed=args.seed,
    )
    eml_result, tree, x_mean, x_std, y_mean, y_std = train_eml_probe(
        name="full_model_candidate_logit_xyz",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        depth=args.depth,
        steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        out_expr_path=out_dir / "direct_eml_tree.eml",
        return_artifacts=True,
    )
    torch.save(
        {
            "state_dict": tree.state_dict(),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "depth": args.depth,
            "n_inputs": 3,
        },
        out_dir / "direct_eml_tree_state.pt",
    )

    full_grid_metrics = None
    if args.eval_full_grid:
        full_grid_metrics = evaluate_eml_tree_full_grid(
            tree=tree,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            labels=labels,
            p=p,
            batch_size=args.eval_batch_size,
        )

    summary = {
        "checkpoint": manifest["checkpoint"],
        "dataset_cache": {
            "dir": str(cache_dir),
            "manifest": str(cache_dir / "manifest.json"),
            "loaded_from_cache": loaded_from_cache,
        },
        "model_replay": {
            "accuracy": model_acc,
            "cross_entropy": model_loss,
            "p": p,
            "n_equation_inputs": int(labels.numel()),
            "n_candidate_io": int(p * p * p),
        },
        "fourier_from_full_io": {
            **fourier_metrics,
            "approximation_error_vs_model_logits": approx_metrics,
            "formula_file": str(out_dir / "fourier_formula_from_full_io.wl"),
        },
        "direct_eml_tree": {
            **eml_result.to_dict(),
            "input_variables": ["x", "y", "z"],
            "target": "model logit for candidate z on input x,y",
            "full_grid_classifier": full_grid_metrics,
            "expression_file": str(out_dir / "direct_eml_tree.eml"),
            "state_file": str(out_dir / "direct_eml_tree_state.pt"),
        },
        "fit_config": {
            "top_k": args.top_k,
            "depth": args.depth,
            "steps": args.steps,
            "lr": args.lr,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "seed": args.seed,
            "eval_full_grid": args.eval_full_grid,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
