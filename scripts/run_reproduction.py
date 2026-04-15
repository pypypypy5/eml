#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from grokking_eml.eml_lowering import eml_stats, lower_to_eml
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
    parser = argparse.ArgumentParser(description="Replay a public grokking checkpoint and lower its Fourier formula to EML.")
    parser.add_argument("--repo-id", default="callummcdougall/grokking_full_run_data")
    parser.add_argument("--filename", default="full_run_data.pth")
    parser.add_argument("--checkpoint-index", type=int, default=400)
    parser.add_argument("--data-dir", default="large_files")
    parser.add_argument("--out-dir", default="runs/modular_addition_eml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lower-dominant-term", action="store_true", help="Also lower the strongest single Fourier term to a pure EML expression.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=data_dir)
    full_run_data = load_full_run_data(checkpoint_path, map_location=args.device)
    state_dict = full_run_data["state_dicts"][args.checkpoint_index]
    model = OneLayerModularTransformer.from_state_dict(state_dict).to(args.device)

    tokens, labels = make_all_mod_add_tokens(model.cfg.p, device=args.device)
    with torch.no_grad():
        logits_full, cache = model(tokens, return_cache=True)
        logits = logits_full[:, -1, : model.cfg.p]
    model_loss, model_acc = cross_entropy_and_accuracy(logits, labels)

    kernel = translation_invariant_kernel(logits, model.cfg.p)
    fit = fit_fourier_kernel(kernel, top_k=args.top_k)
    formula_metrics = evaluate_kernel_formula(fit, labels.cpu())
    approx_metrics = kernel_approximation_error(logits.cpu(), fit)

    elementary_formula = fit.formula(variable="d")
    parameterized_formula = fit.parameterized_formula(variable="d")
    (out_dir / "fourier_formula.wl").write_text(elementary_formula + "\n", encoding="utf-8")
    (out_dir / "fourier_formula_parameterized.wl").write_text(parameterized_formula + "\n", encoding="utf-8")
    torch.save(
        {
            "kernel": torch.tensor(kernel),
            "reconstructed_kernel": torch.tensor(fit.kernel),
            "selected_freqs": fit.selected_freqs,
            "c0": fit.c0,
            "a": fit.a,
            "b": fit.b,
        },
        out_dir / "fourier_kernel_fit.pt",
    )

    lowering: dict[str, object] = {}
    if args.lower_dominant_term and fit.selected_freqs:
        dominant = max(fit.selected_freqs, key=lambda f: abs(fit.a[f]) + abs(fit.b[f]))
        term = f"Cos[(2*Pi*{dominant}*d)/{model.cfg.p}]"
        eml_expr = lower_to_eml(term, root)
        (out_dir / f"dominant_freq_{dominant}_cos.eml").write_text(eml_expr + "\n", encoding="utf-8")
        lowering = {
            "dominant_frequency": dominant,
            "lowered_term": term,
            "dominant_term_eml": eml_stats(eml_expr),
            "dominant_term_file": str(out_dir / f"dominant_freq_{dominant}_cos.eml"),
        }

    summary = {
        "checkpoint": {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "checkpoint_index": args.checkpoint_index,
            "path": str(checkpoint_path),
        },
        "model": {
            "p": model.cfg.p,
            "n_inputs": int(tokens.shape[0]),
            "loss": model_loss,
            "accuracy": model_acc,
        },
        "fourier_formula": {
            **formula_metrics,
            "approximation_error_vs_model_logits": approx_metrics,
            "formula_file": str(out_dir / "fourier_formula.wl"),
            "parameterized_formula_file": str(out_dir / "fourier_formula_parameterized.wl"),
        },
        "eml_lowering": lowering,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
