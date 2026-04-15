#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from grokking_eml.eml_probe import train_eml_probe
from grokking_eml.model import OneLayerModularTransformer, load_full_run_data, make_all_mod_add_tokens
from grokking_eml.node_probes import build_default_probe_datasets, split_probe_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit small EML trees to scalar internal-node probes.")
    parser.add_argument("--repo-id", default="callummcdougall/grokking_full_run_data")
    parser.add_argument("--filename", default="full_run_data.pth")
    parser.add_argument("--checkpoint-index", type=int, default=400)
    parser.add_argument("--data-dir", default="large_files")
    parser.add_argument("--out-dir", default="runs/eml_node_probes")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--train-n", type=int, default=512)
    parser.add_argument("--test-n", type=int, default=512)
    parser.add_argument("--max-probes", type=int, default=0, help="0 means run all default probes.")
    parser.add_argument("--probe-name", action="append", default=[], help="Run only probes whose name contains this string. Repeatable.")
    parser.add_argument("--restarts", type=int, default=1, help="Random EML initializations per probe; best train RMSE is reported.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    expr_dir = out_dir / "expressions"
    data_dir.mkdir(parents=True, exist_ok=True)
    expr_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=data_dir)
    full_run_data = load_full_run_data(checkpoint_path, map_location=args.device)
    state_dict = full_run_data["state_dicts"][args.checkpoint_index]
    model = OneLayerModularTransformer.from_state_dict(state_dict).to(args.device)

    tokens, labels = make_all_mod_add_tokens(model.cfg.p, device=args.device)
    with torch.no_grad():
        logits_full, cache = model(tokens, return_cache=True)
        logits = logits_full[:, -1, : model.cfg.p]

    probes = build_default_probe_datasets(tokens.cpu(), labels.cpu(), logits.cpu(), {k: v.cpu() for k, v in cache.items()}, model.cfg.p)
    if args.probe_name:
        probes = [p for p in probes if any(needle in p.name for needle in args.probe_name)]
    if args.max_probes > 0:
        probes = probes[: args.max_probes]

    results = []
    for i, probe in enumerate(probes):
        x_train, y_train, x_test, y_test = split_probe_dataset(
            probe, train_n=args.train_n, test_n=args.test_n, seed=args.seed + i
        )
        expr_path = expr_dir / f"{probe.name}.eml"
        candidates = []
        for restart in range(max(1, args.restarts)):
            result = train_eml_probe(
                name=probe.name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                depth=args.depth,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed + i * 1009 + restart,
                out_expr_path=expr_path if restart == 0 else None,
            )
            candidates.append((restart, result))
        best_restart, result = min(candidates, key=lambda item: item[1].train_rmse_norm)
        if best_restart != 0:
            # Re-run only to export the selected hard expression. Metrics come
            # from the already completed best run above.
            train_eml_probe(
                name=probe.name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                depth=args.depth,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed + i * 1009 + best_restart,
                out_expr_path=expr_path,
            )
        row = {
            **result.to_dict(),
            "description": probe.description,
            "expression_file": str(expr_path),
            "best_restart": best_restart,
        }
        results.append(row)
        print(
            f"{probe.name}: test_rmse_norm={result.test_rmse_norm:.4g} "
            f"r2={result.test_r2:.4g} rmse/baseline={result.rmse_baseline_ratio:.4g}"
        )

    summary = {
        "checkpoint": {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "checkpoint_index": args.checkpoint_index,
            "path": str(checkpoint_path),
        },
        "fit_config": {
            "depth": args.depth,
            "steps": args.steps,
            "lr": args.lr,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "seed": args.seed,
            "restarts": args.restarts,
        },
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
