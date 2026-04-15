#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from grokking_eml.matrix_probe import build_matrix_probe_datasets, split_matrix_dataset, train_matrix_probe
from grokking_eml.matrix_probe import MatrixProbeDataset
from grokking_eml.model import OneLayerModularTransformer, load_full_run_data, make_all_mod_add_tokens


def parse_depths(value: str) -> list[int]:
    depths = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not depths:
        raise argparse.ArgumentTypeError("depth list cannot be empty")
    return depths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit one vector-valued EML tree per matrix/linear transform.")
    parser.add_argument("--repo-id", default="callummcdougall/grokking_full_run_data")
    parser.add_argument("--filename", default="full_run_data.pth")
    parser.add_argument("--checkpoint-index", type=int, default=400)
    parser.add_argument("--data-dir", default="large_files")
    parser.add_argument("--out-dir", default="runs/eml_matrix_probes")
    parser.add_argument("--dataset-cache", default="", help="Directory for cached matrix-node input/output tensors.")
    parser.add_argument("--refresh-cache", action="store_true", help="Recompute and overwrite cached matrix-node tensors.")
    parser.add_argument("--matrix-name", action="append", default=[], help="Run matrices whose name contains this string.")
    parser.add_argument("--max-probes", type=int, default=0)
    parser.add_argument("--depths", type=parse_depths, default=[2, 3])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--train-n", type=int, default=512)
    parser.add_argument("--test-n", type=int, default=512)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def save_probe_cache(cache_dir: Path, probes: list[MatrixProbeDataset], metadata: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_probes = []
    for probe in probes:
        path = cache_dir / f"{probe.name}.pt"
        torch.save(
            {
                "name": probe.name,
                "inputs": probe.inputs,
                "target": probe.target,
                "description": probe.description,
            },
            path,
        )
        manifest_probes.append(
            {
                "name": probe.name,
                "path": str(path),
                "input_shape": list(probe.inputs.shape),
                "target_shape": list(probe.target.shape),
                "description": probe.description,
            }
        )
    manifest = {**metadata, "probes": manifest_probes}
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_probe_cache(cache_dir: Path) -> tuple[list[MatrixProbeDataset], dict] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    probes = []
    for item in manifest["probes"]:
        payload = torch.load(item["path"], map_location="cpu", weights_only=False)
        probes.append(
            MatrixProbeDataset(
                name=payload["name"],
                inputs=payload["inputs"],
                target=payload["target"],
                description=payload["description"],
            )
        )
    return probes, manifest


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = root / args.data_dir
    out_dir = root / args.out_dir
    state_dir = out_dir / "states"
    cache_dir = root / args.dataset_cache if args.dataset_cache else out_dir / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    cached = None if args.refresh_cache else load_probe_cache(cache_dir)
    if cached is not None:
        probes, cache_manifest = cached
        checkpoint_info = cache_manifest["checkpoint"]
        print(f"Loaded {len(probes)} cached matrix probe datasets from {cache_dir}")
    else:
        checkpoint_path = hf_hub_download(repo_id=args.repo_id, filename=args.filename, local_dir=data_dir)
        full_run_data = load_full_run_data(checkpoint_path, map_location=args.device)
        model = OneLayerModularTransformer.from_state_dict(full_run_data["state_dicts"][args.checkpoint_index]).to(args.device)

        tokens, _ = make_all_mod_add_tokens(model.cfg.p, device=args.device)
        with torch.no_grad():
            logits_full, cache = model(tokens, return_cache=True)
            logits = logits_full[:, -1, : model.cfg.p]

        probes = build_matrix_probe_datasets(logits.cpu(), {k: v.cpu() for k, v in cache.items()}, model.cfg.p)
        checkpoint_info = {
            "repo_id": args.repo_id,
            "filename": args.filename,
            "checkpoint_index": args.checkpoint_index,
            "path": str(checkpoint_path),
        }
        save_probe_cache(
            cache_dir,
            probes,
            metadata={
                "checkpoint": checkpoint_info,
                "p": model.cfg.p,
                "n_probe_datasets": len(probes),
            },
        )
        print(f"Saved {len(probes)} matrix probe datasets to {cache_dir}")
    if args.matrix_name:
        probes = [probe for probe in probes if any(needle in probe.name for needle in args.matrix_name)]
    if args.max_probes > 0:
        probes = probes[: args.max_probes]

    rows = []
    best_by_matrix = {}
    for probe_idx, probe in enumerate(probes):
        x_train, y_train, x_test, y_test = split_matrix_dataset(
            probe, train_n=args.train_n, test_n=args.test_n, seed=args.seed + probe_idx
        )
        matrix_rows = []
        for depth in args.depths:
            candidates = []
            for restart in range(max(1, args.restarts)):
                state_path = state_dir / f"{probe.name}_d{depth}_r{restart}.pt"
                result = train_matrix_probe(
                    dataset=probe,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    depth=depth,
                    steps=args.steps,
                    lr=args.lr,
                    seed=args.seed + probe_idx * 1009 + depth * 101 + restart,
                    out_state_path=state_path,
                )
                candidates.append((restart, state_path, result))
            best_restart, state_path, result = min(candidates, key=lambda item: item[2].train_rmse_norm)
            row = {
                **result.to_dict(),
                "description": probe.description,
                "best_restart": best_restart,
                "state_file": str(state_path),
            }
            rows.append(row)
            matrix_rows.append(row)
            print(
                f"{probe.name} depth={depth}: rmse/baseline={result.rmse_baseline_ratio:.4g} "
                f"mean_r2={result.mean_output_r2:.4g} test_rmse_norm={result.test_rmse_norm:.4g}"
            )
        best_by_matrix[probe.name] = min(matrix_rows, key=lambda row: float(row["rmse_baseline_ratio"]))

    summary = {
        "checkpoint": checkpoint_info,
        "fit_config": {
            "depths": args.depths,
            "steps": args.steps,
            "lr": args.lr,
            "train_n": args.train_n,
            "test_n": args.test_n,
            "restarts": args.restarts,
            "seed": args.seed,
            "dataset_cache": str(cache_dir),
            "refresh_cache": args.refresh_cache,
        },
        "best_by_matrix": best_by_matrix,
        "results": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
