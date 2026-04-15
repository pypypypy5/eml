from __future__ import annotations

import importlib.util
from pathlib import Path


def load_eml_compiler(repo_root: str | Path):
    path = Path(repo_root) / "vendor" / "SymbolicRegressionPackage" / "EML_toolkit" / "EmL_compiler" / "eml_compiler_v4.py"
    if not path.exists():
        raise FileNotFoundError(f"EML compiler not found at {path}")
    spec = importlib.util.spec_from_file_location("eml_compiler_v4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import EML compiler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def lower_to_eml(expr: str, repo_root: str | Path) -> str:
    compiler = load_eml_compiler(repo_root)
    return compiler.eml_compile_from_string(expr)


def eml_stats(eml_expr: str) -> dict[str, int]:
    return {
        "chars": len(eml_expr),
        "eml_nodes": eml_expr.count("EML["),
        "terminal_ones": sum(1 for tok in eml_expr.replace("[", ",").replace("]", ",").split(",") if tok == "1"),
    }
