# EML Grokking Reproduction

This workspace wires together two public artifacts:

- Andrzej Odrzywolek's `VA00/SymbolicRegressionPackage`, cloned under `vendor/`, for the EML compiler and PyTorch EML-tree reference code from arXiv `2603.21852`.
- The public modular-addition grokking run `callummcdougall/grokking_full_run_data`, used in the Neel Nanda / ARENA mechanistic-interpretability notebooks.

The practical experiment is:

1. Load the published transformer checkpoint weights directly from `full_run_data.pth`.
2. Replay the model on all `113 * 113` modular-addition inputs.
3. Fit the model's final logits with the known mechanistic form
   `c0 + sum_f a_f cos(2*pi*f*(x+y-z)/113) + b_f sin(2*pi*f*(x+y-z)/113)`.
4. Use the paper author's EML compiler to lower a representative Fourier term into a pure EML tree.

This is intentionally not a blind EML-tree search over the entire classifier. The EML paper's PyTorch trainer is reliable for shallow elementary targets, while modular addition is a discrete classification table. The reproducible bridge here is to recover the elementary Fourier formula that the grokked model uses, then lower that formula into EML.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-torch-cpu.txt
pip install -r requirements.txt
```

`torch` must be installed from the CPU-only PyTorch wheel index on machines
without an NVIDIA GPU. Installing plain `torch` from PyPI on Linux can pull CUDA
runtime packages that are not needed for this experiment.

## Run

```bash
PYTHONPATH=src .venv/bin/python scripts/run_reproduction.py --lower-dominant-term
```

Outputs are written to `runs/modular_addition_eml/`:

- `summary.json`: losses, accuracy, selected frequencies, and EML lowering stats.
- `fourier_formula.wl`: fitted elementary formula with numeric coefficients.
- `fourier_formula_parameterized.wl`: same formula with symbolic coefficient terminals.
- `dominant_freq_*_cos.eml`: pure EML lowering of one dominant cosine term.
- `eml_validation.txt`: numerical spot-check of the lowered EML term.

## Current Result

The run in this workspace used checkpoint index `400` from
`callummcdougall/grokking_full_run_data/full_run_data.pth`.

- Replayed model: accuracy `1.0`, cross entropy `2.4122e-7`.
- Five-frequency Fourier formula: frequencies `[14, 35, 41, 42, 52]`,
  accuracy `1.0`, cross entropy `4.3975e-8`.
- EML lowering example: `Cos[(2*Pi*42*d)/113]` became a pure EML expression
  with `5729` EML nodes, written to
  `runs/modular_addition_eml/dominant_freq_42_cos.eml`.
