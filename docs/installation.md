# Installation

## From PyPI

```bash
pip install pmf-acls
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add pmf-acls
```

## Optional: JAX acceleration

For GPU-accelerated solvers, install JAX separately:

```bash
pip install pmf-acls jax jaxlib
```

Then pass `solver="jax"` or `solver="jax_sparse"` to `pmf()`.

## Requirements

- Python ≥ 3.9
- NumPy, SciPy (installed automatically)
- Matplotlib (for plotting helpers)

## Development install

```bash
git clone <repo-url>
cd pmf
pip install -e ".[dev]"
```
