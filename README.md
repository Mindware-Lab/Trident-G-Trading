# trident-trader

Python-first trading research and execution scaffold aligned to Trident-G/G-Loop:
World -> Lambda + mismatch -> Zone gate -> Operators -> Evidence banking.

## Quick start

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
pre-commit install
trident-trader --help
```

## Layout

- `src/trident_trader/`: core package (src-layout)
- `operators_bank/`: promoted operators + evidence manifests
- `configs/`: environment/venue/universe config files
- `data/`: local-only raw/interim/processed storage
- `docs/`: design + runbooks
