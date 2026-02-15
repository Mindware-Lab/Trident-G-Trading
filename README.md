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

# Trident G Trading Platform — System Architecture (High-Level)

This repository implements an event-driven, reproducible trading research and execution stack aligned to Trident-G control principles (learnability-gated risk, explore/exploit control, relational world modelling, and evidence banking).

---

## Architecture Overview

### 1) Event-Driven Core
- One chronological event loop processes **market bars** and **news events**.
- Multi-timeframe consolidators build **fast / medium / slow** views from base data.
- Decision clock runs on **medium-bar close** to reduce lookahead risk.

### 2) World Layer (Data Ingestion + Canonical Schemas)
- Adapters/loaders normalise heterogeneous feeds (market/news) into shared event types.
- Canonical `Bar` and `NewsEvent` objects keep backtest/live interfaces aligned.

### 3) Storage + Reproducibility
- Local-first raw/processed data pipeline (**CSV/Parquet + config-driven runs**).
- Frozen run configs + artefact logs for **replayable experiments**.

### 4) Feature & World-Model Layer
- Rolling microstructure/regime features (spread, volume, volatility behaviour, integrity checks).
- News intensity enters as `event_intensity_z`.
- Produces **Λ(t)**: information quality / learnability score.

### 5) Trident-G Control Layer
- Zone Gate maps **Λ + stress/mismatch signals** to **Reset / Light / Full**.
- Purpose: engage risk only when market conditions are sufficiently **learnable/clean**.

### 6) Explore/Exploit Policy Layer
- **Entropy–MI controller**:
  - Entropy side controls exploration temperature (`tau`).
  - Mutual information side checks whether signals/operators remain outcome-relevant.
- Operator choice is **stochastic** but constrained by relevance.

### 7) Relational Cognitive Map Layer
- Builds a dynamic graph across the **4 streams** (coupling structure).
- Compresses into relational state features.
- Successor representation (SR) learns expected future relational states.
- SR uncertainty modulates exploration pressure.

### 8) Execution + Portfolio + Risk
- Simulated fill model with spread/slippage realism.
- Portfolio accounting (equity, PnL, exposure, turnover proxy).
- Risk limits / kill-switch behaviours integrated with controller states.

### 9) Backtesting & Validation Framework
- Walk-forward harness (train/test rolling windows) for out-of-sample evaluation.
- Profitability pack: return, CAGR, vol, Sharpe, max drawdown, exposure/turnover.
- Diagnostics by **Λ bins** and **SR/temperature** to test mechanism validity.

### 10) Evidence Banking Discipline
- Operators are promoted only with portability evidence (swap/boundary/delay checks).
- Separation between strategy code and banked evidence enforces anti-overfitting governance.

---

## Repo Intent (at a glance)
- **Backtest ↔ Live symmetry** via canonical events + shared execution interface.
- **Reproducibility by default** via frozen configs + artefact logging.
- **Mechanism testing** (Λ gating, SR uncertainty, entropy–MI control) alongside profitability metrics.
- **Governance** via evidence banking separation (anti-overfitting discipline).
