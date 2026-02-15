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

--

# Trident-G-Trading — Trident Trader (Λ-Gated)

This repo implements **Trident Trader**: a Trident-G inspired, event-driven trading agent that treats **markets + news** as a “world” it adapts to.  
Core idea: only engage risk when **Λ (Lambda) indicates the world is learnable/clean**, then choose actions via an **entropy ↔ mutual-information** explore/exploit controller, refined with **relational graph + successor map (SR)** dynamics.

> ⚠️ Trading is risky. Backtests can look good and still fail live. This repo is designed to make results falsifiable and to enforce strong anti-overfitting discipline.

---

## 0) Naming & mental model

- **Repo name:** Trident-G-Trading  
- **Runtime agent name:** **Trident Trader** (or **Λ-Gated Trident Trader**)
- **World:** `Bar | NewsEvent` event stream
- **Decision clock:** on **medium bar close**
- **Safety gate:** engage only when **Λ** is sufficiently high (plus k-of-n arming)
- **Policy:** entropy–MI operator selection, tempered by SR uncertainty  
- **Maps:** relational graph state → SR (successor representation) uncertainty & transitions

---

## 1) Start with a “clean learnability” 4-market universe (Databento)

Goal: pick instruments that are (a) deep and regulated, (b) diverse in drivers, (c) not dominated by single-venue “whale” behaviour, and (d) informative across regimes so Λ gating is meaningful.

### Recommended v1 universe (4 streams)

**Option A (cleanest, balanced macro drivers)**
- **MES** (Micro E-mini S&P 500) — equity risk
- **ZN** (10Y US Treasury Note) — rates risk
- **6E** (EUR/USD futures) — FX macro risk
- **GC or MGC** (Gold / Micro Gold) — risk-off / inflation hedge

**Option B (more event-heavy, good stress-test for Λ+news)**
- Replace **Gold** with **CL** (Crude Oil)

> Practical note: If your early tests show Λ constantly disarming around crude-driven event spikes, that is not a failure. It is the system refusing to trade “messy” conditions.

### Why this mapping fits Λ logic

- Cross-asset diversity makes relational structure meaningful (graph motifs/coupling).
- Deep liquidity reduces microstructure artefacts and “phantom edges”.
- Different “news sensitivity” profiles make event_intensity_z a real control signal.

---

## 2) Backtesting phase (Databento historical)

### 2.1 Prerequisites

- Create a Databento account and generate an API key.
- Set it in your terminal session (do **not** commit keys):

```powershell
$env:DATABENTO_API_KEY="db-REPLACE_WITH_REAL_KEY"
````

Optional: estimate cost before downloading via Databento metadata calls (preferred).

### 2.2 Download minimal “probe” first (confirm entitlements + costs)

```powershell
python scripts/download_databento_ohlcv.py `
  --start 2024-01-01 `
  --end 2024-01-03 `
  --symbols MES.v.0 ZN.v.0 6E.v.0 GC.v.0 `
  --skip-download
```

> `.v.0` means continuous front-month by roll rule. Use the same form for CL if you choose Option B.

### 2.3 Download your first research window (example)

```powershell
python scripts/download_databento_ohlcv.py `
  --start 2022-01-01 `
  --end 2024-01-01 `
  --symbols MES.v.0 ZN.v.0 6E.v.0 GC.v.0 `
  --out-dir data/raw/databento/GLBX.MDP3/ohlcv-1m
```

Verify files exist:

```powershell
Get-ChildItem data/raw/databento/GLBX.MDP3/ohlcv-1m
```

### 2.4 Run walk-forward (frozen config)

```powershell
python scripts/run_walkforward.py `
  --universe configs/universes/four_streams_databento.toml `
  --output-dir reports/walkforward/databento_4stream `
  --train-days 180 --test-days 30 --step-days 30
```

Analyse:

```powershell
python scripts/analyze_walkforward.py --input-dir reports/walkforward/databento_4stream
```

---

## 3) What counts as “profitability evidence” in this repo

Profit is only meaningful if it survives:

* realistic costs (spread/slippage/fees)
* strict out-of-sample evaluation (walk-forward)
* stability across folds and environments

### 3.1 Primary outputs to review (each run)

* `reports/.../analysis_summary.json`
* `reports/.../fold_summary.csv`
* `reports/.../decision_log.csv`
* binned diagnostics (Λ bins, SR uncertainty bins, τ/temperature bins)

### 3.2 Minimum “go/no-go” gates (before paper trading)

**Gate A — Mechanism sanity**

* The system trades less in low-Λ periods
* PnL contribution is stronger in high-Λ bins than low-Λ bins
* SR uncertainty spikes around regime shifts, not randomly
* Entropy–MI temperature τ increases with uncertainty (explore) and settles when learnable (exploit)

**Gate B — Out-of-sample robustness**

* Fold-to-fold performance is not dominated by one short window
* Drawdowns are controlled and not “one trade away” from ruin
* Performance does not vanish when you slightly perturb thresholds (small sensitivity checks)

**Gate C — Anti-overfitting discipline**

* Maintain an ablation ladder (below) and a strict holdout set you do not touch.

---

## 4) Ablation ladder (required before any “victory” claims)

Run the same universe and walk-forward splits, changing one component at a time:

1. Baseline operator set without Λ gating
2. * Λ gating only
3. * news intensity integration
4. * entropy–MI selector
5. * relational graph state (motifs/coupling)
6. * successor map + SR uncertainty-tempered τ
7. Full stack

Success criterion: improvements show up primarily as **robustness** (lower drawdown, better survival across regimes), not just higher mean return.

---

## 5) Move to live safely: Shadow → Paper → Micro live

### 5.1 Minimum live architecture (services)

You already have the “brain”. Add production plumbing:

* **MarketData service**

  * Live bars (and optional news) → canonical `Bar | NewsEvent`

* **Strategy service**

  * Runs on medium close → outputs *intent* (target position, stops, exits)

* **Risk service (non-negotiable)**

  * Hard limits: max position per symbol, max daily loss, max leverage/margin use, max order rate
  * Kill switch: flatten + disarm if breached (or if Λ disarms)

* **Execution/OMS service**

  * Intent → broker orders, tracks fills/rejects, reconciles broker state

* **Monitoring + artefacts**

  * Persist every decision/gate input/order/fill
  * Alerts on disconnects, rejects, slippage spikes, drawdown, data gaps, kill switch events

### 5.2 Deployment ladder (do not skip steps)

**Step 1 — Shadow mode (live data, no orders)**

* Run the full agent on live feed
* Log decisions
* Compare behaviour to backtest expectations by Λ/SR bins

**Step 2 — Paper trading (IBKR)**

* Use IBKR paper account via API
* Treat as execution + failure-mode testing, not “proof of edge”

**Step 3 — Micro live (strict limits)**

* 1 micro contract, one instrument first
* Daily loss limit + hard flatten switch
* Scale only after weeks of stable behaviour

**Step 4 — Scale gradually**

* Increase size OR number of instruments, never both at once

---

## 6) Tooling plan: Databento for market data, IBKR for execution

### Databento (market data)

* Historical replay for research
* Live feed and intraday replay for shadow testing

### IBKR (broker execution)

* TWS API / IB Gateway for automated order placement and monitoring
* Paper trading account for API tests in a simulated environment
* Respect margin and liquidation risk, especially for futures

---

## 7) Concrete “next build” tasks in this repo

### Backtest hygiene (before any live trading)

* Ensure the execution model is conservative (spread + slippage + fees)
* Save run artefacts for reproducibility (configs, fold summaries, full decision logs)
* Add a “locked holdout” period and a parameter freeze protocol

### Live modules to implement

* `src/trident_trader/live/databento_live.py` (bars + intraday replay wiring)
* `src/trident_trader/brokers/ibkr.py` (connect, place/cancel, positions/fills, reconcile)
* `src/trident_trader/risk/limits.py` (hard limits + kill switch)
* `scripts/run_live_shadow.py`
* `scripts/run_live_paper.py`

---

## 8) Operational controls (how not to blow up unattended)

* Pre-trade checks: data integrity, connectivity, positions flat at start, limits loaded
* Change control: configs are versioned, “frozen config” per live release
* Continuous monitoring: alerts + daily reports + incident log
* Post-trade review: slippage, rejects, limit breaches, Λ/uncertainty anomalies
* Kill switch always available and tested

---

## Appendix: Quickstart commands (one place)

### Databento download (example)

```powershell
pip install -U databento pandas pyarrow
$env:DATABENTO_API_KEY="db-REPLACE_WITH_REAL_KEY"

python scripts/download_databento_ohlcv.py `
  --start 2022-01-01 `
  --end 2024-01-01 `
  --symbols MES.v.0 ZN.v.0 6E.v.0 GC.v.0 `
  --out-dir data/raw/databento/GLBX.MDP3/ohlcv-1m
```

### Walk-forward + analysis

```powershell
python scripts/run_walkforward.py `
  --universe configs/universes/four_streams_databento.toml `
  --output-dir reports/walkforward/databento_4stream `
  --train-days 180 --test-days 30 --step-days 30

python scripts/analyze_walkforward.py --input-dir reports/walkforward/databento_4stream
```

```

**Sources used (for the plan’s external claims):**  
- Databento GLBX.MDP3 dataset overview and “same API for real-time and historical”. :contentReference[oaicite:0]{index=0}  
- Databento Live API intraday replay. :contentReference[oaicite:1]{index=1}  
- Databento continuous contract symbology (`ES.v.0` pattern) and symbology docs. :contentReference[oaicite:2]{index=2}  
- Databento metered pricing and `metadata.get_cost`. :contentReference[oaicite:3]{index=3}  
- IBKR TWS API page (API use + paper trading) and paper trading glossary. :contentReference[oaicite:4]{index=4}  
- IBKR note that paper trading access follows an actual funded account. :contentReference[oaicite:5]{index=5}  
- IBKR futures margin requirements and liquidation/exposure-fee cautions. :contentReference[oaicite:6]{index=6}  
- FCA “algorithmic trading controls” high-level observations (governance/testing/risk controls/monitoring emphasis). :contentReference[oaicite:7]{index=7}
::contentReference[oaicite:8]{index=8}
```

