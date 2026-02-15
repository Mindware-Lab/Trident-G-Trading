from __future__ import annotations

import argparse
import random
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from trident_trader.backtest.engine import BacktestEngine
from trident_trader.backtest.metrics import summarize
from trident_trader.execution.oms import OrderIntent, SimulatedOMS
from trident_trader.portfolio.book import PortfolioBook
from trident_trader.risk.limits import RiskLimits, RiskState, check_order
from trident_trader.world.loaders.csv_bars import iter_csv_bars, merge_sorted
from trident_trader.world.schemas import Bar


def _parse_duration(value: str) -> timedelta:
    unit = value[-1]
    count = int(value[:-1])
    if unit == "m":
        return timedelta(minutes=count)
    if unit == "h":
        return timedelta(hours=count)
    if unit == "d":
        return timedelta(days=count)
    raise ValueError(f"Unsupported duration: {value}")


def _load_toml(path: Path) -> dict[str, object]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _generate_smoke_bars(symbols: list[str], base_period: timedelta, steps: int = 400) -> list[Bar]:
    rng = random.Random(7)
    start = datetime(2025, 1, 6, 0, 0, tzinfo=UTC)

    bars: list[Bar] = []
    base_prices = {symbol: 100.0 + (idx * 20.0) for idx, symbol in enumerate(symbols)}

    for step in range(steps):
        ts = start + (step + 1) * base_period
        for symbol in symbols:
            drift = rng.uniform(-0.003, 0.003)
            vol = rng.uniform(0.001, 0.008)
            prev = base_prices[symbol]
            close = max(1.0, prev * (1.0 + drift))
            high = max(prev, close) * (1.0 + vol)
            low = min(prev, close) * (1.0 - vol)
            spread = close * rng.uniform(0.00002, 0.00008)
            bid = close - spread / 2.0
            ask = close + spread / 2.0
            volume = rng.uniform(200.0, 2000.0)

            bars.append(
                Bar(
                    ts=ts,
                    symbol=symbol,
                    open=prev,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    bid=bid,
                    ask=ask,
                )
            )
            base_prices[symbol] = close

    bars.sort(key=lambda b: b.ts)
    return bars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timescales", default="configs/timescales.toml")
    parser.add_argument("--universe", default="configs/universes/four_streams.toml")
    parser.add_argument("--lambda-gate", dest="lambda_gate", default="configs/lambda_gate.toml")
    parser.add_argument(
        "--data-root",
        default=".",
        help="Root folder for relative data_file entries in universe config.",
    )
    args = parser.parse_args()

    timescales_cfg = _load_toml(Path(args.timescales))
    universe_cfg = _load_toml(Path(args.universe))
    lambda_cfg = _load_toml(Path(args.lambda_gate))

    streams_cfg = cast(list[dict[str, Any]], universe_cfg["streams"])
    symbols = [cast(str, item["symbol"]) for item in streams_cfg]
    periods = {
        "fast": _parse_duration(timescales_cfg["timescales"]["fast"]),
        "medium": _parse_duration(timescales_cfg["timescales"]["medium"]),
        "slow": _parse_duration(timescales_cfg["timescales"]["slow"]),
    }
    base_period = _parse_duration(timescales_cfg["clock"]["base_resolution"])

    decisions: list[dict[str, object]] = []
    book = PortfolioBook(initial_cash=1_000_000.0)
    oms = SimulatedOMS(slippage_bps=0.6, fee_bps=0.2)
    limits = RiskLimits(
        max_notional_per_order=250_000.0, max_daily_loss=20_000.0, max_gross_notional=1_000_000.0
    )
    risk_state = RiskState()
    spread_samples: list[float] = []
    slippage_samples: list[float] = []

    def _on_decision(ctx: dict[str, object]) -> None:
        gate = cast(dict[str, object], ctx["gate"])
        ts = cast(datetime, ctx["ts"])
        medium_bars = cast(dict[str, Bar], ctx["medium_bars"])

        # Simple baseline policy for infrastructure validation:
        # armed -> hold +1 unit each stream, disarmed -> flat.
        for symbol in symbols:
            target_qty = 1.0 if cast(bool, gate["armed"]) else 0.0
            current_qty = book.qty(symbol)
            delta = target_qty - current_qty
            if abs(delta) < 1e-12:
                continue

            side = "buy" if delta > 0 else "sell"
            qty = abs(delta)
            bar = medium_bars[symbol]

            intent = OrderIntent(symbol=symbol, side=side, qty=qty)
            preview_fill = oms.execute(intent, bar)

            gross_after = 0.0
            for sym in symbols:
                q = book.qty(sym)
                if sym == symbol:
                    q += delta
                gross_after += abs(q * medium_bars[sym].close)

            reduce_only = abs(target_qty) < abs(current_qty)
            allowed = check_order(
                limits=limits,
                state=risk_state,
                order_notional=preview_fill.notional,
                gross_notional_after=gross_after,
                daily_pnl=book.daily_pnl(ts),
                reduce_only=reduce_only,
            )
            if not allowed:
                continue

            book.apply_fill(preview_fill, ts=ts)
            spread_samples.append(preview_fill.spread_bps)
            slippage_samples.append(preview_fill.slippage_bps)

        prices = {sym: medium_bars[sym].close for sym in symbols}
        equity = book.mark_to_market(prices, ts=ts)

        decisions.append(
            {
                "ts": str(ts),
                "armed": cast(bool, gate["armed"]),
                "lambda_global": round(cast(float, gate["lambda_global"]), 4),
                "good_streams": cast(int, gate["good_streams"]),
                "equity": round(equity, 2),
                "daily_pnl": round(book.daily_pnl(ts), 2),
            }
        )

    engine = BacktestEngine(
        symbols=symbols,
        periods=periods,
        lambda_cfg=lambda_cfg,
        on_decision=_on_decision,
    )

    if args.smoke:
        bars = _generate_smoke_bars(symbols=symbols, base_period=base_period)
    else:
        data_root = Path(args.data_root)
        streams: list[list[Bar]] = []
        for stream in streams_cfg:
            symbol = cast(str, stream["symbol"])
            data_file = cast(str | None, stream.get("data_file"))
            if not data_file:
                raise ValueError(
                    f"Universe stream for {symbol} is missing required data_file field."
                )
            path = data_root / data_file
            if not path.exists():
                raise FileNotFoundError(f"Missing data file for {symbol}: {path}")
            streams.append(list(iter_csv_bars(path=path, symbol=symbol)))
        bars = list(merge_sorted(streams))

    engine.run(bars)

    stats = summarize(
        equity_curve=book.equity_curve,
        turnover=book.turnover,
        spread_samples=spread_samples,
        slippage_samples=slippage_samples,
    )

    armed_steps = sum(1 for d in decisions if cast(bool, d["armed"]))
    print(
        "Backtest complete: "
        f"decisions={len(decisions)} "
        f"armed={armed_steps} "
        f"rate={(armed_steps / max(1, len(decisions))):.2%} "
        f"return={stats.total_return:.2%} "
        f"max_dd={stats.max_drawdown:.2%} "
        f"turnover={stats.turnover:.2f} "
        f"avg_spread_bps={stats.avg_spread_bps:.3f} "
        f"avg_slippage_bps={stats.avg_slippage_bps:.3f}"
    )


if __name__ == "__main__":
    main()
