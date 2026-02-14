from __future__ import annotations

import argparse
import random
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

from trident_trader.backtest.engine import BacktestEngine
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
    args = parser.parse_args()

    timescales_cfg = _load_toml(Path(args.timescales))
    universe_cfg = _load_toml(Path(args.universe))
    lambda_cfg = _load_toml(Path(args.lambda_gate))

    symbols = [item["symbol"] for item in universe_cfg["streams"]]
    periods = {
        "fast": _parse_duration(timescales_cfg["timescales"]["fast"]),
        "medium": _parse_duration(timescales_cfg["timescales"]["medium"]),
        "slow": _parse_duration(timescales_cfg["timescales"]["slow"]),
    }
    base_period = _parse_duration(timescales_cfg["clock"]["base_resolution"])

    decisions: list[dict[str, object]] = []

    def _on_decision(ctx: dict[str, object]) -> None:
        gate = ctx["gate"]
        decisions.append(
            {
                "ts": str(ctx["ts"]),
                "armed": gate["armed"],
                "lambda_global": round(gate["lambda_global"], 4),
                "good_streams": gate["good_streams"],
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
        bars = _generate_smoke_bars(symbols=symbols, base_period=base_period, steps=1200)

    engine.run(bars)

    armed_steps = sum(1 for d in decisions if d["armed"])
    print(
        f"Backtest complete: decisions={len(decisions)} armed={armed_steps} "
        f"rate={(armed_steps / max(1, len(decisions))):.2%}"
    )


if __name__ == "__main__":
    main()
