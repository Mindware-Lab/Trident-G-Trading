from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

from trident_trader.backtest.data_loading import (
    generate_smoke_bars,
    load_events_from_universe,
    load_toml,
    parse_duration,
)
from trident_trader.backtest.simulation import run_simulation
from trident_trader.world.schemas import Bar, NewsEvent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timescales", default="configs/timescales.toml")
    parser.add_argument("--universe", default="configs/universes/four_streams.toml")
    parser.add_argument("--lambda-gate", dest="lambda_gate", default="configs/lambda_gate.toml")
    parser.add_argument("--news-config", default="configs/news_gdelt.toml")
    parser.add_argument("--news-file", default=None)
    parser.add_argument("--data-root", default=".")
    args = parser.parse_args()

    timescales_cfg = load_toml(Path(args.timescales))
    universe_cfg = load_toml(Path(args.universe))
    lambda_cfg = load_toml(Path(args.lambda_gate))
    news_cfg = load_toml(Path(args.news_config)) if Path(args.news_config).exists() else None

    streams_cfg = cast(list[dict[str, Any]], universe_cfg["streams"])
    symbols = [cast(str, item["symbol"]) for item in streams_cfg]
    periods = {
        "fast": parse_duration(
            cast(str, cast(dict[str, object], timescales_cfg["timescales"])["fast"])
        ),
        "medium": parse_duration(
            cast(str, cast(dict[str, object], timescales_cfg["timescales"])["medium"])
        ),
        "slow": parse_duration(
            cast(str, cast(dict[str, object], timescales_cfg["timescales"])["slow"])
        ),
    }
    base_period = parse_duration(
        cast(str, cast(dict[str, object], timescales_cfg["clock"])["base_resolution"])
    )

    events: list[Bar | NewsEvent]
    if args.smoke:
        events = generate_smoke_bars(symbols=symbols, base_period=base_period)
    else:
        events = load_events_from_universe(
            streams_cfg=streams_cfg,
            data_root=Path(args.data_root),
            news_cfg=news_cfg,
            news_file_override=cast(str | None, args.news_file),
        )

    result = run_simulation(
        events=events,
        symbols=symbols,
        periods=periods,
        lambda_cfg=lambda_cfg,
    )
    decisions = result.decisions
    stats = result.stats
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
