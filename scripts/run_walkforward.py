from __future__ import annotations

import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from trident_trader.backtest.data_loading import (
    generate_smoke_bars,
    load_events_from_universe,
    load_toml,
    parse_duration,
)
from trident_trader.backtest.simulation import run_simulation
from trident_trader.backtest.walkforward import (
    build_walkforward_windows,
    config_fingerprint,
    fold_window_to_dict,
    split_events_for_window,
)
from trident_trader.world.schemas import Bar, NewsEvent


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timescales", default="configs/timescales.toml")
    parser.add_argument("--universe", default="configs/universes/four_streams.toml")
    parser.add_argument("--lambda-gate", dest="lambda_gate", default="configs/lambda_gate.toml")
    parser.add_argument("--news-config", default="configs/news_gdelt.toml")
    parser.add_argument("--news-file", default=None)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output-dir", default="reports/walkforward/latest")
    parser.add_argument("--train-days", type=int, default=30)
    parser.add_argument("--test-days", type=int, default=7)
    parser.add_argument("--step-days", type=int, default=7)
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
        # ~10+ days at 1m base bars to ensure day-based walk-forward windows exist.
        events = generate_smoke_bars(symbols=symbols, base_period=base_period, steps=15000)
    else:
        events = load_events_from_universe(
            streams_cfg=streams_cfg,
            data_root=Path(args.data_root),
            news_cfg=news_cfg,
            news_file_override=cast(str | None, args.news_file),
        )

    if not events:
        raise RuntimeError("No events loaded.")

    windows = build_walkforward_windows(
        event_start=events[0].ts,
        event_end=events[-1].ts,
        train_period=timedelta(days=args.train_days),
        test_period=timedelta(days=args.test_days),
        step_period=timedelta(days=args.step_days),
    )
    if not windows:
        raise RuntimeError(
            "No walk-forward windows produced. Increase data span or reduce window sizes."
        )

    output_dir = Path(args.output_dir)
    config_bundle: dict[str, object] = {
        "timescales": timescales_cfg,
        "universe": universe_cfg,
        "lambda_gate": lambda_cfg,
        "news": news_cfg if news_cfg is not None else {},
        "walkforward": {
            "train_days": args.train_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "smoke": args.smoke,
        },
    }
    fingerprint = config_fingerprint(config_bundle)
    _write_json(
        output_dir / "frozen_config.json",
        {"fingerprint": fingerprint, "config": config_bundle},
    )

    fold_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    for window in windows:
        train_events, test_events = split_events_for_window(events, window)
        if not train_events or not test_events:
            continue

        combined = train_events + test_events
        combined.sort(key=lambda e: e.ts)
        sim = run_simulation(
            events=combined,
            symbols=symbols,
            periods=periods,
            lambda_cfg=lambda_cfg,
        )

        test_decisions = [
            d for d in sim.decisions if window.test_start <= cast(object, d["ts"]) < window.test_end
        ]
        if not test_decisions:
            continue

        for d in test_decisions:
            decision_rows.append(
                {
                    "fold_index": window.fold_index,
                    "ts": cast(object, d["ts"]).isoformat(),
                    "armed": d["armed"],
                    "lambda_global": round(cast(float, d["lambda_global"]), 6),
                    "operator": d["operator"],
                    "mi_score": round(cast(float, d["mi_score"]), 6),
                    "temperature": round(cast(float, d["temperature"]), 6),
                    "sr_uncertainty": round(cast(float, d["sr_uncertainty"]), 6),
                    "equity": round(cast(float, d["equity"]), 6),
                    "daily_pnl": round(cast(float, d["daily_pnl"]), 6),
                }
            )

        fold_rows.append(
            {
                **fold_window_to_dict(window),
                "events_train": len(train_events),
                "events_test": len(test_events),
                "decisions_test": len(test_decisions),
                "armed_rate_test": round(
                    sum(1 for d in test_decisions if cast(bool, d["armed"])) / len(test_decisions),
                    6,
                ),
                "return_total": round(sim.stats.total_return, 8),
                "max_drawdown": round(sim.stats.max_drawdown, 8),
                "turnover": round(sim.stats.turnover, 6),
                "avg_spread_bps": round(sim.stats.avg_spread_bps, 6),
                "avg_slippage_bps": round(sim.stats.avg_slippage_bps, 6),
            }
        )

    _write_rows_csv(output_dir / "fold_summary.csv", fold_rows)
    _write_rows_csv(output_dir / "decision_log.csv", decision_rows)
    _write_json(
        output_dir / "run_metadata.json",
        {
            "fingerprint": fingerprint,
            "folds_total": len(windows),
            "folds_completed": len(fold_rows),
            "decisions_logged": len(decision_rows),
        },
    )
    print(
        f"Walk-forward complete: folds={len(fold_rows)}/{len(windows)} "
        f"decisions={len(decision_rows)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
