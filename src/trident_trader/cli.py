from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from trident_trader.features.lambda_world import compute_lambda
from trident_trader.world.adapters.gdelt import aggregate_intensity, load_gdelt_events
from trident_trader.world.schemas import WorldSnapshot


def _cmd_ingest(args: argparse.Namespace) -> int:
    if not args.gdelt_file:
        print("No source provided. Use --gdelt-file PATH")
        return 2

    events = load_gdelt_events(args.gdelt_file)
    intensity = aggregate_intensity(events)
    print(
        json.dumps(
            {
                "source": args.gdelt_file,
                "events": len(events),
                "news_intensity": round(intensity, 4),
            },
            indent=2,
        )
    )
    return 0


def _cmd_compute_lambda(args: argparse.Namespace) -> int:
    snapshot = WorldSnapshot(
        ts=datetime.now(tz=timezone.utc),
        spread_bps=args.spread_bps,
        depth_score=args.depth_score,
        realized_vol=args.realized_vol,
        vol_of_vol=args.vol_of_vol,
        slippage_bps=args.slippage_bps,
        latency_ms=args.latency_ms,
        fee_bps=args.fee_bps,
        news_intensity=args.news_intensity,
        news_tone=args.news_tone,
    )
    result = compute_lambda(snapshot)
    print(
        json.dumps(
            {
                "liquidity": round(result.liquidity, 4),
                "regime": round(result.regime, 4),
                "friction": round(result.friction, 4),
                "news": round(result.news, 4),
                "lambda_world": round(result.lambda_world, 4),
            },
            indent=2,
        )
    )
    return 0


def _cmd_paper(_: argparse.Namespace) -> int:
    print("Paper stub: run paper session with zone gate controls.")
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    mode = "smoke" if args.smoke else "full"
    print(f"Backtest stub: running {mode} mode.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trident-trader")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Run world ingestion adapters")
    ingest.add_argument("--gdelt-file", type=str, help="Path to GDELT CSV/TSV export")
    ingest.set_defaults(func=_cmd_ingest)

    compute_lambda_cmd = sub.add_parser("compute-lambda", help="Compute Lambda(t)")
    compute_lambda_cmd.add_argument("--spread-bps", type=float, default=1.2)
    compute_lambda_cmd.add_argument("--depth-score", type=float, default=0.8)
    compute_lambda_cmd.add_argument("--realized-vol", type=float, default=0.9)
    compute_lambda_cmd.add_argument("--vol-of-vol", type=float, default=0.4)
    compute_lambda_cmd.add_argument("--slippage-bps", type=float, default=0.8)
    compute_lambda_cmd.add_argument("--latency-ms", type=float, default=25.0)
    compute_lambda_cmd.add_argument("--fee-bps", type=float, default=0.5)
    compute_lambda_cmd.add_argument("--news-intensity", type=float, default=8.0)
    compute_lambda_cmd.add_argument("--news-tone", type=float, default=0.0)
    compute_lambda_cmd.set_defaults(func=_cmd_compute_lambda)

    paper = sub.add_parser("paper", help="Run paper trading loop")
    paper.set_defaults(func=_cmd_paper)

    backtest = sub.add_parser("backtest", help="Run backtests")
    backtest.add_argument("--smoke", action="store_true", help="Run minimal smoke backtest")
    backtest.set_defaults(func=_cmd_backtest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
