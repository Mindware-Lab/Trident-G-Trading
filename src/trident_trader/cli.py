from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from trident_trader.core.entropy_mi_controller import ControllerInputs, EntropyMIController
from trident_trader.core.mismatch import (
    BurdenInputs,
    LoadInputs,
    compute_load,
    expected_burden,
    mismatch,
)
from trident_trader.core.policy_selector import PolicyContext, select_operator
from trident_trader.core.type2_gate import (
    Type2GateInputs,
    propose_heavy_tail_step,
    should_trigger_type2,
)
from trident_trader.core.zone_gate import ZoneInputs, select_zone
from trident_trader.features.lambda_world import compute_lambda
from trident_trader.features.mutual_info import summarize_mi
from trident_trader.features.regime import RegimeFeatures, classify_regime
from trident_trader.observability.metrics import update_tail_diagnostics
from trident_trader.risk.sizing import type1_risk_multiplier
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
        ts=datetime.now(tz=UTC),
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


def _load_snapshots(path: str) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append({k: float(v) for k, v in raw.items()})
    return out


def _cmd_control_cycle(args: argparse.Namespace) -> int:
    if not args.snapshots:
        print("No snapshots file. Use --snapshots PATH (JSONL).")
        return 2

    snapshots = _load_snapshots(args.snapshots)
    controller = EntropyMIController()

    mismatch_hist: list[float] = []
    mi_hist: list[float] = []
    update_sizes: list[float] = []
    big_steps: list[int] = []

    for idx, row in enumerate(snapshots):
        snap = WorldSnapshot(
            ts=datetime.now(tz=UTC),
            spread_bps=row["spread_bps"],
            depth_score=row["depth_score"],
            realized_vol=row["realized_vol"],
            vol_of_vol=row["vol_of_vol"],
            slippage_bps=row["slippage_bps"],
            latency_ms=row["latency_ms"],
            fee_bps=row["fee_bps"],
            news_intensity=row["news_intensity"],
            news_tone=row.get("news_tone", 0.0),
        )

        lambda_result = compute_lambda(snap)
        load_value = compute_load(
            LoadInputs(
                realized_vol=row["realized_vol"],
                drawdown_velocity=row.get("drawdown_velocity", 0.0),
                error_rate=row.get("error_rate", 0.0),
                slippage_spike=row.get("slippage_spike", 0.0),
            )
        )
        burden_value = expected_burden(
            BurdenInputs(
                forecast_vol=row.get("forecast_vol", row["realized_vol"]),
                expected_slippage=row.get("expected_slippage", row["slippage_bps"] / 10.0),
                event_risk=row.get("event_risk", row["news_intensity"] / 100.0),
            )
        )
        eps = mismatch(load_value, burden_value)
        mismatch_hist.append(eps)

        mi_proxy = max(0.0, row.get("signal_mi", 0.0) * 0.6 + row.get("operator_mi", 0.0) * 0.4)
        mi_hist.append(mi_proxy)
        mi_summary = summarize_mi(mi_hist)

        zone = select_zone(
            ZoneInputs(lambda_world=lambda_result.lambda_world, load=load_value, mismatch=abs(eps))
        )
        regime = classify_regime(
            RegimeFeatures(
                realized_vol=row["realized_vol"],
                vol_of_vol=row["vol_of_vol"],
                corr_spike=row.get("corr_spike", 0.0),
            )
        )

        ctrl = controller.step(
            ControllerInputs(
                policy_scores=[row.get("score_mr", 0.5), row.get("score_bo", 0.5)],
                signal_mi=row.get("signal_mi", 0.0),
                operator_mi=row.get("operator_mi", 0.0),
                mismatch_value=eps,
                lambda_world=lambda_result.lambda_world,
            )
        )

        operator = select_operator(
            PolicyContext(
                regime_label=regime.label, zone=zone, explore_pressure=ctrl.explore_pressure
            ),
            {"mean_reversion": row.get("score_mr", 0.5), "breakout": row.get("score_bo", 0.5)},
        )

        type2 = should_trigger_type2(
            Type2GateInputs(
                mismatch_history=mismatch_hist,
                mi_history=mi_hist,
                lambda_world=lambda_result.lambda_world,
            )
        )

        update_size = (
            abs(propose_heavy_tail_step(scale=0.04, seed=idx)) if type2 else abs(eps) * 0.02
        )
        update_sizes.append(update_size)
        if update_size >= 0.12:
            big_steps.append(idx)

        risk_mult = type1_risk_multiplier(
            zone=zone, lambda_world=lambda_result.lambda_world, mismatch_abs=abs(eps)
        )

        print(
            json.dumps(
                {
                    "step": idx,
                    "zone": zone,
                    "regime": regime.label,
                    "lambda": round(lambda_result.lambda_world, 4),
                    "mismatch": round(eps, 4),
                    "mi": round(mi_summary.value, 4),
                    "mode": ctrl.mode,
                    "operator": operator,
                    "type2": type2,
                    "risk_mult": round(risk_mult, 4),
                    "update_size": round(update_size, 5),
                }
            )
        )

    diag = update_tail_diagnostics(update_magnitudes=update_sizes, big_update_steps=big_steps)
    print(
        json.dumps(
            {
                "tail_exponent": round(diag.tail_exponent, 4),
                "interevent_mean": round(diag.interevent_mean, 4),
                "interevent_count": diag.interevent_count,
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

    control_cycle = sub.add_parser(
        "control-cycle", help="Run entropy-MI control cycle on snapshot JSONL"
    )
    control_cycle.add_argument("--snapshots", type=str, help="Path to JSONL snapshots")
    control_cycle.set_defaults(func=_cmd_control_cycle)

    paper = sub.add_parser("paper", help="Run paper trading loop")
    paper.set_defaults(func=_cmd_paper)

    backtest = sub.add_parser("backtest", help="Run backtests")
    backtest.add_argument("--smoke", action="store_true", help="Run minimal smoke backtest")
    backtest.set_defaults(func=_cmd_backtest)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    func: Callable[[argparse.Namespace], int] = args.func
    return int(func(args))


if __name__ == "__main__":
    raise SystemExit(main())

