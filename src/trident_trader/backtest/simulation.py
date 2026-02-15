from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

from trident_trader.backtest.engine import BacktestEngine
from trident_trader.backtest.metrics import BacktestStats, summarize
from trident_trader.core.operator_selector_entropy_mi import OperatorSelectorEntropyMI
from trident_trader.execution.oms import OrderIntent, SimulatedOMS
from trident_trader.features.relational_graph import RelationalGraphMap
from trident_trader.features.successor_map import SuccessorMap
from trident_trader.portfolio.book import PortfolioBook
from trident_trader.risk.limits import RiskLimits, RiskState, check_order
from trident_trader.world.schemas import Bar, NewsEvent


def _build_feature_vector(gate: dict[str, object], symbols: list[str]) -> list[float]:
    inputs = cast(dict[str, dict[str, float]], gate.get("inputs", {}))
    if not inputs:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _mean(name: str) -> float:
        vals = [inputs[s].get(name, 0.0) for s in symbols if s in inputs]
        return (sum(vals) / len(vals)) if vals else 0.0

    return [
        cast(float, gate.get("lambda_global", 0.0)),
        _mean("volume_z"),
        _mean("gap_rate"),
        _mean("outlier_rate"),
        _mean("vol_of_vol"),
        _mean("corr_shock"),
        _mean("event_intensity_z"),
    ]


def _target_qty(operator: str, last_return: float) -> float:
    if operator == "flat":
        return 0.0
    if operator == "mean_reversion":
        return -1.0 if last_return > 0 else 1.0
    if operator == "breakout":
        return 1.0 if last_return > 0 else -1.0
    return 0.0


@dataclass(frozen=True)
class SimulationConfig:
    initial_cash: float = 1_000_000.0
    slippage_bps: float = 0.6
    fee_bps: float = 0.2
    max_notional_per_order: float = 250_000.0
    max_daily_loss: float = 20_000.0
    max_gross_notional: float = 1_000_000.0


@dataclass(frozen=True)
class SimulationResult:
    stats: BacktestStats
    decisions: list[dict[str, object]]
    equity_curve_points: int


def run_simulation(
    *,
    events: list[Bar | NewsEvent],
    symbols: list[str],
    periods: dict[str, timedelta],
    lambda_cfg: dict[str, Any],
    config: SimulationConfig | None = None,
) -> SimulationResult:
    cfg = config or SimulationConfig()

    decisions: list[dict[str, object]] = []
    book = PortfolioBook(initial_cash=cfg.initial_cash)
    oms = SimulatedOMS(slippage_bps=cfg.slippage_bps, fee_bps=cfg.fee_bps)
    limits = RiskLimits(
        max_notional_per_order=cfg.max_notional_per_order,
        max_daily_loss=cfg.max_daily_loss,
        max_gross_notional=cfg.max_gross_notional,
    )
    risk_state = RiskState()
    selector = OperatorSelectorEntropyMI()
    graph_map = RelationalGraphMap(symbols=symbols)
    successor_map = SuccessorMap()
    spread_samples: list[float] = []
    slippage_samples: list[float] = []

    prev_equity: float | None = None
    last_operator: str | None = None
    last_feature_vector: list[float] | None = None

    def _on_decision(ctx: dict[str, object]) -> None:
        nonlocal prev_equity, last_operator, last_feature_vector

        gate = cast(dict[str, object], ctx["gate"])
        ts = cast(datetime, ctx["ts"])
        medium_bars = cast(dict[str, Bar], ctx["medium_bars"])

        if (
            prev_equity is not None
            and last_operator is not None
            and last_feature_vector is not None
        ):
            reward = (book.equity - prev_equity) / max(abs(prev_equity), 1.0)
            selector.observe(
                operator=last_operator,
                reward=reward,
                feature_vector=last_feature_vector,
            )

        feature_vector = _build_feature_vector(gate, symbols)
        inputs = cast(dict[str, dict[str, float]], gate.get("inputs", {}))
        mismatch_proxy = sum(
            abs(inputs[s].get("last_return", 0.0)) for s in symbols if s in inputs
        ) / max(1, len(symbols))

        returns_by_symbol = {s: inputs.get(s, {}).get("last_return", 0.0) for s in symbols}
        relational_state = graph_map.update(returns_by_symbol)
        armed = cast(bool, gate.get("armed", False))
        sr_snapshot = successor_map.update(relational_state, learn=armed)

        operator, mi_score = selector.select(
            armed=armed,
            mismatch=mismatch_proxy,
            feature_vector=feature_vector,
            sr_uncertainty=sr_snapshot.uncertainty,
        )

        for symbol in symbols:
            last_ret = inputs.get(symbol, {}).get("last_return", 0.0)
            target_qty = _target_qty(operator, last_ret)
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
        prev_equity = equity
        last_operator = operator
        last_feature_vector = feature_vector

        decisions.append(
            {
                "ts": ts,
                "armed": cast(bool, gate["armed"]),
                "operator": operator,
                "mi_score": mi_score,
                "temperature": selector.temperature,
                "relational_cluster": relational_state.cluster_label,
                "relational_coupling": relational_state.coupling_index,
                "relational_state_key": relational_state.motif_key,
                "sr_state_id": sr_snapshot.state_id,
                "sr_uncertainty": sr_snapshot.uncertainty,
                "sr_transition_entropy": sr_snapshot.transition_entropy,
                "sr_td_error_norm": sr_snapshot.td_error_norm,
                "sr_learned": sr_snapshot.learned,
                "lambda_global": cast(float, gate["lambda_global"]),
                "good_streams": cast(int, gate["good_streams"]),
                "equity": equity,
                "daily_pnl": book.daily_pnl(ts),
            }
        )

    engine = BacktestEngine(
        symbols=symbols,
        periods=periods,
        lambda_cfg=lambda_cfg,
        on_decision=_on_decision,
    )
    engine.run(events)

    stats = summarize(
        equity_curve=book.equity_curve,
        turnover=book.turnover,
        spread_samples=spread_samples,
        slippage_samples=slippage_samples,
    )
    return SimulationResult(
        stats=stats,
        decisions=decisions,
        equity_curve_points=len(book.equity_curve),
    )
