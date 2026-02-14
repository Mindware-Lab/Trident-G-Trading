from trident_trader.cli import build_parser


def test_cli_has_commands() -> None:
    parser = build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    commands = set(actions[0].choices.keys())
    assert {"ingest", "compute-lambda", "paper", "backtest"}.issubset(commands)
