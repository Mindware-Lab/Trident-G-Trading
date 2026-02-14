from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    mode = "smoke" if args.smoke else "full"
    print(f"Backtest script stub ({mode}).")


if __name__ == "__main__":
    main()
