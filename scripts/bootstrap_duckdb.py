from __future__ import annotations

from pathlib import Path

import duckdb


def main() -> None:
    db_path = Path("data") / "processed" / "trident.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("create table if not exists heartbeat(ts timestamp, ok boolean)")
    con.close()
    print(f"Initialized {db_path}")


if __name__ == "__main__":
    main()
