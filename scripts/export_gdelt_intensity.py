from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _run_bigquery_export(project: str, dataset: str, table: str, output_path: Path) -> None:
    try:
        from google.cloud import bigquery
    except Exception as exc:  # pragma: no cover - dependency is optional in CI
        raise RuntimeError(
            "google-cloud-bigquery is required for export mode. "
            "Install it and configure credentials."
        ) from exc

    client = bigquery.Client(project=project)
    query = f"""
    SELECT
      TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), HOUR) AS ts,
      COUNT(*) AS count
    FROM `{project}.{dataset}.{table}`
    GROUP BY ts
    ORDER BY ts
    """
    rows = list(client.query(query).result())
    ts_values = [row["ts"] for row in rows]
    count_values = [float(row["count"]) for row in rows]

    table_out = pa.table({"ts": ts_values, "count": count_values})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table_out, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--dataset", default="gdeltv2")
    parser.add_argument("--table", default="events")
    parser.add_argument("--output", default="data/news/gdelt_intensity.parquet")
    args = parser.parse_args()

    out_path = Path(args.output)
    _run_bigquery_export(
        project=args.project,
        dataset=args.dataset,
        table=args.table,
        output_path=out_path,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
