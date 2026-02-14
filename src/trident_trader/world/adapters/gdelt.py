from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class GDELTEvent:
    ts: datetime
    source_url: str
    avg_tone: float
    num_mentions: int
    num_articles: int


def _coerce_datetime(raw: str) -> datetime:
    raw = raw.strip()
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d")
    if len(raw) == 14 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d%H%M%S")
    return datetime.fromisoformat(raw)


def _detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t"


def load_gdelt_events(path: str) -> list[GDELTEvent]:
    """Load a GDELT CSV/TSV export into typed event records."""
    p = Path(path)
    delimiter = _detect_delimiter(p)
    events: list[GDELTEvent] = []

    with p.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            date_raw = row.get("DATE") or row.get("SQLDATE") or row.get("datetime") or ""
            if not date_raw:
                continue

            tone_raw = row.get("AvgTone") or row.get("avg_tone") or "0"
            mentions_raw = row.get("NumMentions") or row.get("num_mentions") or "0"
            articles_raw = row.get("NumArticles") or row.get("num_articles") or "0"
            src_raw = row.get("SOURCEURL") or row.get("source_url") or ""

            try:
                event = GDELTEvent(
                    ts=_coerce_datetime(date_raw),
                    source_url=src_raw,
                    avg_tone=float(tone_raw),
                    num_mentions=int(float(mentions_raw)),
                    num_articles=int(float(articles_raw)),
                )
            except ValueError:
                continue
            events.append(event)

    return events


def aggregate_intensity(events: list[GDELTEvent]) -> float:
    """Simple world-news intensity metric for Lambda input."""
    if not events:
        return 0.0
    weighted = [max(0.0, e.num_mentions * 0.6 + e.num_articles * 0.4) for e in events]
    return sum(weighted) / len(weighted)
