"""Wraps template.html (an Artifact-style fragment -- no <html>/<head>/<body>,
those were auto-injected by the Artifact tool when this dashboard was first
published there) into a complete, standalone HTML document at site/index.html,
with the current dashboard_data.json baked in. Static output, no server or
Python runtime needed to serve it -- that's what the root vercel.json declares.

Run order to refresh the published site:
  1. python src/engine/train.py             (or whatever changed)
  2. python src/comparison/compare_tickers.py
  3. python src/engine/backtest_dates.py 5    (keeps the "last 5 trading days" section current)
  4. python src/engine/long_horizon_drift.py  (keeps the horizon-selector section current)
  5. python src/dashboard/build_dashboard_data.py
  6. python src/dashboard/build_site.py
"""
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
DATA_JSON = HERE / "dashboard_data.json"
SITE_DIR = HERE.parent.parent / "site"

HEAD_BODY_SPLIT_MARKER = "</style>"


def build():
    template = TEMPLATE.read_text()
    data = DATA_JSON.read_text()  # already valid JSON text, embed as-is

    asof = date.today().strftime("%b %-d, %Y")
    page = template.replace("__DATA_JSON__", data).replace("__ASOF_DATE__", asof)

    split_at = page.index(HEAD_BODY_SPLIT_MARKER) + len(HEAD_BODY_SPLIT_MARKER)
    head_content, body_content = page[:split_at], page[split_at:]

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{head_content}
</head>
<body>
{body_content}
</body>
</html>
"""
    SITE_DIR.mkdir(exist_ok=True)
    out_path = SITE_DIR / "index.html"
    out_path.write_text(doc)
    print(f"Saved {out_path} ({len(doc):,} bytes)")


if __name__ == "__main__":
    build()
