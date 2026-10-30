#!/bin/bash
# Exits non-zero unless torchpalmm is installed from
# pip install git+https://github.com/rrrrn/torchpalmm.git
uv run python - <<'PY' || { echo "Install with: pip install git+https://github.com/rrrrn/torchpalmm.git"; exit 1; }
import importlib.metadata as md, json, sys
try:
    d = md.distribution("torchpalmm")
except md.PackageNotFoundError:
    sys.exit("torchpalmm is not installed")
url = json.loads(d.read_text("direct_url.json") or "{}").get("url", "")
if "rrrrn/torchpalmm" not in url:
    sys.exit(f"torchpalmm was not installed from the GitHub repo (source: {url or 'package index'})")
print(f"torchpalmm {d.version} from {url}")
PY
