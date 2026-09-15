#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false > artifacts/workspace_server.log 2>&1 &
server_pid=$!
trap 'agent-browser close >/dev/null 2>&1 || true; kill "$server_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 30); do
  if curl --silent --fail http://127.0.0.1:8501/_stcore/health >/dev/null; then break; fi
  sleep 1
done
# Initial dev-server visual check through the required browser CLI.
agent-browser open http://127.0.0.1:8501
agent-browser wait --load networkidle
agent-browser snapshot -i > artifacts/workspace_initial.txt
agent-browser screenshot artifacts/workspace_initial.png
agent-browser eval 'document.querySelectorAll("[data-testid=stException]").length === 0' | grep 'true'
agent-browser close
# Native browser clicks (not JavaScript state injection) exercise the complete workflow.
python scripts/workspace_browser.py
python - <<'PY'
from pathlib import Path
from Utils.paper_record import parse_backup
record=parse_backup(Path('artifacts/browser_record.json').read_bytes())
assert record['revision']==3
print('BROWSER_BACKUP_VALIDATED=3 events')
PY
