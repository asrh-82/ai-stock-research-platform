#!/usr/bin/env bash
# A browser check against the CI server, not a claim about the protected Vercel preview.
set -euo pipefail
mkdir -p artifacts
python -m streamlit run pages/2_Walk_Forward.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false > artifacts/streamlit.log 2>&1 &
server_pid=$!
cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    agent-browser screenshot artifacts/browser_failure.png --full || true
    agent-browser get text body > artifacts/browser_failure.txt || true
    tail -n 60 artifacts/streamlit.log || true
  fi
  agent-browser close || true
  kill "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT
for attempt in $(seq 1 30); do
  if curl --silent --fail http://127.0.0.1:8501/_stcore/health >/dev/null; then break; fi
  sleep 1
done
agent-browser open http://127.0.0.1:8501
agent-browser set viewport 1440 1200
agent-browser wait --text "Walk-forward Research"
agent-browser snapshot -i | tee artifacts/browser_initial.txt
agent-browser screenshot artifacts/browser_initial.png
wait_for_text() {
  for attempt in $(seq 1 30); do
    agent-browser get text body > artifacts/browser_current.txt
    if grep --fixed-strings --quiet "$1" artifacts/browser_current.txt; then return 0; fi
    sleep 1
  done
  cat artifacts/browser_current.txt
  return 1
}
# Refresh refs on every interaction. Streamlit's visible LabelText wraps a hidden
# radio input; selecting the input by role is not reliable in the browser CLI.
click_snapshot_ref() {
  agent-browser snapshot -i > artifacts/browser_current_snapshot.txt
  ref=$(python - "$1" "$2" <<'PYREF'
import re
import sys
from pathlib import Path
role, name = sys.argv[1:]
pattern = re.compile(r'^\s*-\s+' + re.escape(role) + r'\s+"' + re.escape(name) + r'".*?\bref=(e\d+)', re.I)
lines = Path("artifacts/browser_current_snapshot.txt").read_text().splitlines()
matches = [match.group(1) for line in lines if (match := pattern.search(line))]
if len(matches) != 1:
    raise SystemExit(f"Expected one {role} named {name!r}; found {len(matches)}.\n" + "\n".join(lines))
print(matches[0])
PYREF
)
  echo "BROWSER_TARGET $1 $2 @$ref"
  agent-browser scrollintoview "@$ref"
  agent-browser wait 500
  agent-browser is enabled "@$ref" | grep 'true'
  agent-browser click "@$ref"
  agent-browser wait 500
}
wait_for_text 'Walk-forward Research'
agent-browser eval 'document.querySelectorAll("[data-testid=stException]").length === 0' | grep 'true'
click_snapshot_ref LabelText 'Synthetic software demo'
wait_for_text 'SYNTHETIC SOFTWARE DEMO'
agent-browser snapshot -i > artifacts/browser_demo.txt
click_snapshot_ref button 'Load data'
wait_for_text 'Loaded 1,260 sessions'
agent-browser snapshot -i > artifacts/browser_loaded.txt
click_snapshot_ref button 'Run development and freeze plan'
wait_for_text 'Strategy status: NOT VALIDATED'
agent-browser screenshot artifacts/browser_development.png --full
agent-browser snapshot -i > artifacts/browser_development.txt
click_snapshot_ref tab 'Reserved holdout'
agent-browser snapshot -i > artifacts/browser_reserved.txt
click_snapshot_ref button 'Evaluate frozen holdout'
wait_for_text 'Holdout evaluated after freeze'
agent-browser screenshot artifacts/browser_holdout.png --full
agent-browser snapshot -i > artifacts/browser_holdout.txt
agent-browser eval 'document.querySelectorAll("[data-testid=stException]").length === 0' | grep 'true'
echo 'BROWSER_RESEARCH_SMOKE=PASSED (CI-local server, synthetic interactions only)'
