#!/usr/bin/env bash
# A browser check against the CI server, not a claim about the protected Vercel preview.
set -euo pipefail
mkdir -p artifacts
python -m streamlit run pages/2_Walk_Forward.py --server.address=127.0.0.1 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false > artifacts/streamlit.log 2>&1 &
server_pid=$!
trap 'agent-browser close || true; kill "$server_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 30); do
  if curl --silent --fail http://127.0.0.1:8501/_stcore/health >/dev/null; then break; fi
  sleep 1
done
agent-browser open http://127.0.0.1:8501
agent-browser wait --load networkidle
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
wait_for_text 'Walk-forward Research'
agent-browser eval 'document.querySelectorAll("[data-testid=stException]").length === 0' | grep 'true'
agent-browser find role radio click --name 'Synthetic software demo'
agent-browser snapshot -i > artifacts/browser_demo.txt
agent-browser find role button click --name 'Load data'
wait_for_text 'Loaded 1,260 sessions'
agent-browser snapshot -i > artifacts/browser_loaded.txt
agent-browser find role button click --name 'Run development and freeze plan'
wait_for_text 'Strategy status: NOT VALIDATED'
agent-browser screenshot artifacts/browser_development.png --full
agent-browser snapshot -i > artifacts/browser_development.txt
agent-browser find role tab click --name 'Reserved holdout'
agent-browser snapshot -i > artifacts/browser_reserved.txt
agent-browser find role button click --name 'Evaluate frozen holdout'
wait_for_text 'Holdout evaluated after freeze'
agent-browser screenshot artifacts/browser_holdout.png --full
agent-browser snapshot -i > artifacts/browser_holdout.txt
agent-browser eval 'document.querySelectorAll("[data-testid=stException]").length === 0' | grep 'true'
echo 'BROWSER_RESEARCH_SMOKE=PASSED (CI-local server, synthetic interactions only)'
