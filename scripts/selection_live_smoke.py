"""Bounded provider smoke check, not historical trading performance or a recommendation."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from Utils.selection import scan
from Utils.selection_data import load_universe

out=Path('artifacts');out.mkdir(exist_ok=True)
try:
    batch=load_universe('AAPL,MSFT,JNJ,XOM,JPM,PG')
    result=scan(batch)
    report={'status':'SCANNER_RUN_COMPLETED', 'asof':result['asof'],
            'created_at':result['created_at'],'eligible':len(result['ranked']),
            'excluded':result['excluded'],'scan_id':result['id'],
            'note':'A live-data software smoke test, not a backtest or evidence of an edge.'}
    (out/'selection_provider.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
    if len(result['ranked'])<2:sys.exit(2)
except Exception as exc:
    report={'status':'PROVIDER_UNAVAILABLE_OR_FAILED', 'error':str(exc),
            'note':'No synthetic fallback was used.'}
    (out/'selection_provider.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report));sys.exit(2)
