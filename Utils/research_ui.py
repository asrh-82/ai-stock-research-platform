from datetime import date
import streamlit as st
from Utils.research import load_evidence, render_report


def render_research(context):
    st.subheader('Source-backed company report')
    if context['ticker']!='MTY.TO':
        st.info('The verified research pack currently covers Kahala Brands and MTY. Search Kahala Brands to open it. Other tickers retain their provider snapshot; they do not yet have verified report packs.')
        return
    evidence=load_evidence()
    st.caption(f"Evidence verified {evidence['verified_on']}. Parent financials and parent stock are identified throughout.")
    report=render_report(evidence,as_of=max(date.today().isoformat(),evidence['verified_on']))
    st.download_button('Download sourced report',report,'Kahala_Brands_Report.md','text/markdown')
    st.markdown(report)
    from pathlib import Path
    import json
    import pandas as pd
    snapshot_path=Path(__file__).resolve().parents[1]/'Data'/'research'/'kahala_study.json'
    if snapshot_path.exists():
        snapshot=json.loads(snapshot_path.read_text())
        st.subheader('Frozen historical study')
        st.caption(f"Verified {snapshot['verified_on']}; {snapshot['holdout_start']} through {snapshot['holdout_end']}. This result is a dated snapshot, not recalculated from current prices.")
        rows=[]
        for label,key in [('Selected RSI rule','holdout_metrics'),('MTY buy and hold','mty_buy_hold_metrics'),('XIU benchmark','benchmark_metrics')]:
            value=snapshot[key]
            rows.append({'Portfolio':label,'Return':f"{value['total_return']:.2%}",'Maximum drawdown':f"{value['maximum_drawdown']:.2%}"})
        st.dataframe(pd.DataFrame(rows),hide_index=True)
        st.warning('Only three completed final-period trades. Earlier fixed-rule RSI performance was negative. No repeatable trading edge is established.')
        st.download_button('Download frozen study summary',snapshot_path.read_text(),'Kahala_MTY_Study.json','application/json')
