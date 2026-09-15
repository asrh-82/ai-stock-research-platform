"""Application entrypoint: workspaces first, research methods second."""
import streamlit as st

st.set_page_config(page_title="Equity Research Platform", layout="wide", initial_sidebar_state="expanded")
page = st.navigation({
    "Workspace": [
        st.Page("pages/0_Stock_Selection.py", title="Stock selection", default=True),
        st.Page("pages/4_Market_Monte_Carlo.py", title="Market Monte Carlo", url_path="market-monte-carlo"),
        st.Page("pages/3_Company_Tools.py", title="Company tools"),
    ],
    "Research methods": [
        st.Page("pages/1_Validation_Lab.py", title="Execution validation"),
        st.Page("pages/2_Walk_Forward.py", title="Walk-forward research"),
    ],
})
page.run()
