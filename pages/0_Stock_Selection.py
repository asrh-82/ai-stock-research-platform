import streamlit as st

from Utils.selection_ui import render_workspace

st.set_page_config(page_title="Stock Selection Workspace", layout="wide")
render_workspace()
