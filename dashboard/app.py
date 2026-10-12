"""Entry point: streamlit run dashboard/app.py"""
import streamlit as st

st.set_page_config(
    page_title='Provider Fraud Detection',
    page_icon=':material/hub:',
    layout='wide',
)

pages = [
    st.Page('views/overview.py', title='Overview', icon=':material/dashboard:', default=True),
    st.Page('views/model_performance.py', title='Model performance', icon=':material/monitoring:'),
]

with st.sidebar:
    st.caption(
        'Graph-based anomaly detection for healthcare provider fraud. '
        'Data: Kaggle Healthcare Provider Fraud Detection (Medicare claims, 2009).'
    )

st.navigation(pages).run()
