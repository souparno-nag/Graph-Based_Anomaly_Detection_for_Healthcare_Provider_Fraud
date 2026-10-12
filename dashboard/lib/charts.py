"""Shared Plotly styling so every chart in the dashboard reads as one system."""
import plotly.graph_objects as go
import streamlit as st

from lib.data import MODELS

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# Categorical slots, always assigned in this order.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

# Not fraud / fraud use the blue <-> red diverging pair everywhere,
# so "toward fraud" in SHAP charts matches the fraud label colour.
LEGIT = '#2a78d6'
FRAUD = '#e34948'
LABEL_COLORS = {'Not fraud': LEGIT, 'Fraud': FRAUD}

MODEL_COLORS = dict(zip(MODELS.values(), SERIES))

PHYSICIAN = '#1baf7a'
BENEFICIARY = '#b5b3ab'


def style(fig: go.Figure, height: int = 340, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=8, r=8, t=36 if legend else 12, b=8),
        font=dict(family=FONT, size=13, color=INK_2),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        showlegend=legend,
        legend=dict(orientation='h', yanchor='bottom', y=1.0, x=0, title_text=''),
        hoverlabel=dict(bgcolor='white', bordercolor=AXIS, font=dict(family=FONT, color=INK)),
        bargap=0.25,
        barcornerradius=4,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED,
                     title_font_color=INK_2)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont_color=MUTED,
                     title_font_color=INK_2)
    return fig


def show(fig: go.Figure, **kwargs):
    """Render with our own styling, not Streamlit's theme override."""
    return st.plotly_chart(fig, theme=None, config={'displayModeBar': False}, **kwargs)
