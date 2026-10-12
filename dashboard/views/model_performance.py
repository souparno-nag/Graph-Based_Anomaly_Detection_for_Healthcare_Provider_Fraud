import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from lib.charts import FRAUD, INK, INK_2, LEGIT, MODEL_COLORS, MUTED, SERIES, show, style
from lib.data import MODELS, eval_rows, load_cv, load_metadata, load_scores

scores = load_scores()
meta = load_metadata()

st.title('Model performance')
st.markdown(
    'Only 9% of providers are labelled fraud, so accuracy is misleading: a model that flags nobody is 91% '
    'accurate. The headline metric here is **PR-AUC** (area under the precision–recall curve), which '
    'only rewards a model for ranking real fraud above everything else. Everything below is measured on '
    f"the **{(scores['split'] == 'test').sum():,} held-out test providers** the models never trained on."
)


@st.cache_data
def threshold_free_metrics() -> pd.DataFrame:
    rows = []
    for col, name in MODELS.items():
        d = eval_rows(scores, col)
        rows.append({'col': col, 'Model': name, 'AUC': roc_auc_score(d['PotentialFraud'], d[col]),
                     'PR-AUC': average_precision_score(d['PotentialFraud'], d[col]), 'Evaluated on': len(d)})
    return pd.DataFrame(rows)


base = threshold_free_metrics()
# Ensembles are dashed so they're distinguishable from the single models without relying on colour.
DASH = {col: 'dash' if col.startswith('p_ens') else 'solid' for col in MODELS}

# --- Headline: PR-AUC climbs as the models see more of the graph ---
pr = base.set_index('col')['PR-AUC']
cols = st.columns(4)
for c, (col, label) in zip(cols, [('p_xgb_baseline', 'Billing only'),
                                  ('p_xgb_graph', '+ graph features'),
                                  ('p_gnn', 'Full GNN'),
                                  ('p_ens_stacked', 'Stacked ensemble')]):
    delta = None if col == 'p_xgb_baseline' else f"{pr[col] - pr['p_xgb_baseline']:+.3f} vs billing only"
    c.metric(f'PR-AUC · {label}', f'{pr[col]:.3f}', delta, border=True)

# --- Threshold-dependent table ---
st.subheader('All models at one decision threshold')
threshold = st.slider(
    'Flag a provider as fraud when its score is at least', 0.05, 0.95, float(meta['threshold']), 0.05,
    help=f"The notebook used {meta['threshold']}. Lower catches more fraud (recall) but flags more honest "
         'providers (precision drops).',
)
rows = []
for _, r in base.iterrows():
    d = eval_rows(scores, r['col'])
    y, pred = d['PotentialFraud'], (d[r['col']] >= threshold).astype(int)
    rows.append({**r.drop('col'), 'Precision': precision_score(y, pred, zero_division=0),
                 'Recall': recall_score(y, pred), 'F1': f1_score(y, pred),
                 'Flagged': int(pred.sum()), 'Fraud caught': int((pred & y).sum()), 'Fraud total': int(y.sum())})
table = pd.DataFrame(rows)
st.dataframe(
    table, hide_index=True,
    column_order=['Model', 'PR-AUC', 'AUC', 'Precision', 'Recall', 'F1', 'Flagged', 'Fraud caught',
                  'Fraud total', 'Evaluated on'],
    column_config={
        'PR-AUC': st.column_config.ProgressColumn(format='%.3f', min_value=0, max_value=1, color='blue'),
        'AUC': st.column_config.NumberColumn(format='%.3f'),
        'Precision': st.column_config.NumberColumn(format='%.3f', help='Of the providers flagged, the share that are labelled fraud'),
        'Recall': st.column_config.NumberColumn(format='%.3f', help='Of all fraud-labelled providers, the share that got flagged'),
        'F1': st.column_config.NumberColumn(format='%.3f'),
        'Evaluated on': st.column_config.NumberColumn(help='Test providers this model is scored on'),
    },
)
st.caption(
    'The stacked ensemble was fitted on half of the test set, so it is scored only on the other half '
    f"({int(table.loc[table['Model'] == MODELS['p_ens_stacked'], 'Evaluated on'].iloc[0])} providers). "
    'Its numbers are not directly comparable with the rest.'
)

# --- Curves ---
left, right = st.columns(2)
with left:
    st.subheader('Precision–recall curves')
    fig = go.Figure()
    prevalence = (scores['split'] == 'test') & (scores['PotentialFraud'] == 1)
    prevalence = prevalence.sum() / (scores['split'] == 'test').sum()
    fig.add_hline(y=prevalence, line_dash='dot', line_color=MUTED, line_width=1.5,
                  annotation_text=f'Random guessing ({prevalence:.0%})', annotation_font_color=INK_2,
                  annotation_position='bottom left')
    for col, name in MODELS.items():
        d = eval_rows(scores, col)
        p, r, t = precision_recall_curve(d['PotentialFraud'], d[col])
        fig.add_scatter(x=r, y=p, mode='lines', name=name, line=dict(color=MODEL_COLORS[name], width=2, dash=DASH[col]),
                        customdata=np.append(t, np.nan),
                        hovertemplate=name + '<br>Recall %{x:.2f} · Precision %{y:.2f}'
                                             '<br>Threshold %{customdata:.2f}<extra></extra>')
    fig.update_xaxes(title='Recall', range=[0, 1.01])
    fig.update_yaxes(title='Precision', range=[0, 1.02])
    show(style(fig, height=420))

with right:
    st.subheader('ROC curves')
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color=MUTED, dash='dot', width=1.5),
                    name='Random guessing', hoverinfo='skip', showlegend=False)
    for col, name in MODELS.items():
        d = eval_rows(scores, col)
        fpr, tpr, _ = roc_curve(d['PotentialFraud'], d[col])
        fig.add_scatter(x=fpr, y=tpr, mode='lines', name=name, line=dict(color=MODEL_COLORS[name], width=2, dash=DASH[col]),
                        hovertemplate=name + '<br>False-positive rate %{x:.2f} · Recall %{y:.2f}<extra></extra>')
    fig.update_xaxes(title='False-positive rate', range=[0, 1])
    fig.update_yaxes(title='True-positive rate (recall)', range=[0, 1.02])
    show(style(fig, height=420))

# --- One model in detail ---
st.subheader('One model up close')
model_name = st.selectbox('Model', list(MODELS.values()), index=2)
model_col = {v: k for k, v in MODELS.items()}[model_name]
d = eval_rows(scores, model_col)
y = d['PotentialFraud'].to_numpy()
s = d[model_col].to_numpy()

left, right = st.columns([2, 3])
with left:
    cm = confusion_matrix(y, (s >= threshold).astype(int), labels=[0, 1])
    names = [['True negative', 'False positive'], ['Missed fraud', 'Fraud caught']]
    fig = go.Figure(go.Heatmap(
        z=cm, x=['Predicted<br>not fraud', 'Predicted<br>fraud'], y=['Actually<br>not fraud', 'Actually<br>fraud'],
        colorscale=[[0, '#f0efec'], [1, '#86b6ef']], showscale=False, xgap=2, ygap=2,
        text=[[f'{names[i][j]}<br><b>{cm[i, j]:,}</b>' for j in range(2)] for i in range(2)],
        texttemplate='%{text}', textfont=dict(color=INK, size=14),
        hovertemplate='%{y}, %{x}: %{z:,}<extra></extra>',
    ))
    fig.update_yaxes(autorange='reversed', showgrid=False)
    fig.update_xaxes(side='top', showgrid=False)
    show(style(fig, height=320, legend=False))
    st.caption(f'At threshold {threshold:.2f}, on {len(d):,} test providers.')

with right:
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curves = pd.DataFrame({
        't': grid,
        'Precision': [precision_score(y, s >= t, zero_division=0) for t in grid],
        'Recall': [recall_score(y, s >= t) for t in grid],
        'F1': [f1_score(y, s >= t) for t in grid],
    })
    fig = go.Figure()
    for metric, color in zip(['Precision', 'Recall', 'F1'], SERIES):
        fig.add_scatter(x=curves['t'], y=curves[metric], mode='lines', name=metric,
                        line=dict(color=color, width=2),
                        hovertemplate=metric + ' %{y:.3f} at threshold %{x:.2f}<extra></extra>')
    fig.add_vline(x=threshold, line_color=INK_2, line_width=1.5, line_dash='dash',
                  annotation_text=f'Threshold {threshold:.2f}', annotation_font_color=INK_2)
    fig.update_xaxes(title='Threshold', range=[0.05, 0.95])
    fig.update_yaxes(title='Score', range=[0, 1.02])
    fig.update_layout(hovermode='x unified')
    show(style(fig, height=320))
    st.caption('Move the slider above to see the trade-off: a lower threshold catches more fraud but '
               'flags more honest providers.')

# --- Cross-validation ---
st.subheader('Does the graph feature help consistently? 5-fold cross-validation')
cv = load_cv()
metric_names = {'auc': 'AUC', 'pr_auc': 'PR-AUC', 'recall': 'Recall', 'precision': 'Precision'}
long = cv.melt(id_vars=['model', 'fold'], value_vars=list(metric_names), var_name='metric')
long['metric'] = long['metric'].map(metric_names)
fig = go.Figure()
for model, color in [('XGBoost (baseline)', LEGIT), ('XGBoost + graph-density', FRAUD)]:
    m = long[long['model'] == model]
    fig.add_box(x=m['metric'], y=m['value'], name=model, marker_color=color, boxpoints='all',
                jitter=0.3, pointpos=0, marker_size=8, line_width=1.5, boxmean=True,
                hovertemplate=model + '<br>%{x}: %{y:.3f}<extra></extra>')
fig.update_layout(boxmode='group')
fig.update_yaxes(title='Score (one dot per fold)', range=[0.5, 1])
show(style(fig, height=360))
summary = cv.groupby('model')[list(metric_names)].agg(['mean', 'std'])
gain = summary[('pr_auc', 'mean')]['XGBoost + graph-density'] - summary[('pr_auc', 'mean')]['XGBoost (baseline)']
st.caption(
    f'Across 5 folds, adding the graph features raises mean PR-AUC by {gain:+.3f}. That is small next to '
    f"the fold-to-fold spread (std ≈ {summary[('pr_auc', 'std')].mean():.3f}), so treat the XGBoost gain "
    'as suggestive, not proven.'
)

with st.expander('Caveats'):
    st.markdown(
        '- **Fraud-neighbour density uses every provider\'s label**, including test providers, when it '
        'is computed. This leaks some test information into the graph-augmented XGBoost, so its gain over '
        'the baseline is probably overstated.\n'
        '- **Labels are "potential fraud"**, not convictions, and they apply to the provider as a whole, '
        'not to individual claims.\n'
        '- **The GNN learns from the whole graph at once** (transductive). Test providers\' labels are hidden '
        'from the loss, but their links are visible during training.'
    )
