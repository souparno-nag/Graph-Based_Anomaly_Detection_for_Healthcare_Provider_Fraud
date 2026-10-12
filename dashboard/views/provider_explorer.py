import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.charts import (AXIS, BENEFICIARY, FRAUD, INK, INK_2, LABEL_COLORS, LEGIT, MODEL_COLORS, MUTED,
                        PHYSICIAN, show, style)
from lib.data import (FEATURE_NAMES, GRAPH_FEATURES, MODELS, load_claims, load_edges, load_metadata,
                      load_scores, load_shap)
from lib.gnn import explain_provider

scores = load_scores()
meta = load_metadata()
MODEL_COL = {v: k for k, v in MODELS.items()}

st.title('Provider explorer')
st.markdown('Pick a provider from the ranked list to see **how risky each model thinks it is**, '
            '**why**, and **who it is connected to**.')

# --- Filters ---
f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
model_name = f1.selectbox('Rank by model', list(MODELS.values()), index=2)
split_choice = f2.segmented_control(
    'Providers', ['Test only', 'All'], default='Test only', required=True,
    help='Train and validation providers were seen during training, so their scores are optimistic.')
label_choice = f3.segmented_control('Label', ['All', 'Fraud', 'Not fraud'], default='All', required=True)
query = f4.text_input('Search provider ID', placeholder='e.g. PRV51459')
threshold = float(meta['threshold'])
model_col = MODEL_COL[model_name]

view = scores
if split_choice != 'All':
    view = view[view['split'] == 'test']
if label_choice in ('Fraud', 'Not fraud'):
    view = view[view['label'] == label_choice]
if query:
    view = view[view['Provider'].str.contains(query.strip(), case=False)]
view = view.assign(score=view[model_col], flagged=view[model_col] >= threshold)
view = view.sort_values('score', ascending=False).reset_index(drop=True)

if view.empty:
    st.info('No providers match these filters.')
    st.stop()

flagged = view['flagged']
st.caption(f"{len(view):,} providers · {int(flagged.sum()):,} flagged at threshold {threshold:.2f} · "
           f"{int((flagged & (view['PotentialFraud'] == 1)).sum()):,} of those are labelled fraud. "
           'Click a row to inspect it.')

event = st.dataframe(
    view, hide_index=True, height=300, on_select='rerun', selection_mode='single-row', key='ranked',
    column_order=['Provider', 'score', 'flagged', 'label', 'split', 'num_claims', 'total_reimbursed',
                  'fraud_neighbor_density'],
    column_config={
        'score': st.column_config.ProgressColumn(f'Risk score ({model_name})', format='%.3f',
                                                 min_value=0, max_value=1, color='red'),
        'flagged': st.column_config.CheckboxColumn('Flagged'),
        'label': st.column_config.TextColumn('Label'),
        'split': st.column_config.TextColumn('Split'),
        'num_claims': st.column_config.NumberColumn('Claims', format='localized'),
        'total_reimbursed': st.column_config.NumberColumn('Total reimbursed', format='dollar'),
        'fraud_neighbor_density': st.column_config.NumberColumn('Fraud-neighbour density', format='%.2f'),
    },
)
rows = event.selection.rows
row = view.iloc[rows[0] if rows and rows[0] < len(view) else 0]
pid = row['Provider']

# --- Selected provider ---
st.divider()
st.header(pid)
b = st.container(horizontal=True)
b.badge(f"Labelled {row['label'].lower()}", color='red' if row['PotentialFraud'] else 'blue')
b.badge(f"{row['split']} set", color='gray')
if row['score'] >= threshold:
    b.badge(f'Flagged by {model_name}', icon=':material/flag:', color='orange')
if row['split'] != 'test':
    st.caption('This provider was in the training data, so the models have seen its label. Its scores are '
               'optimistic; switch the filter to "Test only" for a fair picture.')

m = st.columns(5)
m[0].metric('Claims', f"{row['num_claims']:,}")
m[1].metric('Total reimbursed', f"${row['total_reimbursed']:,.0f}")
m[2].metric('Beneficiaries', f"{row['num_unique_beneficiaries']:,}")
m[3].metric('Physicians', f"{row['num_unique_physicians']:,}")
m[4].metric('Fraud-neighbour density', f"{row['fraud_neighbor_density']:.0%}",
            help='Share of providers within two hops (sharing a beneficiary or physician) that are labelled fraud.')

left, right = st.columns(2)
with left:
    st.subheader('Every model\'s score')
    names = list(MODELS.values())
    vals = [float(row[c]) for c in MODELS]
    fig = go.Figure(go.Bar(
        y=names, x=vals, orientation='h', marker_color=[MODEL_COLORS[n] for n in names],
        text=[f'{v:.2f}' for v in vals], textposition='outside', textfont_color=INK_2,
        hovertemplate='%{y}: %{x:.3f}<extra></extra>',
    ))
    fig.add_vline(x=threshold, line_color=INK_2, line_dash='dash', line_width=1.5)
    fig.update_xaxes(range=[0, 1.12], title='Fraud score', tickvals=[0, 0.25, 0.5, 0.75, 1])
    fig.update_yaxes(autorange='reversed')
    show(style(fig, height=330, legend=False))
    st.caption(f'Dashed line: the {threshold:.2f} decision threshold.')

with right:
    st.subheader('Why: feature contributions (SHAP)')
    which = st.segmented_control('Explain', ['XGBoost + graph', 'XGBoost (baseline)'],
                                 default='XGBoost + graph', required=True,
                                 label_visibility='collapsed')
    sv, base_value = load_shap('graph' if which != 'XGBoost (baseline)' else 'baseline')
    contrib = sv.loc[pid]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10][::-1]

    def fmt(feat):
        v = row[feat]
        if feat in ('total_reimbursed', 'avg_reimbursed', 'std_reimbursed'):
            v = f'${v:,.0f}'
        elif feat == 'community_id':
            v = f'#{int(v)}'
        elif float(v).is_integer():
            v = f'{int(v):,}'
        else:
            v = f'{v:.2f}'
        tag = ' ◆' if feat in GRAPH_FEATURES else ''
        return f'{FEATURE_NAMES[feat]}{tag} = {v}'

    fig = go.Figure(go.Bar(
        y=[fmt(f) for f in top.index], x=top.values, orientation='h',
        marker_color=[FRAUD if v > 0 else LEGIT for v in top.values],
        hovertemplate='%{y}<br>SHAP %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_xaxes(title='Contribution (log-odds)')
    show(style(fig, height=330, legend=False))
    st.caption(f'Red pushes toward fraud, blue away from it. ◆ marks a graph feature. '
               f'The model starts from {base_value:+.2f} log-odds and adds these contributions.')

# --- Live GNN what-if ---
st.subheader('Ask the GNN (live)')
st.markdown('Runs the trained GNN on the full graph right now and asks what moves **this provider\'s** score.')
if st.button('Run what-if tests', icon=':material/play_arrow:') or st.session_state.get('gnn_done') == pid:
    st.session_state['gnn_done'] = pid
    with st.spinner('Running the GNN on 226k nodes...'):
        res = explain_provider(pid)
    occ = pd.Series(res['occlusion'])
    occ = occ.reindex(occ.abs().sort_values(ascending=False).index)[:8][::-1]
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Bar(
            y=[FEATURE_NAMES[f] + (' ◆' if f in GRAPH_FEATURES else '') for f in occ.index], x=occ.values,
            orientation='h', marker_color=[FRAUD if v > 0 else LEGIT for v in occ.values],
            hovertemplate='%{y}: score falls by %{x:+.3f} when reset to average<extra></extra>',
        ))
        fig.add_vline(x=0, line_color=AXIS, line_width=1)
        fig.update_xaxes(title='Score drop when reset to average')
        show(style(fig, height=300, legend=False))
        st.caption('Each feature is reset to the average provider\'s value, one at a time. Red: this feature '
                   'was pushing the score up. Blue: it was holding the score down.')
        if res['score'] > 0.97 or res['score'] < 0.03:
            st.caption('This score is close to 0 or 1, where the model\'s output is saturated, so no single '
                       'feature moves it much on its own.')
    with right:
        rel = pd.Series(res['relations']).round(3) + 0.0  # + 0.0 avoids printing -0.000
        fig = go.Figure(go.Bar(
            x=rel.index, y=rel.values, marker_color=[PHYSICIAN, BENEFICIARY],
            text=[f'{v:+.3f}' for v in rel.values], textposition='outside', textfont_color=INK_2,
            hovertemplate='%{x}: score falls by %{y:+.3f} without these links<extra></extra>',
        ))
        fig.add_hline(y=0, line_color=AXIS, line_width=1)
        pad = max(abs(rel).max() * 1.4, 0.02)
        fig.update_yaxes(title='Score drop without links', range=[min(0, rel.min()) - pad * 0.3, max(0, rel.max()) + pad])
        show(style(fig, height=300, legend=False))
        st.caption(f"GNN score {res['score']:.3f}. Bars show how much it falls when one kind of link "
                   'is removed from the graph.')

# --- Compared with everyone else ---
st.subheader('Compared with all providers')
billing = [f for f in meta['baseline_features']] + ['fraud_neighbor_density']
pct = [(scores[f] < row[f]).mean() * 100 for f in billing]
fig = go.Figure(go.Bar(
    y=[FEATURE_NAMES[f] for f in billing], x=pct, orientation='h', marker_color=LEGIT,
    customdata=[row[f] for f in billing],
    hovertemplate='%{y}: higher than %{x:.0f}% of providers<br>Value %{customdata:,.2f}<extra></extra>',
))
fig.add_vline(x=50, line_dash='dot', line_color=MUTED, line_width=1.5)
fig.update_xaxes(range=[0, 100], title='Percentile (dotted line = median)', ticksuffix='%')
fig.update_yaxes(autorange='reversed')
show(style(fig, height=420, legend=False))


# --- Network ---
@st.cache_data(max_entries=64)
def ego_network(pid: str, max_physicians: int = 15, max_benes: int = 25, max_neighbours: int = 30):
    pb, pp = load_edges()
    labels = scores.set_index('Provider')['label']
    my_ph = pp[pp['Provider'] == pid].nlargest(max_physicians, 'n_claims')['AttendingPhysician']
    my_be = pb[pb['Provider'] == pid].nlargest(max_benes, 'n_claims')['BeneID']
    # Other providers reached through the shown physicians/beneficiaries, strongest links first.
    hop = pd.concat([pp[pp['AttendingPhysician'].isin(my_ph)].rename(columns={'AttendingPhysician': 'via'}),
                     pb[pb['BeneID'].isin(my_be)].rename(columns={'BeneID': 'via'})])
    hop = hop[hop['Provider'] != pid]
    keep = hop.groupby('Provider').size().nlargest(max_neighbours).index
    hop = hop[hop['Provider'].isin(keep)]

    G = nx.Graph()
    G.add_node(pid, kind='provider', label=labels.get(pid))
    for ph in my_ph:
        G.add_node(ph, kind='physician')
        G.add_edge(pid, ph)
    for be in my_be:
        G.add_node(be, kind='beneficiary')
        G.add_edge(pid, be)
    for _, e in hop.iterrows():
        G.add_node(e['Provider'], kind='provider', label=labels.get(e['Provider']))
        G.add_edge(e['Provider'], e['via'])
    pos = nx.spring_layout(G, seed=42, k=1.2 / np.sqrt(max(G.number_of_nodes(), 1)), iterations=80)
    nodes = pd.DataFrame([{'id': n, **G.nodes[n], 'x': pos[n][0], 'y': pos[n][1], 'degree': G.degree[n]}
                          for n in G.nodes])
    edges = [(pos[a], pos[b]) for a, b in G.edges]
    return nodes, edges


st.subheader('Who it is connected to')
nodes, edges = ego_network(pid)
ex, ey = [], []
for (x0, y0), (x1, y1) in edges:
    ex += [x0, x1, None]
    ey += [y0, y1, None]
fig = go.Figure()
fig.add_scatter(x=ex, y=ey, mode='lines', line=dict(color=AXIS, width=0.8), hoverinfo='skip', showlegend=False)
groups = [
    ('Beneficiary', nodes['kind'] == 'beneficiary', BENEFICIARY, 'square', 8),
    ('Physician', nodes['kind'] == 'physician', PHYSICIAN, 'diamond', 11),
    ('Provider · not fraud', (nodes['kind'] == 'provider') & (nodes['label'] == 'Not fraud'), LEGIT, 'circle', 14),
    ('Provider · fraud', (nodes['kind'] == 'provider') & (nodes['label'] == 'Fraud'), FRAUD, 'circle', 14),
]
for name, mask, color, symbol, size in groups:
    d = nodes[mask & (nodes['id'] != pid)]
    fig.add_scatter(x=d['x'], y=d['y'], mode='markers', name=name, text=d['id'],
                    marker=dict(color=color, symbol=symbol, size=size, line=dict(color='#fcfcfb', width=1.5)),
                    hovertemplate='%{text}<extra>' + name + '</extra>')
c = nodes[nodes['id'] == pid].iloc[0]
fig.add_scatter(x=[c['x']], y=[c['y']], mode='markers+text', name=f'{pid} (selected)', text=[pid],
                textposition='top center', textfont=dict(color=INK, size=13),
                marker=dict(color=LABEL_COLORS[row['label']], size=26, symbol='circle',
                            line=dict(color=INK, width=2.5)),
                hovertemplate=pid + '<extra>selected</extra>')
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False, scaleanchor='x')
show(style(fig, height=560))
n_prov = int(((nodes['kind'] == 'provider') & (nodes['id'] != pid)).sum())
n_fraud = int(((nodes['kind'] == 'provider') & (nodes['id'] != pid) & (nodes['label'] == 'Fraud')).sum())
st.caption(
    f'Showing up to 15 physicians and 25 beneficiaries with the most claims, and the {n_prov} other providers '
    f'most strongly linked through them ({n_fraud} labelled fraud). '
    'A provider surrounded by red is what the graph features and the GNN pick up on.'
)

# --- Claims ---
st.subheader('Claims')
claims = load_claims()
mine = claims[claims['Provider'] == pid].sort_values('ClaimStartDt')
monthly = mine.set_index('ClaimStartDt').resample('MS')['InscClaimAmtReimbursed'].agg(['count', 'sum'])
fig = go.Figure(go.Bar(x=monthly.index, y=monthly['count'], marker_color=LABEL_COLORS[row['label']],
                       customdata=monthly['sum'],
                       hovertemplate='%{x|%b %Y}: %{y:,} claims · $%{customdata:,.0f}<extra></extra>'))
fig.update_yaxes(title='Claims per month')
fig.update_xaxes(dtick='M1', tickformat='%b %Y')
show(style(fig, height=240, legend=False))
st.dataframe(
    mine, hide_index=True, height=300,
    column_order=['ClaimID', 'ClaimType', 'ClaimStartDt', 'ClaimEndDt', 'InscClaimAmtReimbursed',
                  'DeductibleAmtPaid', 'BeneID', 'AttendingPhysician', 'OperatingPhysician',
                  'num_diagnoses', 'num_procedures', 'ClaimDuration', 'State'],
    column_config={
        'ClaimStartDt': st.column_config.DateColumn('Start'),
        'ClaimEndDt': st.column_config.DateColumn('End'),
        'InscClaimAmtReimbursed': st.column_config.NumberColumn('Reimbursed', format='dollar'),
        'DeductibleAmtPaid': st.column_config.NumberColumn('Deductible', format='dollar'),
        'num_diagnoses': 'Diagnoses', 'num_procedures': 'Procedures', 'ClaimDuration': 'Days',
    },
)
