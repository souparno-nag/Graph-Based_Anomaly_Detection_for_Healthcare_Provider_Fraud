import plotly.graph_objects as go
import streamlit as st

from lib.charts import (BENEFICIARY, FRAUD, INK_2, LABEL_COLORS, LEGIT, MUTED, PHYSICIAN, show, style)
from lib.data import load_ablation, load_edges, load_metadata, load_scores

scores = load_scores()
meta = load_metadata()
pb, pp = load_edges()

st.title('What the graph reveals')
st.markdown(
    'The graph links each provider to the beneficiaries it billed for and the physicians on its claims. '
    'Two providers are "neighbours" when they share a beneficiary or a physician. This page asks what '
    'that structure adds.'
)

# --- Graph at a glance ---
phys_deg = pp.groupby('AttendingPhysician').size()
bene_deg = pb.groupby('BeneID').size()
comm = scores.groupby('community_id')['PotentialFraud'].agg(['mean', 'size'])
big = comm[comm['size'] >= 5].sort_values('mean', ascending=False)

c = st.columns(4)
c[0].metric('Nodes', f"{meta['graph']['nodes']:,}", border=True)
c[1].metric('Edges', f"{meta['graph']['edges']:,}", border=True)
c[2].metric('Louvain communities', f"{meta['n_communities']:,}",
            f'{len(big)} contain 5+ providers', delta_color='off', delta_arrow='off', border=True)
c[3].metric('Single-provider physicians', f'{(phys_deg == 1).mean():.0%}',
            f'beneficiaries: {(bene_deg == 1).mean():.0%}', delta_color='off', delta_arrow='off', border=True)

# --- Communities ---
st.subheader('Fraud is not spread evenly across communities')
overall = scores['PotentialFraud'].mean()
fig = go.Figure(go.Bar(
    x=big.index.astype(str), y=big['mean'], customdata=big['size'],
    marker_color=[FRAUD if v > overall else LEGIT for v in big['mean']],
    hovertemplate='Community %{x}: %{y:.0%} fraud · %{customdata} providers<extra></extra>',
))
fig.add_hline(y=overall, line_dash='dot', line_color=MUTED, line_width=1.5,
              annotation_text=f'All providers: {overall:.1%}', annotation_font_color=INK_2,
              annotation_position='top right')
fig.update_xaxes(type='category', title='Louvain community (5+ providers), sorted by fraud rate',
                 showticklabels=False)
fig.update_yaxes(tickformat='.0%', title='Share labelled fraud')
show(style(fig, height=320, legend=False))
st.caption(
    f"Fraud rates range from {big['mean'].min():.0%} to {big['mean'].max():.0%} across these {len(big)} "
    'communities. Red bars are above the overall rate. Community ID is given to the graph-augmented XGBoost as '
    'a feature, though as a raw number it has no natural order.'
)

# --- Fraud-neighbour density ---
left, right = st.columns(2)
with left:
    st.subheader('Fraud-neighbour density by label')
    fig = go.Figure()
    for label in ['Not fraud', 'Fraud']:
        d = scores[scores['label'] == label]
        fig.add_histogram(x=d['fraud_neighbor_density'], name=label, histnorm='percent',
                          xbins=dict(start=0, end=0.8, size=0.025), marker_color=LABEL_COLORS[label],
                          opacity=0.6, hovertemplate=label + ': %{y:.1f}% of providers<extra></extra>')
    fig.update_layout(barmode='overlay')
    fig.update_xaxes(title='Share of 2-hop neighbours labelled fraud', tickformat='.0%')
    fig.update_yaxes(title='% of providers', ticksuffix='%')
    show(style(fig, height=340))
    means = scores.groupby('label')['fraud_neighbor_density'].mean()
    st.caption(
        f"Means: {means['Fraud']:.2f} for fraud vs {means['Not fraud']:.2f} for not fraud. On its own the feature "
        'barely separates the groups; it helps the model in combination with billing volume. It is also '
        'computed from every provider\'s label, test providers included, so its value is somewhat inflated.'
    )

with right:
    st.subheader('Neighbourhood vs GNN score (test set)')
    t = scores[scores['split'] == 'test']
    fig = go.Figure()
    for label in ['Not fraud', 'Fraud']:
        d = t[t['label'] == label]
        fig.add_scatter(x=d['fraud_neighbor_density'], y=d['p_gnn'], mode='markers', name=label,
                        text=d['Provider'],
                        marker=dict(color=LABEL_COLORS[label], size=8 if label == 'Fraud' else 6,
                                    opacity=0.85 if label == 'Fraud' else 0.45,
                                    line=dict(color='#fcfcfb', width=1)),
                        hovertemplate='%{text}<br>Density %{x:.2f} · GNN score %{y:.2f}<extra>' + label + '</extra>')
    fig.add_hline(y=meta['threshold'], line_dash='dash', line_color=INK_2, line_width=1)
    fig.update_xaxes(title='Fraud-neighbour density', tickformat='.0%')
    fig.update_yaxes(title='GNN fraud score', range=[-0.02, 1.02])
    show(style(fig, height=340))
    st.caption(f"Dashed line: the {meta['threshold']} threshold. High GNN scores are not explained by "
               'neighbourhood alone; the GNN also uses each provider\'s own billing features.')

# --- What the GNN listens to ---
left, right = st.columns(2)
with left:
    st.subheader('Which links does the GNN rely on?')
    abl = load_ablation()
    order = ['physician->provider', 'beneficiary->provider', 'provider->physician', 'provider->beneficiary']
    agg = abl.groupby('relation')['drop'].agg(['mean', 'std']).reindex(order)
    names = {'physician->provider': 'Physician → provider', 'beneficiary->provider': 'Beneficiary → provider',
             'provider->physician': 'Provider → physician', 'provider->beneficiary': 'Provider → beneficiary'}
    fig = go.Figure(go.Bar(
        y=[names[r] for r in agg.index], x=agg['mean'], orientation='h',
        error_x=dict(type='data', array=agg['std'], color=MUTED, thickness=1.2),
        marker_color=[PHYSICIAN, BENEFICIARY, MUTED, MUTED],
        hovertemplate='%{y}: mean drop %{x:+.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title='Score drop when removed')
    fig.update_yaxes(autorange='reversed')
    show(style(fig, height=300, legend=False))
    n = abl['provider_pos'].nunique()
    st.caption(
        f'Averaged over {n} fraud-labelled test providers (bars show ±1 std). Removing physician links drops '
        'the score most. Links pointing *out* of a provider show zero by design: the GNN has one layer and reads '
        'only provider nodes, so messages sent to physicians and beneficiaries never come back.'
    )

with right:
    st.subheader('How many providers does each node connect to?')
    cap = 8
    fig = go.Figure()
    for name, deg, color in [('Physicians', phys_deg, PHYSICIAN), ('Beneficiaries', bene_deg, BENEFICIARY)]:
        counts = deg.clip(upper=cap).value_counts(normalize=True).sort_index()
        fig.add_bar(x=[str(k) if k < cap else f'{cap}+' for k in counts.index], y=counts.values, name=name,
                    marker_color=color,
                    hovertemplate=name + ': %{y:.1%} connect to %{x} provider(s)<extra></extra>')
    fig.update_layout(barmode='group', bargroupgap=0.08)
    fig.update_xaxes(type='category', title='Providers connected to')
    fig.update_yaxes(tickformat='.0%', title='Share of nodes')
    show(style(fig, height=300))
    st.caption(
        f'{(phys_deg == 1).mean():.0%} of physicians work with just one provider, so physician links mostly '
        'summarise a provider\'s own claims. Beneficiaries are the real bridges: '
        f'{(bene_deg > 1).mean():.0%} of them visit two or more providers.'
    )
