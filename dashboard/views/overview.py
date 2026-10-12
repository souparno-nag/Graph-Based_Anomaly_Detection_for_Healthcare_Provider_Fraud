import plotly.graph_objects as go
import streamlit as st

from lib.charts import FRAUD, INK_2, LABEL_COLORS, LEGIT, MUTED, show, style
from lib.data import load_claims, load_metadata, load_scores

scores = load_scores()
claims = load_claims()
meta = load_metadata()

st.title('Can the network expose fraudulent providers?')
st.markdown(
    'Medicare providers are labelled **potentially fraudulent** or not. Each provider is linked '
    'to the **beneficiaries** they billed for and the **physicians** who attended those claims. '
    'This dashboard compares models that only see each provider\'s own billing numbers with models '
    'that also see who the provider is connected to.'
)

# --- Headline numbers ---
fraud_claims = claims['PotentialFraud'] == 'Yes'
n_fraud = int(scores['PotentialFraud'].sum())
dollar_share = claims.loc[fraud_claims, 'InscClaimAmtReimbursed'].sum() / claims['InscClaimAmtReimbursed'].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric('Providers', f"{len(scores):,}", border=True)
c2.metric('Claims', f"{len(claims):,}", border=True)
c3.metric('Labelled fraud', f"{n_fraud:,}", f"{n_fraud / len(scores):.1%} of providers",
          delta_color='off', delta_arrow='off', border=True)
c4.metric('Dollars to fraud-labelled', f"{dollar_share:.0%}",
          f"${claims.loc[fraud_claims, 'InscClaimAmtReimbursed'].sum() / 1e6:,.0f}M",
          delta_color='off', delta_arrow='off', border=True)

# --- Pipeline ---
st.subheader('How it works')
steps = [
    ('1 · Claims', f"{len(claims):,} inpatient and outpatient claims, joined to beneficiaries"),
    ('2 · Billing features', '12 numbers per provider: claim counts, amounts, patients, physicians'),
    ('3 · Graph', f"{meta['graph']['nodes']:,} provider, beneficiary and physician nodes, "
                  f"{meta['graph']['edges']:,} edges, {meta['n_communities']:,} Louvain communities"),
    ('4 · Models', 'XGBoost on billing only, XGBoost + graph features, a heterogeneous GNN, '
                   'and three ensembles'),
]
for col, (head, body) in zip(st.columns(len(steps)), steps):
    with col.container(border=True, height='stretch'):
        st.markdown(f'**{head}**')
        st.caption(body)

# --- Charts ---
left, right = st.columns(2)

with left:
    st.subheader('A small share of providers, a large share of money')
    shares = {
        'Providers': n_fraud / len(scores),
        'Claims': fraud_claims.mean(),
        'Dollars reimbursed': dollar_share,
    }
    fig = go.Figure(go.Bar(
        y=list(shares), x=list(shares.values()), orientation='h', marker_color=FRAUD,
        text=[f'{v:.0%}' for v in shares.values()], textposition='outside',
        textfont_color=INK_2, hovertemplate='%{y}: %{x:.1%}<extra></extra>',
    ))
    fig.update_xaxes(range=[0, 0.75], tickformat='.0%', title='Share from fraud-labelled providers')
    fig.update_yaxes(autorange='reversed')
    show(style(fig, height=260, legend=False))

    st.subheader('Per-claim amounts are almost identical')
    by_type = (claims.assign(label=fraud_claims.map({True: 'Fraud', False: 'Not fraud'}))
               .groupby(['ClaimType', 'label'])['InscClaimAmtReimbursed'].mean().reset_index())
    fig = go.Figure()
    for label in ['Not fraud', 'Fraud']:
        d = by_type[by_type['label'] == label]
        fig.add_bar(x=d['ClaimType'], y=d['InscClaimAmtReimbursed'], name=label,
                    marker_color=LABEL_COLORS[label], text=d['InscClaimAmtReimbursed'].map('${:,.0f}'.format),
                    textposition='outside', textfont_color=INK_2,
                    hovertemplate=f'{label} · %{{x}}: $%{{y:,.0f}} per claim<extra></extra>')
    fig.update_layout(barmode='group', bargroupgap=0.08)
    fig.update_yaxes(title='Avg reimbursed per claim ($)', tickprefix='$')
    show(style(fig, height=320))
    st.caption('So the money gap above comes from volume, not from pricier claims.')

with right:
    st.subheader('Fraud-labelled providers file far more claims')
    fig = go.Figure()
    for label in ['Not fraud', 'Fraud']:
        d = scores[scores['label'] == label]
        fig.add_box(x=d['num_claims'], name=label, marker_color=LABEL_COLORS[label],
                    line_width=1.5, boxpoints='outliers', marker_size=4,
                    hovertemplate=f'{label}: %{{x:,}} claims<extra></extra>')
    fig.update_xaxes(type='log', title='Claims per provider (log scale)')
    show(style(fig, height=260, legend=False))

    st.subheader('Share of claims from fraud-labelled providers, by state')
    by_state = (claims.groupby('State')['PotentialFraud'].apply(lambda s: (s == 'Yes').mean())
                .sort_values(ascending=False).reset_index(name='share'))
    overall = fraud_claims.mean()
    fig = go.Figure(go.Bar(
        x=by_state['State'].astype(str), y=by_state['share'], marker_color=LEGIT,
        hovertemplate='State %{x}: %{y:.1%} of claims<extra></extra>',
    ))
    fig.add_hline(y=overall, line_dash='dot', line_color=MUTED, line_width=1.5,
                  annotation_text=f'All states: {overall:.0%}', annotation_font_color=INK_2,
                  annotation_position='top right')
    fig.update_xaxes(type='category', title='State code (anonymised)', tickfont_size=10)
    fig.update_yaxes(tickformat='.0%', title='Fraud share of claims')
    show(style(fig, height=320, legend=False))

st.caption(
    '"Fraud" means the provider is labelled *PotentialFraud = Yes* in the dataset. '
    'Every claim inherits its provider\'s label.'
)
