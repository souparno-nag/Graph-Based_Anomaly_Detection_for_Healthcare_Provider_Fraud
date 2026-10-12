"""Cached loaders for everything the notebook exported to artifacts/."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parents[2] / 'artifacts'

# Score column -> display name, in a fixed order (also fixes each model's colour).
MODELS = {
    'p_xgb_baseline': 'XGBoost (baseline)',
    'p_xgb_graph': 'XGBoost + graph',
    'p_gnn': 'Full GNN',
    'p_ens_avg': 'Ensemble (average)',
    'p_ens_weighted': 'Ensemble (weighted)',
    'p_ens_stacked': 'Ensemble (stacked)',
}
LABELS = {0: 'Not fraud', 1: 'Fraud'}

FEATURE_NAMES = {
    'num_claims': 'Claims',
    'total_reimbursed': 'Total reimbursed ($)',
    'avg_reimbursed': 'Avg reimbursed per claim ($)',
    'std_reimbursed': 'Std of reimbursement ($)',
    'num_unique_beneficiaries': 'Unique beneficiaries',
    'num_unique_physicians': 'Unique physicians',
    'num_inpatient': 'Inpatient claims',
    'num_outpatient': 'Outpatient claims',
    'avg_num_diagnoses': 'Avg diagnoses per claim',
    'avg_num_procedures': 'Avg procedures per claim',
    'claims_per_beneficiary': 'Claims per beneficiary',
    'claims_per_physician': 'Claims per physician',
    'community_id': 'Louvain community ID',
    'fraud_neighbor_density': 'Fraud-neighbour density',
}
GRAPH_FEATURES = {'community_id', 'fraud_neighbor_density'}


@st.cache_data
def load_metadata() -> dict:
    return json.loads((ART / 'metadata.json').read_text())


@st.cache_data
def load_scores() -> pd.DataFrame:
    df = pd.read_parquet(ART / 'provider_scores.parquet')
    df['label'] = df['PotentialFraud'].map(LABELS)
    return df


def eval_rows(scores: pd.DataFrame, model_col: str) -> pd.DataFrame:
    """Held-out rows a model can be fairly scored on.

    The stacked ensemble was fitted on half of the test set, so it is only
    evaluated on the other half (stack_eval); every other model uses the full test set.
    """
    mask = scores['split'] == 'test'
    if model_col == 'p_ens_stacked':
        mask &= scores['stack_eval']
    return scores[mask]


@st.cache_data
def load_claims() -> pd.DataFrame:
    return pd.read_parquet(ART / 'claims_slim.parquet')


@st.cache_data
def load_edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(provider-beneficiary, provider-physician) edge lists, one row per unique pair."""
    pb = pd.read_parquet(ART / 'edges_provider_beneficiary.parquet')
    pp = pd.read_parquet(ART / 'edges_provider_physician.parquet')
    return pb, pp


@st.cache_data
def load_shap(name: str) -> tuple[pd.DataFrame, float]:
    """SHAP values for every provider ('baseline' or 'graph') and the explainer's base value."""
    sv = pd.read_parquet(ART / f'shap_{name}.parquet').set_index('Provider')
    base = json.loads((ART / f'shap_{name}_base.json').read_text())['expected_value']
    return sv, base


@st.cache_data
def load_metrics() -> pd.DataFrame:
    return pd.read_csv(ART / 'metrics_final.csv')


@st.cache_data
def load_cv() -> pd.DataFrame:
    return pd.read_csv(ART / 'cv_folds.csv')


@st.cache_data
def load_ablation() -> pd.DataFrame:
    return pd.read_csv(ART / 'gnn_edge_ablation.csv')
