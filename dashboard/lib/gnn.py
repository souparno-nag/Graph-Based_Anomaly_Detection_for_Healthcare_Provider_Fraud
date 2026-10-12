"""The heterogeneous GNN from the notebook, plus a cached loader for the saved weights."""
import json

import joblib
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from lib.data import ART


class FraudGNN(torch.nn.Module):
    """Must match the class in hda_prototype.ipynb exactly, or the saved weights won't load."""

    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = HeteroConv({
            ('provider', 'to', 'beneficiary'): SAGEConv((-1, -1), hidden_channels),
            ('beneficiary', 'to', 'provider'): SAGEConv((-1, -1), hidden_channels),
            ('provider', 'to', 'physician'): SAGEConv((-1, -1), hidden_channels),
            ('physician', 'to', 'provider'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        return torch.sigmoid(self.lin(x_dict['provider']))


@st.cache_resource(show_spinner='Loading the GNN and the full graph...')
def load_gnn():
    """(model, HeteroData, provider id -> node index)."""
    meta = json.loads((ART / 'metadata.json').read_text())
    data = torch.load(ART / 'hetero_data.pt', weights_only=False)
    model = FraudGNN(hidden_channels=meta['gnn_hidden_channels'])
    with torch.no_grad():
        model(data.x_dict, data.edge_index_dict)  # materialise the lazy (-1) input sizes
    model.load_state_dict(torch.load(ART / 'gnn_state.pt'))
    model.eval()
    provider_ids = joblib.load(ART / 'id_maps.joblib')['provider']
    return model, data, provider_ids
