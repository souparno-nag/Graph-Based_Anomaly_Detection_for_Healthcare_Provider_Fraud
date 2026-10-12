# Graph-Based Anomaly Detection for Healthcare Provider Fraud

Detecting potentially fraudulent Medicare providers by combining each provider's billing
behaviour with the network of beneficiaries and physicians it is connected to.

Data: [Kaggle – Healthcare Provider Fraud Detection Analysis](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)
(5,410 labelled providers, 558,211 inpatient and outpatient claims).

## Models

| Model | Test PR-AUC |
|---|---|
| XGBoost on 12 billing features | 0.699 |
| XGBoost + graph features (Louvain community, fraud-neighbour density) | 0.713 |
| Heterogeneous GNN (GraphSAGE over provider–beneficiary–physician links) | 0.753 |
| Stacked ensemble (scored on half the test set) | 0.798 |

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. **Notebook** – `hda_prototype.ipynb` builds the features, graph and models, then exports everything
   the dashboard needs to `artifacts/` (the last section). Skip the first cell; it is a Colab upload
   step and the CSVs are already in `data/`.
2. **Dashboard** – from the repo root:

   ```bash
   streamlit run dashboard/app.py
   ```

## Dashboard pages

- **Overview** – the dataset and the pipeline at a glance.
- **Model performance** – PR and ROC curves, a live decision-threshold slider, confusion matrix and
  5-fold cross-validation.
- **Provider explorer** – ranked risk list; for any provider, every model's score, SHAP explanation,
  its network neighbourhood, its claims, and live what-if tests on the GNN.
- **Graph insights** – community fraud rates, the fraud-neighbour density feature, and which links the
  GNN relies on.

## Caveats

- `fraud_neighbor_density` is computed from all providers' labels, including test providers, so the
  graph-augmented XGBoost's gain is likely overstated.
- Labels mark *potential* fraud at the provider level, not proven fraud on individual claims.
- Scores for training-set providers are optimistic; the dashboard defaults to test providers.
