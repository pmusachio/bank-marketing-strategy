# Bank Marketing Strategy — Customer Segmentation

> Unsupervised learning · KMeans clustering · Behavioural segmentation

## Business Problem

A bank wants to move from one-size-fits-all credit-card marketing to **profile-based strategy**.
The question is not a prediction but a structure: *what natural groups exist in how customers use
their cards*, so that each group can get a tailored action (rewards, upsell, risk monitoring,
reactivation).

There is no label to predict here, so this is an **unsupervised segmentation** problem rather than
classification. The "cost of error" is a poorly-defined segmentation that sends the wrong offer to
the wrong group — wasted marketing spend and missed cross-sell. The model is judged on cluster
quality (separation and cohesion) and, above all, on whether the segments are **interpretable and
actionable**.

## Dataset

[Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

| Property | Value |
|----------|-------|
| Rows | 8,950 active card holders |
| Features | 17 usage behaviours (balance, purchases, cash advance, frequencies, limit, payments, tenure) |
| Target | none (unsupervised) |
| Missing | `CREDIT_LIMIT` (1), `MINIMUM_PAYMENTS` (313), median-imputed |

## Solution Strategy

1. **Acquisition** — pull the dataset from Kaggle on demand; a versioned sample backs an offline run.
2. **Preparation** — median imputation and a log1p transform of the heavily right-skewed monetary and count features, inside the model `Pipeline` so serving reuses the exact transform; standardization follows.
3. **Model selection** — KMeans is fit across k = 3..8 and the number of segments is chosen by silhouette, cross-checked against Davies-Bouldin and Calinski-Harabasz.
4. **Profiling** — each segment is profiled on the original feature scale and labelled automatically from its most distinctive behaviours.
5. **Activation** — every segment is mapped to a suggested marketing action.

## Top Insights & Hypotheses

- **Three behavioural segments emerge** cleanly from card usage, each about a third of the base.
- **Active spenders** (high purchase frequency, high purchases) are the prime cross-sell and rewards audience.
- **Cash-advance revolvers** (high cash advance, almost no purchases) are a distinct risk-and-credit group, not a spending one.
- **Light / low-balance users** form the third group, a reactivation target.
- **Separation is moderate** (silhouette 0.23): card behaviour is continuous, so the segments are useful summaries rather than hard boundaries — noted in Next Steps.

## Model

KMeans on standardized, log-transformed features, with k chosen by clustering quality.

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|-----------:|---------------:|------------------:|
| **3 (selected)** | **0.226** | 1.681 | 2643 |
| 4 | 0.212 | 1.663 | 2256 |
| 5 | 0.218 | 1.595 | 2129 |
| 6 | 0.218 | 1.467 | 2009 |

The full preprocessing-plus-KMeans pipeline is serialized, so a new customer is assigned to a
segment with the identical transform used in training.

## Business Results

| Segment | Share | Profile | Suggested action |
|---------|------:|---------|------------------|
| 0 | 35% | High purchase frequency (active spenders) | Reward and cross-sell premium products. |
| 1 | 32% | Low balance (light users) | Standard engagement; reactivation. |
| 2 | 32% | High cash advance, low purchases (revolvers) | Monitor risk, offer structured credit. |

The bank can now run three targeted programs instead of one generic campaign, concentrating
rewards on spenders, credit products on revolvers and reactivation on light users.

## How to Run

1. **Clone**
   ```
   git clone https://github.com/pmusachio/bank-marketing-strategy.git
   cd bank-marketing-strategy
   ```
2. **Environment**
   ```
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Kaggle access** — place a Kaggle API token at `~/.kaggle/`; the pipeline falls back to the versioned sample if none is present.
4. **Run the pipeline**
   ```
   python -m src.pipeline
   ```
5. **Tests**
   ```
   pytest tests/
   ```
6. **App (local)**
   ```
   streamlit run app/streamlit_app.py
   ```
7. **Live app** — [huggingface.co/spaces/pmusachio/bank-marketing-strategy](https://huggingface.co/spaces/pmusachio/bank-marketing-strategy) — profile a customer and see their segment on the map.

## Next Steps

- Because separation is moderate, validate the segments with marketing-response data: the real test of a segmentation is whether segment-specific offers outperform a generic one.
- Try Gaussian mixture or hierarchical clustering for soft assignments, which may suit the continuous nature of card behaviour better than hard KMeans boundaries; deferred until the current segments are validated in market.
- Refresh the segmentation periodically, since usage behaviour drifts with the economy and product changes.
