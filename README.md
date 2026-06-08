# Bank Marketing Strategy

A data science portfolio project based on the Kaggle dataset [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

The project follows the end-to-end machine learning project workflow described in
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (Aurélien Géron): frame the
business problem, explore and prepare the data, build a reproducible pipeline, evaluate
with the right metrics, and ship the result through channels that real users (a marketing
team, in this case) can actually consume.

## Quick Start for Recruiters

You don't need to set up a Python environment to see this project working. With
[Docker](https://www.docker.com/) installed, run:

```bash
git clone <THIS_REPOSITORY_URL> bank-marketing-strategy
cd bank-marketing-strategy
# place the Kaggle CSV at data/raw/cc_general.csv (see data/raw/README.md)
docker compose up --build
```

Then open:

- **Dashboard:** [http://localhost:8501](http://localhost:8501) — interactive view of customer segments
- **API docs:** [http://localhost:8000/docs](http://localhost:8000/docs) — Swagger UI for the prediction endpoint

The first run automatically trains the model and generates the reports — there is nothing
else to configure. See [Section 12](#12-how-to-run) for local (non-Docker) instructions and
[`docs/deployment.md`](docs/deployment.md) for deployment details.

## 1. Business Problem

A bank marketing team needs to turn credit card behavior into actionable customer segments.

**Objective:** Segment credit card customers to support marketing, relationship, and offer strategies.

**Primary metric:** Silhouette, Davies-Bouldin, Calinski-Harabasz, and cluster profile.

## 2. Business Assumptions

- Segments must be interpretable, not just technically separable.
- Customers with distinct purchase, payment, and credit-usage patterns require different approaches.
- Choosing the number of clusters must combine technical metrics with business judgment.

## 3. Solution Strategy

1. **Step 01. Data Description:** Validate schema, dimensions, missing values, types, and granularity.
2. **Step 02. Feature Engineering:** Create variables oriented to the problem domain.
3. **Step 03. Data Filtering:** Remove records with no analytical value or leakage risk.
4. **Step 04. Exploratory Data Analysis:** Validate hypotheses and separate relevant signal from noise.
5. **Step 05. Data Preparation:** Impute, scale, and encode variables for modeling.
6. **Step 06. Feature Selection:** Separate IDs, target, input variables, and dropped columns.
7. **Step 07. Machine Learning Modelling:** Train a reproducible baseline and evaluate technical metrics.
8. **Step 08. Hyperparameter/Fit Strategy:** Reserve room for tuning, thresholds, and model comparison.
9. **Step 09. Business Translation:** Convert metrics into decisions, prioritization, risk, revenue, or operations.
10. **Step 10. Delivery:** Generate the segmented base, the Streamlit dashboard, and the BI layer for cluster consumption.

## 4. Data Source

Source on Kaggle: [Credit Card Dataset for Clustering](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata).

Expected file:

- `data/raw/cc_general.csv`

See [`data/raw/README.md`](data/raw/README.md) for download instructions via the Kaggle API.

## 5. Development Journey

The notebooks are organized to show the evolution of the analysis, from problem framing to
the business translation of the results. They are also the best place to grab screenshots
for a portfolio walkthrough:

- [`notebooks/00_business_understanding.ipynb`](notebooks/00_business_understanding.ipynb)
- [`notebooks/01_data_understanding.ipynb`](notebooks/01_data_understanding.ipynb)
- [`notebooks/02_exploratory_analysis.ipynb`](notebooks/02_exploratory_analysis.ipynb)
- [`notebooks/03_feature_engineering.ipynb`](notebooks/03_feature_engineering.ipynb)
- [`notebooks/04_modeling_and_business_results.ipynb`](notebooks/04_modeling_and_business_results.ipynb)
- [`notebooks/05_deployment_and_consumption.ipynb`](notebooks/05_deployment_and_consumption.ipynb)

## 6. Top Data Insights and Hypotheses

- Customers with high credit limit and low usage call for a different strategy than revolving customers.
- Purchase frequency separates transactional customers from occasional ones.
- Minimum payments and cash advances signal risk profile or a need for financial education.

## 7. Model or Analysis Applied

Clustering with KMeans, with the number of clusters selected via silhouette,
Davies-Bouldin, and Calinski-Harabasz scores, computed inside a single
`scikit-learn` `Pipeline` (imputation, scaling, encoding, and the estimator) so that
training and inference always apply the exact same transformations — one of the core
recommendations from the book's end-to-end project chapter.

## 8. Performance and Business Results

Data profile reproduced in [`reports/data_profile.json`](reports/data_profile.json): 8,950 rows and 18 columns analyzed.

Main pipeline outputs:

- `reports/cluster_assignments.csv`
- `reports/metrics.json`

## 9. Business Translation

Clusters become campaign groups: retention, credit-limit increase, financial education, or usage incentives.

## 10. Repository Structure

- `configs/project.toml`: project contract — data, target, metrics, and parameters.
- `src/bank_marketing_strategy/`: modular Python code for data, features, modeling, analysis, and serving.
- `notebooks/`: the analytical journey, in notebooks.
- `data/raw/`: files downloaded from Kaggle or the prepared analytical base.
- `reports/`: metrics, profiles, and results generated by the pipeline.
- `docs/deployment.md`: delivery and consumption notes.
- `app/streamlit_app.py`: dashboard for customer segments.
- `models/`: trained model artifact.
- `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh`: one-command, reproducible deployment (training + dashboard + API).
- `.github/workflows/ci.yml`: GitHub Actions pipeline that runs the test suite and builds the Docker image on every push.

## 11. How to Run on Google Colab

1. Open a new notebook on Google Colab.
2. Generate your token at Kaggle > Account > API > Create New Token.
3. Run the cells below.

Clone the repository and install the dependencies:

```python
REPO_URL = "https://github.com/<your-username>/<repository-name>.git"
!git clone {REPO_URL} project
%cd project
!python -m pip install -q -r requirements.txt
```

Download or prepare the data:

```python
from google.colab import files
files.upload()  # upload your kaggle.json file

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
!mkdir -p data/raw
!python -m pip install -q kaggle
!kaggle datasets download -d arjunbhasin2013/ccdata --unzip -p data/raw
!find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
!mv "data/raw/CC GENERAL.csv" data/raw/cc_general.csv 2>/dev/null || true
```

Run the main flow:

```python
!PYTHONPATH=src python -m bank_marketing_strategy.cli validate-config
!PYTHONPATH=src python -m bank_marketing_strategy.cli profile
!PYTHONPATH=src python -m bank_marketing_strategy.cli train
```

## 12. How to Run

### Option A — Docker (recommended, fully automated)

The fastest way to see the whole solution working — training, dashboard, and API — with a
single command (see the [Quick Start for Recruiters](#quick-start-for-recruiters) above):

```bash
docker compose up --build
```

### Option B — Local Python environment

Batch / pipeline mode:

```bash
git clone <REPOSITORY_URL> project
cd project
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=src python -m bank_marketing_strategy.cli profile
PYTHONPATH=src python -m bank_marketing_strategy.cli train
```

Dashboard and API:

```bash
python -m pip install -r requirements-app.txt -r requirements-api.txt
PYTHONPATH=src python -m bank_marketing_strategy.cli train
PYTHONPATH=src streamlit run app/streamlit_app.py
# in another terminal
PYTHONPATH=src uvicorn bank_marketing_strategy.api:app --reload
```

For BI, load `reports/cluster_assignments.csv` into a SQL table and connect Metabase or Power BI to it.
More details in [`docs/deployment.md`](docs/deployment.md).

## 13. Next Steps to Improve

- Name cluster personas using descriptive statistics.
- Build a recommended-action matrix per segment.
- Test cluster stability across time windows.

## 14. Tests

```bash
python -m pytest
```

Continuous integration runs the test suite and builds the Docker image on every push to `main`
(see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).
