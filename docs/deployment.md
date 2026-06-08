# Delivery and Consumption

The project delivers a segmented customer base, an interactive dashboard, and a prediction
API for downstream systems and BI tools.

## Channels

- **Batch:** `reports/cluster_assignments.csv` with one cluster per customer.
- **Streamlit dashboard:** `app/streamlit_app.py` to explore segment volume, average profile, and records.
- **REST API:** `src/bank_marketing_strategy/api.py` (FastAPI) serves predictions from the trained pipeline.
- **BI:** Metabase or Power BI connected to a SQL table loaded from `reports/cluster_assignments.csv`.
- **Cloud (reference architecture):** S3 for file storage, EC2 (or any container host) for scheduled runs, and RDS/Postgres to expose the table to BI tools.

## Run with Docker (recommended)

The repository ships with a `Dockerfile`, an `entrypoint.sh`, and a `docker-compose.yml` that
automate training and serving — no manual environment setup required:

```bash
docker compose up --build
```

This starts two services:

- `app` — the Streamlit dashboard at [http://localhost:8501](http://localhost:8501)
- `api` — the FastAPI prediction service at [http://localhost:8000/docs](http://localhost:8000/docs)

On first run, each container automatically trains the model (`bank_marketing_strategy.cli train`)
if `models/model.joblib` is not present yet, then launches the corresponding service. Generated
artifacts (`models/`, `reports/`, `data/`) are mounted as volumes so they persist across restarts
and are shared between services.

To run a single service:

```bash
docker compose up app   # dashboard only
docker compose up api   # API only
```

To only run the training pipeline inside a container:

```bash
docker compose run --rm app train
```

## Run locally (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-app.txt -r requirements-api.txt
PYTHONPATH=src python -m bank_marketing_strategy.cli train
PYTHONPATH=src streamlit run app/streamlit_app.py
# in another terminal
PYTHONPATH=src uvicorn bank_marketing_strategy.api:app --reload
```

## Consumption

The marketing team uses the dashboard to pick segments, review average indicators, and decide
on retention, credit-limit, or financial-education campaigns. Other systems can call the
`/predict` endpoint of the API to score new customers in real time.
