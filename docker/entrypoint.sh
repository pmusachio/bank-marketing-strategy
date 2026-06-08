#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="${MODEL_FILE:-models/model.joblib}"

train_if_missing() {
  if [ ! -f "${MODEL_FILE}" ]; then
    echo "[entrypoint] No trained model found at ${MODEL_FILE}. Running the training pipeline..."
    python -m bank_marketing_strategy.cli train
  else
    echo "[entrypoint] Found trained model at ${MODEL_FILE}. Skipping training."
  fi
}

case "${1:-app}" in
  train)
    exec python -m bank_marketing_strategy.cli train
    ;;
  api)
    train_if_missing
    exec uvicorn bank_marketing_strategy.api:app --host 0.0.0.0 --port 8000
    ;;
  app)
    train_if_missing
    exec streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
    ;;
  *)
    exec "$@"
    ;;
esac
