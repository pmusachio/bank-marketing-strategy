"""Central configuration: paths, dataset identity, clustering constants and the
Dracula palette shared by the pipeline, the serving layer and the dashboard.
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SAMPLE_DIR: Path = DATA_DIR / "sample"
MODELS_DIR: Path = BASE_DIR / "models"

PIPELINE_PATH: Path = MODELS_DIR / "pipeline.joblib"
MODEL_CARD_PATH: Path = MODELS_DIR / "model_card.json"
PROCESSED_PATH: Path = PROCESSED_DIR / "segments.parquet"

SAMPLE_FILENAME: str = "cc_general_sample.csv"
SAMPLE_PATH: Path = SAMPLE_DIR / SAMPLE_FILENAME

KAGGLE_DATASET: str = "arjunbhasin2013/ccdata"
RAW_FILENAME: str = "cc_general.csv"
ID_COL: str = "CUST_ID"

FEATURES: tuple[str, ...] = (
    "BALANCE", "BALANCE_FREQUENCY", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE", "PURCHASES_FREQUENCY", "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY", "CASH_ADVANCE_FREQUENCY", "CASH_ADVANCE_TRX",
    "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS", "PRC_FULL_PAYMENT", "TENURE",
)
# Heavily right-skewed monetary / count features get a log1p transform before scaling.
LOG_FEATURES: tuple[str, ...] = (
    "BALANCE", "PURCHASES", "ONEOFF_PURCHASES", "INSTALLMENTS_PURCHASES", "CASH_ADVANCE",
    "CASH_ADVANCE_TRX", "PURCHASES_TRX", "CREDIT_LIMIT", "PAYMENTS", "MINIMUM_PAYMENTS",
)
# Subset surfaced in the app; the rest are filled with training medians.
APP_FEATURES: tuple[str, ...] = (
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
    "PURCHASES_FREQUENCY", "PRC_FULL_PAYMENT",
)
# Features that summarize a segment's behaviour in the profile table.
PROFILE_FEATURES: tuple[str, ...] = (
    "BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS",
    "PURCHASES_FREQUENCY", "PRC_FULL_PAYMENT", "TENURE",
)

K_RANGE: tuple[int, ...] = (3, 4, 5, 6, 7, 8)
SEED: int = 42

DRACULA = {
    "background": "#282a36", "current_line": "#44475a", "foreground": "#f8f8f2",
    "comment": "#6272a4", "cyan": "#8be9fd", "green": "#50fa7b", "orange": "#ffb86c",
    "pink": "#ff79c6", "purple": "#bd93f9", "red": "#ff5555", "yellow": "#f1fa8c",
}
SEGMENT_COLORS: tuple[str, ...] = (
    "#bd93f9", "#8be9fd", "#50fa7b", "#ffb86c", "#ff79c6", "#f1fa8c", "#ff5555", "#6272a4",
)
