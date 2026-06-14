"""Smoke tests for the segmentation contract and the serving surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.predict import Predictor  # noqa: E402
from src.preprocessing import FeaturePrep, Preprocessor  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return pd.read_csv(config.SAMPLE_PATH)


def test_features_selected_and_id_dropped(sample):
    X, _ = Preprocessor().run(sample)
    assert config.ID_COL not in X.columns
    assert all(c in X.columns for c in config.FEATURES)


def test_feature_prep_log_transform_and_impute(sample):
    prep = FeaturePrep().fit(sample)
    out = prep.transform(sample.head(20))
    assert list(out.columns) == list(config.FEATURES)
    assert out.notna().all().all()  # medians imputed


def test_predictor_assigns_valid_segment():
    pred = Predictor()
    customer = {"BALANCE": 2000, "PURCHASES": 3000, "CASH_ADVANCE": 0, "CREDIT_LIMIT": 6000,
                "PAYMENTS": 2500, "PURCHASES_FREQUENCY": 0.9, "PRC_FULL_PAYMENT": 0.3}
    res = pred.assign(customer)
    assert 0 <= res["segment"] < pred.best_k
    assert isinstance(res["label"], str) and res["label"]
    assert len(res["coords"]) == 2
    assert len(pred.segment_table()) == pred.best_k


def test_cash_advance_customer_differs_from_spender():
    pred = Predictor()
    spender = {"BALANCE": 1500, "PURCHASES": 4000, "CASH_ADVANCE": 0, "CREDIT_LIMIT": 6000,
               "PAYMENTS": 3500, "PURCHASES_FREQUENCY": 1.0, "PRC_FULL_PAYMENT": 0.5}
    revolver = {"BALANCE": 3000, "PURCHASES": 50, "CASH_ADVANCE": 4000, "CREDIT_LIMIT": 4000,
                "PAYMENTS": 1500, "PURCHASES_FREQUENCY": 0.0, "PRC_FULL_PAYMENT": 0.0}
    assert pred.assign(spender)["segment"] != pred.assign(revolver)["segment"]
