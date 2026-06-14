"""Transformation layer. Median imputation and a log1p transform of skewed monetary
features live in a custom first pipeline step so training and serving share the
identical transform; scaling follows inside the pipeline.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

logger = logging.getLogger(__name__)


class FeaturePrep(BaseEstimator, TransformerMixin):
    """Imputes with learned medians and log1p-transforms skewed features."""

    def fit(self, X, y=None):
        df = pd.DataFrame(X)[list(config.FEATURES)].apply(pd.to_numeric, errors="coerce")
        self.medians_ = df.median()
        return self

    def transform(self, X) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        for c in config.FEATURES:
            if c not in df.columns:
                df[c] = np.nan
        df = df[list(config.FEATURES)].apply(pd.to_numeric, errors="coerce")
        df = df.fillna(self.medians_)
        for c in config.LOG_FEATURES:
            df[c] = np.log1p(df[c].clip(lower=0))
        return df


class Preprocessor:
    def __init__(self, processed_path=config.PROCESSED_PATH) -> None:
        self.processed_path = processed_path

    def run(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        features = df[[c for c in config.FEATURES if c in df.columns]].copy()
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Prepared %d customers x %d features", *features.shape)
        return features, df
