"""Serving layer: load the segmentation pipeline and assign a customer to a segment."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, artifact_path: Path = config.PIPELINE_PATH) -> None:
        import joblib

        if not Path(artifact_path).exists():
            raise FileNotFoundError(f"No artifact at {artifact_path}. Run `python -m src.pipeline` first.")
        art = joblib.load(artifact_path)
        self.pipeline = art["pipeline"]
        self.pca = art["pca"]
        self.best_k: int = art["best_k"]
        self.profiles: Dict[int, Any] = {int(k): v for k, v in art["profiles"].items()}
        self.medians: Dict[str, float] = art["medians"]
        self.pca_sample: Dict[str, Any] = art["pca_sample"]

    def _row(self, features: Dict[str, Any]) -> pd.DataFrame:
        full = {c: self.medians.get(c, 0.0) for c in config.FEATURES}
        full.update({k: v for k, v in features.items() if k in config.FEATURES})
        return pd.DataFrame([full])[list(config.FEATURES)]

    def assign(self, features: Dict[str, Any]) -> Dict[str, Any]:
        row = self._row(features)
        seg = int(self.pipeline.predict(row)[0])
        coords = self.pca.transform(self.pipeline.named_steps["scale"].transform(
            self.pipeline.named_steps["prep"].transform(row)))[0]
        prof = self.profiles[seg]
        return {"segment": seg, "label": prof["label"], "share_pct": prof["share_pct"],
                "coords": [float(coords[0]), float(coords[1])], "profile_means": prof["means"]}

    def segment_table(self) -> List[Dict[str, Any]]:
        return [{"segment": s, "share_pct": p["share_pct"], "profile": p["label"]}
                for s, p in sorted(self.profiles.items())]
