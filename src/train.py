"""Modeling layer for customer segmentation: select k by clustering quality,
fit the final KMeans pipeline, profile and name the segments, project to 2D for
visualization, and serialize a self-contained artifact plus a model card.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.preprocessing import FeaturePrep

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "1.0"


class ClusterTrainer:
    def __init__(self, X: pd.DataFrame, data_source: Path | None = None) -> None:
        self.X = X.reset_index(drop=True)
        self.data_source = data_source
        self.prep = FeaturePrep().fit(self.X)
        self.scaler = StandardScaler().fit(self.prep.transform(self.X))
        self.Xs = self.scaler.transform(self.prep.transform(self.X))
        self.selection: List[Dict[str, Any]] = []
        self.best_k = 0

    def select_k(self) -> List[Dict[str, Any]]:
        rng = min(8000, len(self.Xs))
        idx = np.random.default_rng(config.SEED).choice(len(self.Xs), rng, replace=False)
        for k in config.K_RANGE:
            km = KMeans(n_clusters=k, n_init="auto", random_state=config.SEED).fit(self.Xs)
            labels = km.labels_
            row = {
                "k": k,
                "silhouette": round(float(silhouette_score(self.Xs[idx], labels[idx])), 4),
                "davies_bouldin": round(float(davies_bouldin_score(self.Xs, labels)), 4),
                "calinski_harabasz": round(float(calinski_harabasz_score(self.Xs, labels)), 1),
            }
            self.selection.append(row)
            logger.info("k=%d silhouette=%.4f DB=%.4f CH=%.0f", k, row["silhouette"],
                        row["davies_bouldin"], row["calinski_harabasz"])
        self.best_k = max(self.selection, key=lambda r: r["silhouette"])["k"]
        logger.info("Selected k=%d by silhouette", self.best_k)
        return self.selection

    def fit(self) -> None:
        self.kmeans = KMeans(n_clusters=self.best_k, n_init="auto", random_state=config.SEED).fit(self.Xs)
        self.labels = self.kmeans.labels_
        self.pipeline = Pipeline([("prep", self.prep), ("scale", self.scaler), ("kmeans", self.kmeans)])
        self.pca = PCA(n_components=2, random_state=config.SEED).fit(self.Xs)
        self._profile()

    def _profile(self) -> None:
        df = self.X.copy()
        df["segment"] = self.labels
        prof = df.groupby("segment")[list(config.PROFILE_FEATURES)].mean()
        self.sizes = df["segment"].value_counts().sort_index().to_dict()
        gmean, gstd = df[list(config.PROFILE_FEATURES)].mean(), df[list(config.PROFILE_FEATURES)].std(ddof=0)
        self.profiles: Dict[int, Dict[str, Any]] = {}
        for seg in sorted(self.sizes):
            means = prof.loc[seg]
            z = ((means - gmean) / gstd.replace(0, 1)).sort_values()
            low, high = z.index[0], z.index[-1]
            descriptor = []
            if z[high] > 0.4:
                descriptor.append(f"high {high.replace('_', ' ').lower()}")
            if z[low] < -0.4:
                descriptor.append(f"low {low.replace('_', ' ').lower()}")
            self.profiles[int(seg)] = {
                "size": int(self.sizes[seg]),
                "share_pct": round(100 * self.sizes[seg] / len(df), 1),
                "label": ", ".join(descriptor).capitalize() or "Average profile",
                "means": {c: round(float(means[c]), 2) for c in config.PROFILE_FEATURES},
            }

    def evaluate(self) -> Dict[str, Any]:
        best = next(r for r in self.selection if r["k"] == self.best_k)
        return {"selected_k": self.best_k, "quality": best, "segment_sizes": self.sizes}

    def to_business_metrics(self) -> Dict[str, Any]:
        actions = []
        for seg, p in self.profiles.items():
            actions.append({"segment": seg, "share_pct": p["share_pct"],
                            "profile": p["label"], "suggested_action": _action(p["label"])})
        return {"segments": len(self.profiles), "actions": actions}

    def save(self, evaluation: Dict[str, Any], business: Dict[str, Any]) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        sample_coords = self.pca.transform(self.Xs)
        ix = np.random.default_rng(config.SEED).choice(len(sample_coords), min(2000, len(sample_coords)), replace=False)
        joblib.dump({
            "schema_version": SCHEMA_VERSION, "pipeline": self.pipeline, "pca": self.pca,
            "best_k": self.best_k, "profiles": self.profiles, "medians": self.prep.medians_.to_dict(),
            "pca_sample": {"coords": sample_coords[ix].tolist(), "labels": self.labels[ix].tolist()},
            "feature_columns": list(config.FEATURES),
        }, config.PIPELINE_PATH)
        logger.info("Pipeline artifact written to %s", config.PIPELINE_PATH)
        card = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": config.KAGGLE_DATASET, "data_sha256": self._hash(),
            "problem": "customer segmentation (KMeans clustering)",
            "k_selection": self.selection, "selected_k": self.best_k,
            "evaluation": evaluation, "business": business,
            "segment_profiles": self.profiles,
        }
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        logger.info("Model card written to %s", config.MODEL_CARD_PATH)

    def _hash(self) -> str:
        src = self.data_source or config.SAMPLE_PATH
        return hashlib.sha256(Path(src).read_bytes()).hexdigest() if src and Path(src).exists() else "unknown"


def _action(label: str) -> str:
    l = label.lower()
    if "cash advance" in l:
        return "Revolving / cash-reliant: monitor risk, offer structured credit."
    if "purchases" in l and "high" in l:
        return "Active spenders: reward and cross-sell premium products."
    if "balance" in l and "low purchases" in l:
        return "Dormant high-balance: reactivation and upsell campaigns."
    if "prc full payment" in l:
        return "Full-payers / transactors: retain with loyalty perks."
    return "Average usage: standard engagement and monitoring."
