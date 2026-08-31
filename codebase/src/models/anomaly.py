"""Unsupervised anomaly detectors on log embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class AnomalyDetectorModel:
    def __init__(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.12,
        n_estimators: int = 200,
        n_neighbors: int = 20,
        random_state: int = 42,
    ):
        self.method = method
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.model = None
        self._decision_scores_train = None

    def _build(self):
        if self.method == "isolation_forest":
            return IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.method == "lof":
            return LocalOutlierFactor(
                n_neighbors=self.n_neighbors,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
        elif self.method == "one_class_svm":
            return OneClassSVM(kernel="rbf", nu=self.contamination, gamma="scale")
        raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X: np.ndarray) -> "AnomalyDetectorModel":
        self.model = self._build()
        self.model.fit(X)
        if hasattr(self.model, "decision_function"):
            self._decision_scores_train = self.model.decision_function(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomaly, 0 for normal (sklearn uses -1/1)."""
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Higher = more normal for IsolationForest/OCSVM; we invert for anomaly score."""
        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Convert to anomaly score: higher = more anomalous
            return -scores
        if hasattr(self.model, "score_samples"):
            return -self.model.score_samples(X)
        return self.predict(X).astype(float)

    def predict_with_score(self, X: np.ndarray) -> List[Dict]:
        preds = self.predict(X)
        scores = self.decision_function(X)
        # Normalize scores to approx [0,1] using train distribution if available
        if self._decision_scores_train is not None:
            train_anom = -self._decision_scores_train
            lo, hi = np.percentile(train_anom, 1), np.percentile(train_anom, 99)
            denom = max(hi - lo, 1e-6)
            norm_scores = np.clip((scores - lo) / denom, 0, 1)
        else:
            norm_scores = scores
        results = []
        for p, s, ns in zip(preds, scores, norm_scores):
            results.append({
                "is_anomaly": bool(p),
                "anomaly_score": float(ns),
                "raw_score": float(s),
                "decision": "anomaly" if p else "normal",
            })
        return results

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "method": self.method,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "n_neighbors": self.n_neighbors,
            "random_state": self.random_state,
            "decision_scores_train": self._decision_scores_train,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetectorModel":
        data = joblib.load(path)
        obj = cls(
            method=data["method"],
            contamination=data["contamination"],
            n_estimators=data.get("n_estimators", 200),
            n_neighbors=data.get("n_neighbors", 20),
            random_state=data.get("random_state", 42),
        )
        obj.model = data["model"]
        obj._decision_scores_train = data.get("decision_scores_train")
        return obj
