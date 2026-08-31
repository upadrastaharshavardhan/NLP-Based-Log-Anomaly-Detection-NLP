"""Smoke tests for Project 4."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.generator import generate_log_dataset
from src.data.preprocessing import LogPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.anomaly import AnomalyDetectorModel

def test_generator():
    df = generate_log_dataset(200, anomaly_ratio=0.15, seed=1)
    assert len(df) == 200
    assert df["is_anomaly"].sum() > 0

def test_tiny():
    df = generate_log_dataset(300, seed=2)
    pre = LogPreprocessor()
    df = pre.transform_df(df)
    emb = EmbeddingModel(device="cpu")
    X = emb.encode(df["cleaned_text"].tolist(), batch_size=32, show_progress=False)
    normal = X[df["is_anomaly"].values == 0]
    det = AnomalyDetectorModel(contamination=0.15)
    det.fit(normal if len(normal) > 20 else X)
    preds = det.predict(X[:20])
    assert len(preds) == 20

if __name__ == "__main__":
    test_generator()
    test_tiny()
    print("OK")
