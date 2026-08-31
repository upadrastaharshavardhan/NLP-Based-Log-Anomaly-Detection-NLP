#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from src.pipeline.detector import LogAnomalyDetector
from src.utils.helpers import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    detector = LogAnomalyDetector.load(args.artifacts, args.config)
    data_path = args.data or cfg["paths"]["raw_data"]
    df = pd.read_csv(data_path)
    if len(df) > 800:
        df = df.sample(800, random_state=42)
    results = detector.predict_batch(df["full_text"].tolist())
    y_pred = [1 if r["is_anomaly"] else 0 for r in results]
    y_true = df["is_anomaly"].tolist()
    scores = [r["anomaly_score"] for r in results]
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], digits=3))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_true, scores), 4))
    except Exception:
        pass

if __name__ == "__main__":
    main()
