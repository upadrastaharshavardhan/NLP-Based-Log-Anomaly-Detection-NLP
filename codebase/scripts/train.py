#!/usr/bin/env python
"""Train anomaly detector on log embeddings."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, confusion_matrix
)
import seaborn as sns

from src.data.generator import generate_log_dataset
from src.data.preprocessing import LogPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.anomaly import AnomalyDetectorModel
from src.utils.helpers import load_config, ensure_dirs, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["data"]["random_seed"])
    ensure_dirs(cfg["paths"]["data_dir"], cfg["paths"]["artifacts_dir"])

    data_path = Path(args.data or cfg["paths"]["raw_data"])
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        print("Generating data...")
        df = generate_log_dataset(
            n_samples=cfg["data"]["n_samples"],
            anomaly_ratio=cfg["data"]["anomaly_ratio"],
            seed=cfg["data"]["random_seed"],
        )
        df.to_csv(data_path, index=False)

    pre = LogPreprocessor(**{k: cfg["preprocessing"][k] for k in
                             ["max_text_length", "remove_timestamps", "remove_ids", "remove_hex"]
                             if k in cfg["preprocessing"]})
    df = pre.transform_df(df)

    # Train primarily on normal data (unsupervised), evaluate on held-out mix
    train_df, test_df = train_test_split(
        df, test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_seed"], stratify=df["is_anomaly"],
    )
    # Fit detector mostly on normal logs for classic unsupervised setting
    normal_train = train_df[train_df["is_anomaly"] == 0]
    fit_df = normal_train if len(normal_train) > 50 else train_df

    emb_cfg = cfg["embedding"]
    embedder = EmbeddingModel(
        model_name=emb_cfg["model_name"],
        device=emb_cfg.get("device"),
        normalize=emb_cfg.get("normalize", True),
    )
    print("Encoding...")
    X_fit = embedder.encode(fit_df["cleaned_text"].tolist(), batch_size=emb_cfg.get("batch_size", 64))
    X_test = embedder.encode(test_df["cleaned_text"].tolist(), batch_size=emb_cfg.get("batch_size", 64))

    det_cfg = cfg["detector"]
    detector = AnomalyDetectorModel(
        method=det_cfg["method"],
        contamination=det_cfg.get("contamination", 0.12),
        n_estimators=det_cfg.get("n_estimators", 200),
        n_neighbors=det_cfg.get("n_neighbors", 20),
        random_state=det_cfg.get("random_state", 42),
    )
    print(f"Fitting {det_cfg['method']}...")
    detector.fit(X_fit)

    y_true = test_df["is_anomaly"].values
    y_pred = detector.predict(X_test)
    scores = detector.decision_function(X_test)

    print("\n=== Test Performance ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, target_names=["normal", "anomaly"], digits=3))
    try:
        auc = roc_auc_score(y_true, scores)
        ap = average_precision_score(y_true, scores)
        print(f"ROC-AUC: {auc:.4f} | Average Precision: {ap:.4f}")
    except Exception as e:
        print("AUC computation issue:", e)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"])
    plt.title("Confusion Matrix - Log Anomaly Detection")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = Path(cfg["paths"]["artifacts_dir"]) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=120)
    print(f"Saved {cm_path}")
    plt.close()

    detector.save(cfg["paths"]["detector"])
    test_df = test_df.copy()
    test_df["pred_anomaly"] = y_pred
    test_df["anomaly_score"] = scores
    test_df[["log_id", "label", "is_anomaly", "pred_anomaly", "anomaly_score", "message"]].to_csv(
        cfg["paths"]["metadata"], index=False
    )
    print("\n✅ Training complete. Artifacts in", cfg["paths"]["artifacts_dir"])


if __name__ == "__main__":
    main()
