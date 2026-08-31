# Methodology - Project 4

## Pipeline
Log text -> clean IDs/timestamps/hex -> sentence embedding (MiniLM) -> Isolation Forest / LOF / OCSVM -> anomaly score + binary decision.

## Training protocol
Detector fitted primarily on normal logs (unsupervised). Evaluation on held-out mix of normal and anomalous logs.

## Scoring
Sklearn decision_function inverted so higher score = more anomalous. Optional quantile normalization to [0,1].
