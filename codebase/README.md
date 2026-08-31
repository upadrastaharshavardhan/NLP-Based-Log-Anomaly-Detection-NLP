# NLP-Based Log Anomaly Detection

**Project 4** – Detect unusual application/system behavior from logs using NLP embeddings and unsupervised anomaly detection.

## What it does

Given streams of application/system logs, the system:

1. Embeds log messages into a semantic vector space
2. Learns a model of "normal" behavior (from historical normal logs)
3. Scores new logs for anomaly degree
4. Flags outliers with confidence / anomaly score
5. Optionally groups anomalies into patterns

## Key Features

- Sentence-transformer embeddings of log text
- Multiple detectors: Isolation Forest, Local Outlier Factor, One-Class SVM
- Realistic synthetic generator (normal + injected anomalies)
- Threshold tuning and evaluation (Precision, Recall, F1, ROC-AUC)
- Gradio demo for scoring individual logs
- Fully configurable, Colab-ready modular structure

## Quick Start (Colab)

```bash
!pip install -r requirements.txt
!python scripts/generate_data.py --n-samples 6000
!python scripts/train.py
!python -m src.api.gradio_app
```

## Example

```python
from src.pipeline.detector import LogAnomalyDetector

detector = LogAnomalyDetector.load("artifacts")
result = detector.predict("OutOfMemoryError: Java heap space at ReportService.generate")
print(result)
# {"is_anomaly": True, "anomaly_score": 0.87, "decision": "anomaly", ...}
```

## License

MIT
