# Research Package - Project 4
## NLP-Based Log Anomaly Detection

Complete research paper, documentation, results, and full codebase.

## Key Metrics

| Metric            | Value    |
|-------------------|----------|
| Accuracy          | 96.20%   |
| F1 (anomaly)      | 0.912    |
| ROC-AUC           | **0.968** |
| Average Precision | **0.941** |

## Contents

- paper/ (PDF + Markdown research paper)
- docs/ (methodology, results, data analysis, discussion)
- results/ (CSV metrics)
- codebase/ (full advanced Project 4 source)

## Reproduce

```bash
cd codebase
pip install -r requirements.txt
python scripts/generate_data.py --n-samples 6000 --seed 42
python scripts/train.py
python scripts/evaluate.py
```
