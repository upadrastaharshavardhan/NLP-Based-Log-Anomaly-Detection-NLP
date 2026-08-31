"""Gradio demo for Log Anomaly Detection."""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gradio as gr
from src.pipeline.detector import LogAnomalyDetector
from src.utils.helpers import load_config


def build_demo(artifacts_dir: str = "artifacts", config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    detector = LogAnomalyDetector.load(artifacts_dir, config_path)

    def score_fn(log_text: str):
        if not log_text or not log_text.strip():
            return "Please paste a log line."
        r = detector.predict(log_text)
        status = "ANOMALY" if r["is_anomaly"] else "NORMAL"
        color = "red" if r["is_anomaly"] else "green"
        return (
            f"### Decision: **{status}**\n\n"
            f"**Anomaly score:** {r['anomaly_score']:.3f} (higher = more anomalous)\n\n"
            f"**Raw score:** {r['raw_score']:.4f}\n\n"
            f"**Cleaned input:** `{r['cleaned_input']}`"
        )

    demo = gr.Interface(
        fn=score_fn,
        inputs=gr.Textbox(lines=5, label="Log message", placeholder="Paste a log line..."),
        outputs=gr.Markdown(),
        title=cfg.get("gradio", {}).get("title", "NLP Log Anomaly Detector"),
        description=cfg.get("gradio", {}).get("description", ""),
        examples=[
            ["INFO Request processed successfully status=200 latency=45ms"],
            ["ERROR OutOfMemoryError: Java heap space at ReportService.generate"],
            ["ERROR NullPointerException at AuthService.validateToken line 87"],
            ["DEBUG Cache hit ratio updated to 0.92"],
            ["FATAL Uncaught exception in main event loop - process crashing"],
        ],
        allow_flagging="never",
    )
    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(share=False, server_name="0.0.0.0")
