"""End-to-end Log Anomaly Detection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from src.data.preprocessing import LogPreprocessor
from src.models.embeddings import EmbeddingModel
from src.models.anomaly import AnomalyDetectorModel


class LogAnomalyDetector:
    def __init__(
        self,
        embedder: EmbeddingModel,
        detector: AnomalyDetectorModel,
        preprocessor: LogPreprocessor,
    ):
        self.embedder = embedder
        self.detector = detector
        self.preprocessor = preprocessor

    def predict(self, log_text: str) -> Dict[str, Any]:
        cleaned = self.preprocessor.clean(log_text)
        emb = self.embedder.encode([cleaned], show_progress=False)
        result = self.detector.predict_with_score(emb)[0]
        result["cleaned_input"] = cleaned[:300]
        return result

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        cleaned = self.preprocessor.transform(texts)
        embs = self.embedder.encode(cleaned, show_progress=True)
        return self.detector.predict_with_score(embs)

    @classmethod
    def load(
        cls,
        artifacts_dir: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
    ) -> "LogAnomalyDetector":
        artifacts_dir = Path(artifacts_dir)
        if config_path is None:
            config_path = Path("config/config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        emb_cfg = cfg.get("embedding", {})
        pre_cfg = cfg.get("preprocessing", {})

        embedder = EmbeddingModel(
            model_name=emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            device=emb_cfg.get("device"),
            normalize=emb_cfg.get("normalize", True),
        )
        detector = AnomalyDetectorModel.load(artifacts_dir / "detector.joblib")
        preprocessor = LogPreprocessor(
            max_text_length=pre_cfg.get("max_text_length", 1000),
            remove_timestamps=pre_cfg.get("remove_timestamps", True),
            remove_ids=pre_cfg.get("remove_ids", True),
            remove_hex=pre_cfg.get("remove_hex", True),
        )
        return cls(embedder, detector, preprocessor)
