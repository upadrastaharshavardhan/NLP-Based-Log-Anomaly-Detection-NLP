"""Sentence embedding wrapper."""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=self.normalize,
        )

    def save_embeddings(self, embeddings: np.ndarray, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)
