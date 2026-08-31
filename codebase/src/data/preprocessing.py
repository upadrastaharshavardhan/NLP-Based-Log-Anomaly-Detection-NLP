"""Log preprocessing for anomaly detection."""

from __future__ import annotations

import re
from typing import List
import pandas as pd


class LogPreprocessor:
    def __init__(
        self,
        max_text_length: int = 1000,
        remove_timestamps: bool = True,
        remove_ids: bool = True,
        remove_hex: bool = True,
    ):
        self.max_text_length = max_text_length
        self.remove_timestamps = remove_timestamps
        self.remove_ids = remove_ids
        self.remove_hex = remove_hex

    def clean(self, text: str) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""
        text = str(text)
        if self.remove_ids:
            text = re.sub(r"(?i)(request[_-]?id|correlation[_-]?id|trace[_-]?id)\s*[=:]\s*\S+", "", text)
            text = re.sub(r"\bL-\d+\b", "", text)
        if self.remove_hex:
            text = re.sub(r"\b[0-9a-f]{8,}\b", "", text, flags=re.IGNORECASE)
        if self.remove_timestamps:
            text = re.sub(
                r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?",
                "", text,
            )
        text = re.sub(r"\s+", " ", text).strip()
        return text[: self.max_text_length]

    def transform(self, texts: List[str]) -> List[str]:
        return [self.clean(t) for t in texts]

    def transform_df(self, df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
        df = df.copy()
        df["cleaned_text"] = self.transform(df[text_col].tolist())
        return df
