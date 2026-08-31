from __future__ import annotations
import random
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

def load_config(path: str | Path = "config/config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)

def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
