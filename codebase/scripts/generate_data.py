#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.generator import generate_log_dataset
from src.utils.helpers import load_config, ensure_dirs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--anomaly-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    n = args.n_samples or cfg["data"]["n_samples"]
    ratio = args.anomaly_ratio or cfg["data"]["anomaly_ratio"]
    seed = args.seed or cfg["data"]["random_seed"]
    output = args.output or cfg["paths"]["raw_data"]
    ensure_dirs(Path(output).parent)
    print(f"Generating {n} logs (anomaly_ratio={ratio}, seed={seed})...")
    df = generate_log_dataset(n_samples=n, anomaly_ratio=ratio, seed=seed)
    df.to_csv(output, index=False)
    print(f"Saved → {output}")
    print(df["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
