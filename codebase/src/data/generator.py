"""Synthetic log generator: normal operational logs + injected anomalies."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

NORMAL_TEMPLATES = [
    "INFO Application started successfully on port 8080",
    "INFO Health check passed for service endpoint",
    "DEBUG Cache hit ratio updated to 0.92",
    "INFO User session created successfully",
    "INFO Scheduled job completed in 1.2 seconds",
    "DEBUG Metrics scraped by monitoring agent",
    "INFO Database connection pool initialized with 20 connections",
    "INFO Request processed successfully status=200 latency=45ms",
    "INFO Configuration loaded from application.yml",
    "DEBUG Heartbeat sent to service registry",
    "INFO Graceful shutdown initiated",
    "INFO Backup job completed successfully",
    "DEBUG Feature flag evaluated: new_ui=false",
    "INFO Kafka consumer committed offset",
    "INFO Cache refreshed for key product_catalog",
]

ANOMALY_TEMPLATES = [
    "ERROR OutOfMemoryError: Java heap space at ReportService.generate",
    "ERROR NullPointerException at AuthService.validateToken line 87",
    "ERROR ConnectException: Connection refused to payment-gateway:8080",
    "ERROR Deadlock found when trying to get lock; try restarting transaction",
    "FATAL Segmentation fault in native library crypto.so",
    "ERROR SSLHandshakeException: Remote host terminated the handshake",
    "ERROR CannotGetJdbcConnectionException: Connection pool exhausted",
    "CRITICAL Disk full on /var/log - writes failing",
    "ERROR Kubernetes pod OOMKilled - memory limit exceeded",
    "ERROR JwtException: Token has expired and signature invalid",
    "ERROR FeignException status 503 from inventory-service",
    "ERROR ConcurrentModificationException in shopping cart update",
    "FATAL Uncaught exception in main event loop - process crashing",
    "ERROR SQLException: Duplicate entry for key PRIMARY",
    "ERROR SocketTimeoutException: Read timed out after 30000 ms",
]

SERVICES = [
    "auth-service", "order-service", "payment-service", "inventory-service",
    "user-service", "api-gateway", "report-service", "notification-service",
]


def generate_log_dataset(
    n_samples: int = 6000,
    anomaly_ratio: float = 0.12,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    records = []
    idx = 0

    for _ in range(n_normal):
        msg = random.choice(NORMAL_TEMPLATES)
        service = random.choice(SERVICES)
        if random.random() < 0.2:
            msg = f"{msg} | requestId={random.randint(10000, 99999)}"
        full = f"Service: {service} | {msg}"
        records.append({
            "log_id": f"L-{idx+1:06d}",
            "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 10000)),
            "service": service,
            "message": msg,
            "full_text": full,
            "is_anomaly": 0,
            "label": "normal",
        })
        idx += 1

    for _ in range(n_anomaly):
        msg = random.choice(ANOMALY_TEMPLATES)
        service = random.choice(SERVICES)
        if random.random() < 0.3:
            msg = f"{msg} | requestId={random.randint(10000, 99999)}"
        full = f"Service: {service} | {msg}"
        records.append({
            "log_id": f"L-{idx+1:06d}",
            "timestamp": datetime.now() - timedelta(minutes=random.randint(1, 10000)),
            "service": service,
            "message": msg,
            "full_text": full,
            "is_anomaly": 1,
            "label": "anomaly",
        })
        idx += 1

    random.shuffle(records)
    df = pd.DataFrame(records)
    df["log_id"] = [f"L-{i+1:06d}" for i in range(len(df))]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_log_dataset(500)
    print(df["label"].value_counts())
    print(df.head(2))
