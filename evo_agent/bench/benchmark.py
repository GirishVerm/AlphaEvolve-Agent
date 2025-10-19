#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
import statistics
import signal
from pathlib import Path


def run(cmd, **popen_kwargs):
    return subprocess.Popen(cmd, **popen_kwargs)


def measure_rss(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).decode().strip()
        return int(out) if out else 0
    except Exception:
        return 0


def compute_metrics(alerts_path: Path):
    latencies = []
    y_true = []
    y_pred = []
    with alerts_path.open() as f:
        for line in f:
            try:
                a = json.loads(line)
            except Exception:
                continue
            try:
                tsw = float(a.get("snippet", "").split()[0])
                l = a["timestamp"] - tsw
                if 0 <= l < 10:
                    latencies.append(l)
            except Exception:
                pass
            z = a.get("metrics", {}).get("z_score", 0.0)
            y_true.append(1 if z >= 2.0 else 0)
            y_pred.append(1 if a.get("metrics", {}).get("anomaly") else 0)
    median_latency_ms = statistics.median(latencies) * 1000 if latencies else float("inf")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return median_latency_ms, precision, len(latencies)


def main():
    base = Path("/Users/girishverma/Documents/palantir-agent/evo_agent")
    cfg = base / "examples" / "logcfg.json"
    alerts = Path("/tmp/alerts.jsonl")
    log = Path("/tmp/app.log")

    # Clean prev
    for p in [alerts, log]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # Start writer
    writer = run([
        sys.executable,
        str(base / "scripts" / "synthetic_writer.py"),
        "--log", str(log),
        "--seconds", "30",
        "--rate", "5000",
    ])

    # Start analyzer
    analyzer = run([
        sys.executable,
        str(base / "code" / "gen_6.py"),
        "-c", str(cfg)
    ], stdout=alerts.open("w"), stderr=subprocess.DEVNULL)

    start = time.time()
    rss_samples = []
    try:
        while time.time() - start < 32:
            time.sleep(1)
            rss = measure_rss(analyzer.pid)
            if rss:
                rss_samples.append(rss)
    finally:
        for proc in [writer, analyzer]:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except Exception:
                pass

    median_latency_ms, precision, n = compute_metrics(alerts)
    max_rss_mb = (max(rss_samples) / 1024) if rss_samples else 0

    # Success criteria checks
    ok_latency = median_latency_ms < 100
    ok_rss = max_rss_mb < 300
    ok_precision = precision >= 0.9

    result = {
        "median_latency_ms": round(median_latency_ms, 1),
        "precision": round(precision, 3),
        "max_rss_mb": round(max_rss_mb, 1),
        "latency_ok": ok_latency,
        "rss_ok": ok_rss,
        "precision_ok": ok_precision,
        "samples": n,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


