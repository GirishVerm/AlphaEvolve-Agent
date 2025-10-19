#!/usr/bin/env python3
import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
from collections import deque, defaultdict
from pathlib import Path

# JSON Schema for validation (could be used with jsonschema library)
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "log_files": {"type": "array", "items": {"type": "string"}},
        "patterns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "regex": {"type": "string"},
                    "severity": {"type": "string", "enum": ["DEBUG","INFO","WARN","ERROR","CRITICAL"]},
                    "cooldown": {"type": "number", "minimum": 0},
                    "ignore": {"type": "array", "items": {"type": "string"}},
                    "anomaly_threshold": {"type": "number", "minimum": 0}
                },
                "required": ["id", "regex", "severity", "cooldown"]
            }
        }
    },
    "required": ["log_files", "patterns"]
}


class Config:
    def __init__(self, path, overrides=None):
        self.path = Path(path)
        self.overrides = overrides or {}
        self._load()

    def _load(self):
        cfg = json.loads(self.path.read_text())
        # Here you could validate against CONFIG_SCHEMA
        self.log_files = cfg["log_files"]
        self.patterns = []
        for p in cfg["patterns"]:
            # apply overrides
            if p["id"] in self.overrides:
                p.update(self.overrides[p["id"]])
            regex = re.compile(p["regex"])
            ignore = [re.compile(i) for i in p.get("ignore", [])]
            self.patterns.append({
                "id": p["id"],
                "regex": regex,
                "severity": p["severity"],
                "cooldown": p["cooldown"],
                "ignore": ignore,
                "anomaly_threshold": p.get("anomaly_threshold", 3.0)
            })


class TailTask:
    """
    Tails a single file, yields (line, file_path, lineno)
    """
    def __init__(self, path, queue, loop):
        self.path = Path(path)
        self.loop = loop
        self.queue = queue
        self._stopping = False
        self._inode = None
        self._position = 0
        self._lineno = 0

    async def watch(self):
        while not self._stopping:
            try:
                stat = self.path.stat()
                inode = stat.st_ino
                size = stat.st_size
                if self._inode != inode:
                    # rotated or first open
                    self._inode = inode
                    self._position = 0
                elif size < self._position:
                    # truncated
                    self._position = 0

                with self.path.open('r') as f:
                    f.seek(self._position)
                    for line in f:
                        self._lineno += 1
                        self._position = f.tell()
                        await self.queue.put((line.rstrip('\n'), str(self.path), self._lineno))
                await asyncio.sleep(0.1)
            except FileNotFoundError:
                await asyncio.sleep(1.0)

    def stop(self):
        self._stopping = True


class AnomalyDetector:
    """
    Maintains rolling windows for each pattern to compute simple z-score
    based on 1m and 5m rates.
    """
    def __init__(self):
        # pattern_id -> deque of timestamps
        self.timestamps = defaultdict(lambda: deque())

    def record(self, pat_id, ts=None):
        now = ts or time.time()
        dq = self.timestamps[pat_id]
        dq.append(now)
        # prune older than 5m
        cutoff = now - 300
        while dq and dq[0] < cutoff:
            dq.popleft()

    def z_score(self, pat_id):
        now = time.time()
        dq = self.timestamps[pat_id]
        cnt_5m = len(dq)
        cnt_1m = sum(1 for t in dq if t >= now - 60)
        mean5 = cnt_5m / 5.0  # rate per minute
        var5 = sum(((1 - mean5)**2 for t in dq)) / 5.0 if cnt_5m else 0.0
        std5 = var5**0.5
        if std5 == 0:
            return 0.0
        return (cnt_1m - mean5) / std5


class RateLimiter:
    """
    Ensures cooldown between alerts per (file, pattern_id)
    """
    def __init__(self):
        self.last_alert = {}  # (file, pat_id) -> timestamp

    def allow(self, file, pat_id, cooldown):
        key = (file, pat_id)
        now = time.time()
        last = self.last_alert.get(key, 0)
        if now - last >= cooldown:
            self.last_alert[key] = now
            return True
        return False


class LogAnalyzer:
    def __init__(self, config):
        self.config = config
        self.queue = asyncio.Queue(maxsize=10000)
        self.alert_queue = asyncio.Queue()
        self.tails = []
        self.detector = AnomalyDetector()
        self.ratelimiter = RateLimiter()
        self._stop = asyncio.Event()

    async def start(self):
        loop = asyncio.get_running_loop()
        # start tailers
        for path in self.config.log_files:
            t = TailTask(path, self.queue, loop)
            self.tails.append(t)
            loop.create_task(t.watch())

        # start worker
        workers = [asyncio.create_task(self._worker())
                   for _ in range(min(4, len(self.config.log_files)))]
        # start alert printer
        printer = asyncio.create_task(self._printer())
        await self._stop.wait()
        # shutdown everything
        for t in self.tails:
            t.stop()
        for w in workers:
            w.cancel()
        printer.cancel()

    async def _worker(self):
        while True:
            line, file, lineno = await self.queue.get()
            for pat in self.config.patterns:
                if any(ign.search(line) for ign in pat["ignore"]):
                    continue
                m = pat["regex"].search(line)
                if not m:
                    continue
                pat_id = pat["id"]
                # record for anomaly
                self.detector.record(pat_id)
                # compute anomaly
                z = self.detector.z_score(pat_id)
                is_anomaly = z >= pat["anomaly_threshold"]
                # rate limit
                if not self.ratelimiter.allow(file, pat_id, pat["cooldown"]):
                    continue
                alert = {
                    "timestamp": time.time(),
                    "file": file,
                    "line_no": lineno,
                    "pattern_id": pat_id,
                    "severity": pat["severity"],
                    "snippet": line,
                    "metrics": {
                        "z_score": z,
                        "anomaly": is_anomaly
                    }
                }
                await self.alert_queue.put(alert)

    async def _printer(self):
        while True:
            alert = await self.alert_queue.get()
            sys.stdout.write(json.dumps(alert) + "\n")
            sys.stdout.flush()

    def stop(self):
        self._stop.set()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Log Analyzer")
    parser.add_argument("-c", "--config", required=True, help="Path to config JSON")
    parser.add_argument("-o", "--override", action="append",
                        help="Override pattern fields (id:key=value)")
    return parser.parse_args()


def build_overrides(ov_list):
    o = {}
    if not ov_list:
        return o
    for item in ov_list:
        # format id:key=value
        pid, rest = item.split(":", 1)
        key, val = rest.split("=", 1)
        if pid not in o:
            o[pid] = {}
        # try to cast
        if val.isdigit():
            val = int(val)
        else:
            try:
                val = float(val)
            except:
                pass
        o[pid][key] = val
    return o


def main():
    args = parse_args()
    overrides = build_overrides(args.override)
    cfg = Config(args.config, overrides)
    analyzer = LogAnalyzer(cfg)
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, analyzer.stop)

    try:
        loop.run_until_complete(analyzer.start())
    finally:
        loop.close()


if __name__ == "__main__":
    main()