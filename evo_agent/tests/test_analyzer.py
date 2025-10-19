import asyncio
import time
import types
import re

import importlib.util, sys, pathlib

GEN_PATH = pathlib.Path('/Users/girishverma/Documents/palantir-agent/evo_agent/code/gen_6.py')
spec = importlib.util.spec_from_file_location("analyzer_mod", str(GEN_PATH))
analyzer_mod = importlib.util.module_from_spec(spec)
sys.modules["analyzer_mod"] = analyzer_mod
assert spec.loader is not None
spec.loader.exec_module(analyzer_mod)  # type: ignore


def build_config(tmp_path):
    cfg = {
        "log_files": [str(tmp_path / "app.log")],
        "patterns": [
            {"id": "err", "regex": "ERROR", "severity": "ERROR", "cooldown": 0.2, "ignore": ["IGNORE_ME"], "anomaly_threshold": 1.0},
        ],
    }
    p = tmp_path / "cfg.json"
    p.write_text(__import__("json").dumps(cfg))
    return p


def test_ignore_and_regex(tmp_path):
    cfg_path = build_config(tmp_path)
    cfg = analyzer_mod.Config(cfg_path, {})
    lm = analyzer_mod.LogAnalyzer(cfg)

    # A line matching ERROR but also matching ignore should not alert
    line = "ERROR IGNORE_ME something"
    pat = cfg.patterns[0]
    assert any(ign.search(line) for ign in pat["ignore"]) is True
    assert pat["regex"].search(line)


def test_rate_limiter():
    rl = analyzer_mod.RateLimiter()
    now = time.time()
    assert rl.allow("/tmp/app.log", "err", 0.5) is True
    assert rl.allow("/tmp/app.log", "err", 0.5) is False
    time.sleep(0.6)
    assert rl.allow("/tmp/app.log", "err", 0.5) is True


def test_anomaly_detector():
    det = analyzer_mod.AnomalyDetector()
    pid = "err"
    # record baseline sparse events
    for _ in range(5):
        det.record(pid, ts=time.time() - 240)  # within 5m window
    z0 = det.z_score(pid)
    # simulate a spike (many events in 1m)
    for _ in range(30):
        det.record(pid, ts=time.time() - 5)
    z1 = det.z_score(pid)
    assert z1 >= z0


