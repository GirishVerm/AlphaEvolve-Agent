#!/usr/bin/env python3
import os
import time
import random
import shutil
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/tmp/app.log")
    ap.add_argument("--seconds", type=int, default=60)
    ap.add_argument("--rotate_every", type=int, default=15)
    ap.add_argument("--rate", type=int, default=5000, help="approx lines/sec")
    args = ap.parse_args()

    log = args.log
    os.makedirs(os.path.dirname(log), exist_ok=True)
    open(log, "w").close()

    start = time.time()
    while time.time() - start < args.seconds:
        t = time.time()
        # baseline noise
        for _ in range(max(1, args.rate // 10)):
            with open(log, "a") as f:
                f.write(f"{t} INFO status=ok\n")
        # WARN lines
        for _ in range(max(1, args.rate // 50)):
            with open(log, "a") as f:
                f.write(f"{t} WARN slow op id={random.randint(1,999)}\n")
        # spike burst every ~10s
        if int(t - start) % 10 == 0:
            for _ in range(max(1, args.rate // 5)):
                with open(log, "a") as f:
                    f.write(f"{t} ERROR code=42 critical path\n")
        # ignored line
        with open(log, "a") as f:
            f.write(f"{t} ERROR IGNORE_ME noisy\n")
        # rotate
        if int(t - start) % max(1, args.rotate_every) == 0:
            if os.path.exists(log):
                shutil.move(log, f"{log}-{int(t)}")
                open(log, "w").close()
        time.sleep(0.2)


if __name__ == "__main__":
    main()


