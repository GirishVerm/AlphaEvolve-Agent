#!/bin/sh
set -e

CFG=${1:-/Users/girishverma/Documents/palantir-agent/evo_agent/examples/logcfg.json}

python3 /Users/girishverma/Documents/palantir-agent/evo_agent/scripts/synthetic_writer.py --log /tmp/app.log --seconds 30 --rate 5000 &
WRITER_PID=$!

python3 /Users/girishverma/Documents/palantir-agent/evo_agent/code/gen_6.py -c "$CFG" | tee /tmp/alerts.jsonl

kill $WRITER_PID 2>/dev/null || true


