#!/usr/bin/env bash
# Assemble the labeled dataset and write final counts. Run from anywhere.
set -euo pipefail
cd /workspace/roleplay-probe/src
../.venv/bin/python assemble.py
../.venv/bin/python status.py
../.venv/bin/python report_counts.py
