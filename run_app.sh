#!/usr/bin/env bash
# One-click launcher (macOS / Linux): creates a local venv on first run, keeps packages current, opens the app.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -q -r requirements.txt
.venv/bin/streamlit run app.py
