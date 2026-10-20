#!/usr/bin/env bash
# One-click launcher (macOS / Linux): creates a local venv on first run, then opens the app.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/streamlit run app.py
