@echo off
REM One-click launcher (Windows): creates a local venv on first run, keeps packages current, opens the app.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
)
.venv\Scripts\pip install -q -r requirements.txt
.venv\Scripts\streamlit run app.py
