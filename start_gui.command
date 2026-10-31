#!/bin/bash
# Double-click (macOS) or run ./start_gui.command to launch the GUI in your browser.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: creating virtual environment and installing packages..."
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip -q
  ./.venv/bin/pip install -r requirements.txt -q
fi
# stop any earlier copy of the app so the latest code loads
pkill -f "streamlit run app.py" 2>/dev/null; sleep 1
./.venv/bin/streamlit run app.py
