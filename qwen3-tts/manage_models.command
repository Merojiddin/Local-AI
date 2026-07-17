#!/bin/bash
# ============================================================================
# Chang Local AI Toolbox — manage models (install / remove / verify / storage)
# Double-click this file in Finder to run it.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

hold_open() {
    echo ""
    read -r -p "Press Enter to close this window."
}

if [ ! -d ".venv" ]; then
    echo "❌ .venv not found. Please run install.command first."
    hold_open
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "❌ Could not activate .venv."; hold_open; exit 1; }

python manage_models.py
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo ""
    echo "❌ The model manager exited with an error (code $STATUS)."
fi
hold_open
