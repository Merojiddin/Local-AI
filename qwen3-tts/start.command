#!/bin/bash
# ============================================================================
# Local Chinese TTS — start the app. Double-click this file in Finder.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

hold_open() {
    echo ""
    echo "The app has stopped. If this was unexpected, read the message above."
    read -r -p "Press Enter to close this window."
}

echo "============================================================"
echo " Local Chinese TTS — starting"
echo " Folder: $SCRIPT_DIR"
echo "============================================================"

# --- Virtual environment -----------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "❌ .venv not found. Please run install.command first."
    hold_open
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "❌ Could not activate .venv."; hold_open; exit 1; }

# --- Verify dependencies -----------------------------------------------------
python -c "import gradio, mlx_audio" 2>/dev/null || {
    echo "❌ Dependencies missing. Please run install.command again."
    hold_open
    exit 1
}
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "⚠️  FFmpeg not found. Install it with:  brew install ffmpeg"
fi

echo "Opening http://127.0.0.1:7860 in your browser…"
echo "To stop the app: close this window, or press Control + C here."
echo ""

# The app opens the browser automatically (inbrowser=True).
python app.py
STATUS=$?

if [ "$STATUS" -ne 0 ]; then
    echo ""
    echo "❌ The app exited with an error (code $STATUS)."
fi
hold_open
