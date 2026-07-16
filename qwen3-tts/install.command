#!/bin/bash
# ============================================================================
# Local Chinese TTS — one-time installer for macOS (Apple Silicon)
# Double-click this file in Finder to run it.
# ============================================================================

# Always work inside this script's own folder (handles spaces in the path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Keep the Terminal window open if anything fails.
fail() {
    echo ""
    echo "❌ INSTALL FAILED: $1"
    echo "Please read the message above, fix it, and run install.command again."
    echo ""
    read -r -p "Press Enter to close this window."
    exit 1
}

echo "============================================================"
echo " Local Chinese TTS — Installer"
echo " Folder: $SCRIPT_DIR"
echo "============================================================"

# --- Apple Silicon check -----------------------------------------------------
if [ "$(uname -m)" != "arm64" ]; then
    fail "This app requires an Apple Silicon Mac (M1/M2/M3/M4). Detected: $(uname -m)."
fi

# --- Homebrew ----------------------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
    echo ""
    echo "Homebrew is not installed. Homebrew needs your password and agreement,"
    echo "so it is safest to install it yourself. Please run this line in Terminal:"
    echo ""
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "Then run install.command again."
    fail "Homebrew is required."
fi
# Make sure brew is on PATH (Apple Silicon default location).
eval "$(/opt/homebrew/bin/brew shellenv 2>/dev/null)" || true

# --- Python 3.12 -------------------------------------------------------------
if ! command -v python3.12 >/dev/null 2>&1; then
    echo "Installing Python 3.12 via Homebrew…"
    brew install python@3.12 || fail "Could not install Python 3.12."
fi
PYTHON_BIN="$(command -v python3.12)"
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version))"

# --- FFmpeg ------------------------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing FFmpeg via Homebrew…"
    brew install ffmpeg || fail "Could not install FFmpeg."
fi
echo "Using FFmpeg: $(command -v ffmpeg)"

# --- Virtual environment -----------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment (.venv)…"
    "$PYTHON_BIN" -m venv .venv || fail "Could not create the virtual environment."
fi
# shellcheck disable=SC1091
source .venv/bin/activate || fail "Could not activate the virtual environment."

# --- Python packages ---------------------------------------------------------
echo "Upgrading pip…"
python -m pip install --upgrade pip || fail "Could not upgrade pip."

echo "Installing Python requirements (this can take a few minutes)…"
python -m pip install -r requirements.txt || fail "Could not install requirements."

# --- Folders -----------------------------------------------------------------
mkdir -p outputs cache
touch outputs/.gitkeep

# --- Download both models ----------------------------------------------------
echo ""
echo "Downloading both Qwen3-TTS models (several GB, can take a while)…"
python app.py --download || fail "Model download failed. Check your internet connection."

echo ""
echo "============================================================"
echo " ✅ INSTALL COMPLETE"
echo " Start the app by double-clicking:  start.command"
echo "============================================================"
echo ""
read -r -p "Press Enter to close this window."
