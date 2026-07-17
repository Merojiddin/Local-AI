#!/bin/bash
# ============================================================================
# Chang Local AI Toolbox — one-click launcher.
# Starts the server, opens the app in its own window, and shuts everything
# down again when that window is closed.
#
# The Desktop shortcut ("Chang AI Toolbox.command") runs this script.
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

URL="http://127.0.0.1:7860"
PORT=7860
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# Dedicated Chrome profile so the app window is its own process we can wait on.
PROFILE_DIR="$HOME/Library/Application Support/ChangAIToolboxWindow"

hold_open() {
    echo ""
    read -r -p "Press Enter to close this window."
}

close_terminal_window() {
    # Close the Terminal window running this script, matched by its tty so we
    # never close a window the user is using for something else.
    local tty_dev
    tty_dev=$(tty 2>/dev/null) || return 0
    [[ "$tty_dev" == /dev/* ]] || return 0
    osascript >/dev/null 2>&1 <<EOF &
delay 0.5
tell application "Terminal"
    repeat with w in windows
        repeat with t in tabs of w
            try
                if tty of t is "$tty_dev" then close w saving no
            end try
        end repeat
    end repeat
end tell
EOF
}

echo "============================================================"
echo " Chang Local AI Toolbox"
echo "============================================================"

# --- Virtual environment -----------------------------------------------------
if [ ! -d ".venv" ]; then
    echo "❌ .venv not found. Please run install.command first."
    hold_open; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate || { echo "❌ Could not activate .venv."; hold_open; exit 1; }

# --- Start the server (unless one is already running) ------------------------
SERVER_PID=""
if curl -s -o /dev/null --max-time 1 "$URL"; then
    echo "App is already running — opening the window."
else
    echo "Starting the app…"
    QWEN3_TTS_NO_BROWSER=1 python app.py &
    SERVER_PID=$!
    ready=0
    for _ in $(seq 1 120); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo ""
            echo "❌ The app failed to start (see the messages above)."
            hold_open; exit 1
        fi
        if curl -s -o /dev/null --max-time 1 "$URL"; then ready=1; break; fi
        sleep 0.5
    done
    if [ "$ready" -ne 1 ]; then
        echo "❌ The app did not become ready within 60 seconds."
        kill "$SERVER_PID" 2>/dev/null
        hold_open; exit 1
    fi
fi

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null
    else
        # We attached to an already-running server: stop whatever owns the port.
        lsof -ti tcp:$PORT 2>/dev/null | xargs kill 2>/dev/null
    fi
}
trap stop_server EXIT

# --- Open the app window and wait until it is closed -------------------------
if [ -x "$CHROME" ]; then
    echo "App window is open. Close it (or press Control+C here) to quit."
    "$CHROME" --app="$URL" \
        --user-data-dir="$PROFILE_DIR" \
        --no-first-run --no-default-browser-check \
        >/dev/null 2>&1
else
    echo "Google Chrome not found — opening in your default browser."
    echo "Close this Terminal window (or press Control+C) to stop the app."
    open "$URL"
    wait "$SERVER_PID"
fi

echo "Shutting down…"
stop_server
trap - EXIT
close_terminal_window
exit 0
