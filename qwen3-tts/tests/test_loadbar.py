"""Tests for the window-top load bar + finish chime — run with:
    .venv/bin/python tests/test_loadbar.py

The bar and the chime live in branding.LIBRARY_HEAD as browser JS, so the real
script is pulled straight out of that string and driven against a stub DOM
(loadbar_harness.js) with the assertions in loadbar_checks.js. macOS ships a
JavaScript engine as `osascript -l JavaScript`, so this needs no extra tooling.

What it guards: the bar only shows while an event is actually live, a fraction
turns it determinate, a brief gap between two chained events does NOT count as
"finished", and the chime plays once per real run — success vs error tone,
silent when muted or when the run was too short to be worth a sound.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from modules import branding  # noqa: E402


def main() -> int:
    script = branding.LIBRARY_HEAD.split("<script>")[1].split("</script>")[0]
    bundle = "\n".join([
        (HERE / "loadbar_harness.js").read_text(),
        script,
        (HERE / "loadbar_checks.js").read_text(),
    ])
    tmp = HERE / "_loadbar_run.js"
    tmp.write_text(bundle)
    try:
        proc = subprocess.run(
            ["osascript", "-l", "JavaScript", str(tmp)],
            capture_output=True, text=True,
        )
    finally:
        tmp.unlink(missing_ok=True)

    print(proc.stdout.strip() or proc.stderr.strip())
    if proc.returncode != 0 or "FAILED" in proc.stdout:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
