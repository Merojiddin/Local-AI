# Local Chinese Text-to-Speech (Qwen3-TTS)

Turn Chinese text into natural Mandarin speech — **entirely on your Mac**.
No cloud, no account, no internet needed after installation.

- Models: Qwen3-TTS 0.6B (fast) and 1.7B (higher quality), both 8-bit for Apple Silicon
- Voices: Vivian, Serena, Uncle_Fu, Dylan (Beijing dialect), Eric (Sichuan dialect)
- Works on Apple Silicon Macs (M1/M2/M3/M4), tested on a MacBook Air M2 with 16 GB RAM

---

## 1. What you need

- An Apple Silicon Mac (M1 or newer)
- About **8 GB of free disk space** (≈5 GB for the two models + apps and libraries)
- Internet connection **for the one-time install only**

## 2. How to open Terminal

1. Press `Command + Space` to open Spotlight.
2. Type `Terminal` and press `Return`.

You only need Terminal for the first-time setup steps below.

## 3. How to save the project

Put this whole folder (the one containing `app.py`, `install.command`,
`start.command`) anywhere you like, for example:

```
/Users/yourname/Documents/Local Ai/qwen3-tts
```

Spaces in the folder path are fine.

## 4. First-time installation

1. Open Terminal (see above).
2. Make the two scripts executable — copy-paste this line and press `Return`
   (drag the project folder into the Terminal window to auto-fill its path):

   ```bash
   chmod +x "/Users/yourname/Documents/Local Ai/qwen3-tts/"*.command
   ```

3. In **Finder**, open the project folder and **double-click `install.command`**.
   - If macOS says the file is from an unidentified developer:
     right-click it → **Open** → **Open**.
4. The installer will:
   - install Python 3.12 and FFmpeg via Homebrew if missing
     (if Homebrew itself is missing, it prints the one line you need to run first),
   - create a private Python environment (`.venv`),
   - install the Python packages,
   - download both Qwen3-TTS models (**≈5 GB — 10–40 minutes** depending on
     your internet speed).
5. Wait for **“✅ INSTALL COMPLETE”**.

## 5. Starting the app

Double-click **`start.command`**. Your browser opens
<http://127.0.0.1:7860> automatically. The first generation after each start
takes a little longer because the model is loaded into memory.

## 6. Stopping the app

Close the Terminal window that `start.command` opened
(or click it and press `Control + C`).

## 7. Using the app

- **Text** — type Chinese text into the big box.
- **Voice and Model** — 0.6B is fast; 1.7B sounds better but is slower.
- **Generation mode** — Single Word (slow and extra clear), Vocabulary Phrase,
  Sentence (natural), Paragraph (natural pacing with pauses).
- **Speed** — 0.5×–1.5×, pitch is preserved.
- **Style instruction** — optional, e.g. “Speak slowly and clearly like a
  Chinese teacher”.
- **Pauses and Repeat** — add silence before/after/between repeats; repeating a
  word 2–5 times is great for vocabulary practice.
- **Batch Generation** — paste one word/sentence per line, or upload a CSV with
  a `text` column (optional: `voice`, `model`, `mode`, `speed`). You get a ZIP
  with all audio plus a results CSV.

Repeated requests with identical settings are served instantly from the cache.

## 8. Where files are saved

| What | Where |
|---|---|
| Generated audio, batch ZIPs, results CSVs | `outputs/` |
| Cache records (safe to delete) | `cache/` |
| Model files | `models/` |

## 9. How to uninstall everything

1. Delete the project folder (this removes the app, models, outputs and `.venv`).
2. Optional, only if you don't use them for anything else:
   `brew uninstall ffmpeg python@3.12`

Nothing is installed outside the project folder except Homebrew, Python 3.12
and FFmpeg.

## 10. Common problems

| Problem | Fix |
|---|---|
| “Homebrew is required” during install | Copy the printed `/bin/bash -c "$(curl …)"` line into Terminal, run it, then run `install.command` again. |
| “FFmpeg is not installed” in the app | Run `brew install ffmpeg` in Terminal, then restart the app. |
| “Model files missing” warning in the app | Run `install.command` again (it resumes/repairs the download). |
| “A Python dependency is missing” | Run `install.command` again. |
| Browser doesn't open | Open <http://127.0.0.1:7860> manually. |
| Port already in use | The app automatically starts on a free port — read the Terminal window for the address. |
| Generation is very slow / Mac freezes | Use the Fast (0.6B) model and close other heavy apps — 16 GB RAM is enough for one model at a time, and the app never loads both at once. |
| “cannot be opened because it is from an unidentified developer” | Right-click the `.command` file → **Open** → **Open**. |
