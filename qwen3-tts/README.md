# Local Chinese Text-to-Speech (Qwen3-TTS)

A simple, fully **local** Mandarin Chinese text-to-speech web app for Apple
Silicon Macs (M1/M2/M3/M4). It runs entirely on your Mac — **no cloud, no
login, no account, no database**. Your text never leaves your computer.

It uses:

- **Python 3.12**
- **Gradio** (the web interface)
- **MLX-Audio** with two Qwen3-TTS models
- **FFmpeg** (for MP3 conversion, pauses, and repeats)

Two models are included:

| Choice          | Model                                                        | Notes                    |
| --------------- | ------------------------------------------------------------ | ------------------------ |
| Fast (default)  | `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`          | Quicker, lighter         |
| Higher Quality  | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`          | Better voice, a bit slower |

---

## For a completely clean Mac — step by step

### 1. Open Terminal
- Press **Command (⌘) + Space** to open Spotlight.
- Type **Terminal** and press **Return**. A text window opens. You only need it
  for a couple of commands.

### 2. Save the project
- Put this whole `qwen3-tts` folder anywhere you like, for example inside your
  **Documents** folder. Folder names with spaces are fine.

### 3. Make the installer runnable (one time)
In Terminal, type `chmod +x ` (with a trailing space), then **drag the
`install.command` file from Finder into the Terminal window** and press
**Return**. Do the same for `start.command`. For example:

```bash
chmod +x "/path/to/qwen3-tts/install.command"
chmod +x "/path/to/qwen3-tts/start.command"
```

(Dragging the file fills in the correct path automatically, even with spaces.)

### 4. Run the installer
- In **Finder**, double-click **`install.command`**.
- If macOS says it "cannot be opened because it is from an unidentified
  developer", **right-click** the file → **Open** → **Open**. You only do this
  once.
- The installer will:
  - check you are on Apple Silicon,
  - make sure Homebrew, Python 3.12, and FFmpeg are present,
  - create a private `.venv` environment,
  - install the Python packages,
  - **download both models**,
  - create the `outputs` and `cache` folders.

> **If Homebrew is missing**, the installer will stop and print one line for you
> to paste into Terminal to install Homebrew, then just run `install.command`
> again.

### 5. How long the download takes / storage needed
- **Downloading the models** can take from a few minutes to ~30 minutes
  depending on your internet speed. This happens **only once**.
- The models are stored **inside the project**, in the `models/` folder (not in
  a hidden system cache), so everything lives in one place.
- **Approximate storage:**
  - Python packages + environment: ~2–4 GB
  - Both models (8-bit), in `models/`: ~4.7 GB total (~1.8 GB + ~2.9 GB)
  - Generated audio: small (a few KB–MB per clip)
  - **Plan for about 8–10 GB free** to be comfortable.

### 6. Start the app
- Double-click **`start.command`**.
- Your browser opens automatically at **http://127.0.0.1:7860**.
- If it does not open, type that address into your browser yourself.
- The **first time** you use a model in the app it may take a moment to load
  into memory; a "Loading model…" message appears.

### 7. Stop the app
- Close the Terminal window that `start.command` opened, **or** click that
  window and press **Control + C**.

### 8. Where generated files are saved
- Everything is saved in the **`outputs`** folder inside `qwen3-tts`.
- Filenames include the Chinese text, voice, model, mode, and a short hash, e.g.
  `你好_Vivian_0.6B_word_1a2b3c4d.mp3`.
- Batch runs also produce a `batch_YYYYMMDD_HHMMSS.zip` and a
  `batch_results_YYYYMMDD_HHMMSS.csv` in `outputs`.

---

## Using the app

- **Text** — paste Chinese text into the large box (Chinese only; there is no
  language selector).
- **Voice and Model** — pick a voice (Vivian, Serena, Uncle_Fu, Dylan, Eric) and
  the Fast or Higher Quality model.
- **Pronunciation Settings**
  - **Mode**: *Single Word* (slower, clearer), *Vocabulary Phrase* (clear with a
    short pause), *Sentence* (natural), *Paragraph* (natural with sentence
    pauses).
  - **Speed**: 0.5x–1.5x (handled by the model, so pitch stays natural).
  - **Style instruction** (optional): e.g. "像中文老师一样，慢速清晰地朗读".
  - **Pauses**: before / after / between repeats (0–2 seconds).
  - **Repeat**: 1–5 times — handy for vocabulary practice.
- **Output Settings** — MP3 (default) or WAV; MP3 quality 128 / 192 (default) /
  256 kbps.
- **Generate** — click **Generate Audio**. You get a player, a download, the
  saved path, the model and voice used, the generation time, and whether it was
  newly generated or loaded from cache.
- **Batch Generation** — paste one item per line **or** upload a CSV with
  columns `text, voice, model, mode, speed` (only `text` is required). You get a
  results table, a ZIP of all audio, and a results CSV.

### Caching
The app never regenerates audio when **all** settings are identical (text,
model, voice, mode, speed, style, pauses, repeat count, format, and quality). It
uses a deterministic hash, so repeated clicks are instant.

---

## Uninstall everything

1. **Delete the project folder** (`qwen3-tts`). Because the models live inside
   it (in `models/`), this removes everything: the app, the `.venv`, both
   models, the `cache`, and all generated audio. That's the whole uninstall.
2. *(Optional)* If you no longer want the shared tools:
   ```bash
   brew uninstall ffmpeg
   brew uninstall python@3.12
   ```
   Skip this if other apps use them.

---

## Common error fixes

| Problem | Fix |
| --- | --- |
| **"cannot be opened… unidentified developer"** | Right-click the `.command` file → **Open** → **Open**. |
| **`install.command` does nothing / "permission denied"** | Run `chmod +x` on it (see step 3). |
| **"Homebrew is required"** | Paste the Homebrew install line the installer prints, then run `install.command` again. |
| **"FFmpeg not found" in the app** | Run `brew install ffmpeg`, then restart the app. |
| **Model download failed / network error** | Check your internet connection and run `install.command` again (downloads resume). |
| **"Insufficient memory"** | Close other apps, use the **Fast (0.6B)** model, and shorter text. |
| **App won't start / "Dependencies missing"** | Run `install.command` again. |
| **Port 7860 already in use** | Close the other app on that port, or quit any earlier copy of this app. |
| **CSV rejected** | Make sure the first row has a `text` column header. Allowed columns: `text, voice, model, mode, speed`. |

---

## Notes / limitations
- Apple Silicon only (MLX runs on the Mac GPU). Intel Macs are not supported.
- Only **one** model is kept in memory at a time to stay comfortable on 16 GB
  RAM; switching models unloads the previous one first.
- Speed is applied by the model itself (natural, pitch-preserving). Pauses,
  repeats, and MP3 conversion are done with FFmpeg.
