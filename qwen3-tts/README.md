# Chang Local AI Toolbox (日日向上)

One compact local AI app for Apple Silicon Macs — **everything runs on your
Mac**: no cloud, no login, no database. Your text, audio and documents never
leave your computer.

| Tab | What it does | Model (installed separately) |
|---|---|---|
| 🎧 TTS | Chinese text-to-speech: word / phrase / sentence / paragraph modes, style instructions, batch + CSV, caching, WAV/MP3 | Qwen3-TTS 0.6B (≈1.8 GB) and/or 1.7B (≈2.9 GB) |
| 💬 Chat | Chat & coding assistant, system prompt, temperature, file upload, project-folder questions | Qwen3 4B 4-bit (≈2.3 GB) |
| 🖼 Vision | Image / homework / screenshot analysis, PDF pages | Qwen3-VL 4B 4-bit (≈3.2 GB) |
| 🎙 Transcribe | Speech-to-text for WAV/MP3/M4A/MP4, zh/en/vi + auto-detect, translate-to-English, TXT/SRT export | Whisper large-v3-turbo (≈1.7 GB) |
| 🔎 OCR | Text extraction (zh / en / vi) from images & PDFs | built into macOS — no download |
| 📚 Documents | Index PDF/DOCX/TXT/MD files, semantic search, optional re-ranking & answers | Qwen3-Embedding 0.6B (≈0.4 GB), optional Reranker (≈0.4 GB) |
| 🏗 Generator | Structured JSON generation (dictionaries, examples, quizzes …) from your own document collections: word queues, validation, repair, review, exports | Qwen3 4B + Embedding (Reranker optional) |
| 📦 Models | Install / remove / verify models, presets, disk-space checks | — |
| ⚙️ Settings | Folders (incl. external SSD for models), theme, auto-unload, cache cleaning | — |

Only **one heavy model is in memory at a time** (safe for 16 GB Macs); the
status bar shows the loaded model, RAM use and current task, plus an
**Unload model** button.

## Requirements

- Apple Silicon Mac (M1 or newer), tested on MacBook Air M2 / 16 GB
- macOS with Homebrew; Python 3.12 and FFmpeg are installed automatically
- Disk space depends on the models you choose (up to ~8 GB for the "Full" preset)

## First-time installation

1. Open Terminal once (⌘ + Space, type "Terminal") to make the scripts
   executable — drag the project folder into the Terminal window to
   auto-fill the path (paths with spaces are fine):

   ```bash
   chmod +x "/path/to/qwen3-tts/"*.command
   ```

2. In **Finder**, double-click **`install.command`**.
   If macOS warns about an unidentified developer: right-click → **Open** → **Open**.
   The installer checks Homebrew / Python 3.12 / FFmpeg, creates or reuses
   `.venv`, installs the pinned Python packages, then **asks which models to
   install**:

   | Preset | Contents |
   |---|---|
   | 1. Minimal TTS | Qwen3-TTS 0.6B |
   | 2. TTS Quality | both TTS models |
   | 3. Teaching Essentials | both TTS + Whisper + OCR + Embedding |
   | 4. Developer | both TTS + Qwen3 4B + Whisper + OCR |
   | 5. Full Installation | everything (~8 GB download) |
   | 6. Custom | pick models one by one |

   Nothing is downloaded without asking; the estimated download size and your
   free disk space are shown first.

3. Start the app by double-clicking **`start.command`** —
   it opens <http://127.0.0.1:7860> in your browser.

Models can be installed or removed **at any time** from the 📦 Models tab or
by double-clicking **`manage_models.command`**. Removing a model deletes only
its downloaded weights — never your audio, documents or indexes.

## Where things live

| What | Where |
|---|---|
| Model weights | `~/Library/Application Support/LocalAIToolbox/models` (changeable in Settings, e.g. to an external SSD) |
| Existing TTS weights from the old app | `models/` inside the project (still used, not re-downloaded) |
| Generated audio & transcripts | `outputs/` |
| Document search indexes | `data/indexes/` |
| Generator source collections | `data/collections/<name>/` (copied files + per-collection index) |
| Generator projects | `data/generator_projects/<id>/` (config, queue, results.json, backups, exports) |
| TTS audio cache | `cache/` |
| Temporary files | `cache/temp/` (safe to clear from Settings) |

## 🏗 Generator — structured JSON from your own documents

Turns word lists into validated JSON entries (e.g. an HSK dictionary), one
word at a time, using **only your uploaded reference documents** as evidence.

**Quick start**

1. **Generator tab → B· Source collections → Manage collections**: create a
   collection (e.g. *Official HSK word lists*), add files
   (PDF / scanned PDF / DOCX / TXT / MD / CSV / JSON / PNG / JPG — scans and
   images are OCR'd automatically), then press **🔄 Rebuild index**.
   Make one collection per source type — they are never merged.
2. **A· Project**: create a project (e.g. *HSK Dictionary*).
3. Tick the collections the project may use; optionally mark some as
   **authoritative** and map fields to collections in
   *Field-specific sources* (e.g. `{"examples": ["Example sentences"]}`).
4. **D· Prompt / schema**: paste your prompt template (variables like
   `{{word}}`, `{{retrieved_sources}}`, `{{json_schema}}`) and either a JSON
   Schema or one example object; list the required keys.
5. **C· Input words**: paste words (one per line or 、-separated) or upload
   TXT / CSV / JSON (columns: `hanzi, pinyin, hsk_old, hsk_new, topic, notes,
   expected_pos, custom_instruction`). Duplicates are detected automatically.
6. Press **💾 Save project settings**, then **▶️ Start**.
   Each word is retrieved → generated → validated → repaired if needed →
   appended to `results.json` (always valid JSON, atomic writes, timestamped
   backups). Failures land in `failures.json` and can be retried.
7. Watch progress, pause/resume/stop/skip any time — the queue survives app
   and Mac restarts (press Start again to continue).
8. Review each word in **H· Review & edit** (sources shown beside the JSON),
   approve/reject, edit inline, or regenerate single fields.
9. Export as JSON / JSONL / CSV / ZIP from the Results panel.

Output tokens go up to **16,384** (default 8,192); the context estimator
warns before anything overflows and retrieved text is trimmed before the
output allowance is ever reduced.

## Interface language

The header pill switches the TTS tab between Tiếng Việt / English / 中文
(remembered per browser).

## Troubleshooting

- **"… is not installed. Open the Models tab"** — that feature's model has not
  been downloaded yet; install it from 📦 Models.
- **Slow first response** — the model is loading; watch the status bar.
- **Memory warning in the status bar** — close other apps or press
  ⏏️ Unload model.
- **FFmpeg missing** — `brew install ffmpeg`, then restart the app.
