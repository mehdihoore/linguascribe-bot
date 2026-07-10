# LinguaScribe Bot 🎙️

A Telegram bot that turns voice messages, audio files, and videos into
accurate transcriptions, Persian translations, SRT subtitles, and detailed
study-ready summaries — powered by Google's Gemini models.

Built to run for free inside **Google Colab**, using Colab's GPU-free
runtime as the bot's host and Google Drive as persistent storage for
results.

## Features

- 🎤 **Transcription** — accurate speech-to-text in the original spoken language, with basic speaker labeling when multiple speakers are distinguishable
- 🌐 **Persian translation** — automatic translation of any transcription into Persian
- 🎬 **SRT subtitle generation** — auto-segmented, timestamped Persian subtitles
- 📝 **Exam-prep summaries** — long-form, structured Persian summaries (key concepts, arguments, examples, analysis) tuned for university-entrance exam study
- 📹 **Video support** — automatically extracts and compresses audio from video files before processing
- 🗜️ **ZIP support** — unpacks ZIP archives and batch-processes any audio or text files found inside
- ✂️ **Long-audio splitting** — files longer than 25 minutes are automatically chunked, processed in parallel pieces, and stitched back together
- 🔑 **Multi-key API rotation** — cycles through up to 5 Gemini API keys and multiple model fallbacks (Gemini/Gemma) to ride out rate limits and transient errors
- 💾 **Google Drive backup** — every transcription and summary is automatically saved to a `STTCOLAB` folder in the user's Drive

## How it works

```
Telegram message (voice/audio/video/zip)
        │
        ▼
 Download & normalize (pydub/FFmpeg → mono, 16kHz, ogg/opus)
        │
        ▼
 Gemini transcription (original language)
        │
        ├──▶ Translate to Persian ──▶ Segment ──▶ Generate SRT
        │
        └──▶ Detailed Persian summary (exam-prep structure)
        │
        ▼
 Send results back to Telegram + copy to Google Drive
```

All Gemini calls go through a single retry wrapper
(`gemini_request_with_retry`) that cycles through API keys and a ranked
list of fallback models per task, so a rate-limited or unavailable model
doesn't take the whole bot down.

## Prerequisites

- A Telegram account and a bot token from [@BotFather](https://t.me/BotFather)
- A Telegram **API ID and API hash** from [my.telegram.org](https://my.telegram.org)
- One or more **Gemini API keys** from [Google AI Studio](https://aistudio.google.com/apikey)
- A Google account (for Colab + Drive, if running the intended way)

## Setup (Google Colab — recommended)

1. Open `main.py` as a Colab notebook (or copy its contents into a new Colab notebook cell).
2. In Colab, open the 🔑 **Secrets** panel (left sidebar) and add:

   | Secret name | Value |
   |---|---|
   | `TELEGRAM_API_ID` | Your numeric API ID |
   | `TELEGRAM_API_HASH` | Your API hash |
   | `TELEGRAM_BOT_TTS` | Your bot token from BotFather |
   | `GOOGLE_API_KEY_1` | A Gemini API key |
   | `GOOGLE_API_KEY_2` … `GOOGLE_API_KEY_5` | *(optional)* additional keys for rotation |

3. Run the cell. On first run, Colab will ask you to authorize Google Drive access — accept it so results can be backed up automatically.
4. Once you see `Bot @yourbotname started successfully!` in the logs, message your bot on Telegram.

## Setup (local / non-Colab)

The core logic has no hard Colab dependency except secret loading and
Drive mounting. To run it elsewhere:

1. `pip install -r requirements.txt`
2. Install [FFmpeg](https://ffmpeg.org/download.html) (required by `pydub` for audio/video processing) and make sure it's on your `PATH`.
3. Replace the `google.colab.userdata` / `google.colab.drive` calls near the top of `main.py` with your preferred secret source (e.g. `python-dotenv` + a `.env` file) and drop the Drive-mount step, or point it at a local folder instead.
4. Run: `python main.py`

## Usage

Once the bot is running, message it on Telegram:

- `/start` — welcome message and feature overview
- `/help` — usage instructions
- Send a **voice message, audio file, video, or ZIP** — the bot handles the rest automatically

## Notes & limitations

- Audio longer than 25 minutes is automatically split into chunks and reassembled; very long files will take proportionally longer to process.
- The bot is tuned for **Persian output** (translations, subtitles, summaries are always produced in Persian regardless of the source language), since that's its primary use case.
- Session files (`*.session`) contain your bot's Telegram login state — never commit them. They're already excluded via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
