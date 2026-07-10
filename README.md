# LinguaScribe Bot 🎙️

**[راهنمای کامل فارسی 🇮🇷 →](README.fa.md)**

A Telegram bot that turns long voice notes, audio files, and videos into accurate transcriptions, Persian translations, SRT subtitles, and detailed study-ready summaries — powered by Google's Gemini models.

## The whole point of this bot: it doesn't choke on long files

Most transcription bots fall over the moment you hand them anything longer than a few minutes. This one is built specifically to handle **2, 3, even 10+ hour recordings** — full lectures, entire podcast episodes, multi-hour meetings, whatever you throw at it.

Here's how: instead of trying to send a giant file to Gemini in one shot, the bot measures the audio's real duration, and if it's over 25 minutes, it slices it into consecutive ~25-minute chunks, transcribes each one separately, then stitches everything back together into one continuous transcription, one full SRT file, and one combined summary. You send a 6-hour lecture, you get back one clean transcript and one summary — the chunking happens invisibly in the background. A 10-hour file just means more chunks and more patience, not a failure.

It's built to run for free inside **Google Colab**, using Colab's runtime as the bot's host and Google Drive as persistent storage for results.

## Features

- ⏱️ **Handles genuinely long audio** — automatic chunking and reassembly means hours-long files are no problem, not an edge case
- 🎤 **Transcription** — accurate speech-to-text in the original spoken language, with basic speaker labeling when multiple speakers are distinguishable
- 🌐 **Persian translation** — automatic translation of any transcription into Persian
- 🎬 **SRT subtitle generation** — auto-segmented, timestamped Persian subtitles
- 📝 **Exam-prep summaries** — long-form, structured Persian summaries (key concepts, arguments, examples, analysis) tuned for university-entrance exam study
- 📹 **Video support** — automatically extracts and compresses audio from video files before processing
- 🗜️ **ZIP support** — unpacks ZIP archives and batch-processes any audio or text files found inside
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
 Is it longer than 25 minutes? ──▶ split into ~25-min chunks
        │                                    │
        ▼                                    ▼
 Gemini transcription (original language, per chunk if split)
        │
        ├──▶ Translate to Persian ──▶ Segment ──▶ Generate SRT
        │
        └──▶ Detailed Persian summary (exam-prep structure)
        │
        ▼
 Stitch chunks back into one transcript/SRT/summary
        │
        ▼
 Send results back to Telegram + copy to Google Drive
```

All Gemini calls go through a single retry wrapper (`gemini_request_with_retry`) that cycles through API keys and a ranked list of fallback models per task, so a rate-limited or unavailable model doesn't take the whole bot down — which matters even more on a long file, since a single dropped chunk out of twenty shouldn't ruin the other nineteen.

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
   | `GOOGLE_API_KEY_2` … `GOOGLE_API_KEY_5` | *(optional)* additional keys — the more keys you add, the smoother very long files will process |

3. Run the cell. On first run, Colab will ask you to authorize Google Drive access — accept it so results get backed up automatically.
4. Once you see `Bot @yourbotname started successfully!` in the logs, message your bot on Telegram.

## Setup (local / non-Colab)

The core logic doesn't strictly need Colab, except for how secrets and Drive are loaded. To run it elsewhere:

1. `pip install -r requirements.txt`
2. Install [FFmpeg](https://ffmpeg.org/download.html) (needed by `pydub` for audio/video processing) and make sure it's on your `PATH`.
3. Replace the `google.colab.userdata` / `google.colab.drive` calls near the top of `main.py` with your preferred way of loading secrets (e.g. `python-dotenv` + a `.env` file), and either drop the Drive-mount step or point it at a local folder instead.
4. Run: `python main.py`

## Usage

Once the bot is running, message it on Telegram:

- `/start` — welcome message and feature overview
- `/help` — usage instructions
- Send a **voice message, audio file, video, or ZIP** — including hours-long files. The bot will tell you it's splitting things up if the file is long, then keep working through the chunks until everything's back together.

## Notes & limitations

- A Colab session isn't infinite — free-tier Colab runtimes can disconnect after a stretch of inactivity or after ~12 hours, so an extremely long file (say, a 10+ hour recording) processed on a busy free-tier account could theoretically outlast the session. Colab Pro, or keeping the tab active, reduces this risk.
- The bot is tuned for **Persian output** — translations, subtitles, and summaries always come out in Persian regardless of the source language, since that's the bot's main use case.
- Session files (`*.session`) hold your bot's live Telegram login state — never commit them. They're already excluded via `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
