# Telegram Video Transcription Bot

A Python Telegram bot that receives a video, extracts speech audio with `ffmpeg`, sends the audio to OpenAI transcription, and replies with the transcript.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages and video documents.
- Converts video audio to mono 16 kHz MP3 with `ffmpeg`.
- Splits audio into safe chunks when it would exceed OpenAI's per-file upload limit.
- Refines raw Whisper output into natural Baghdadi Iraqi Arabic before replying.
- Sends short transcripts as Telegram messages and long transcripts as `transcript.txt`.
- Optional Telegram user allowlist to control usage.
- Dockerized runtime with `ffmpeg` included.

## Environment

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

- `TELEGRAM_BOT_TOKEN`: token from BotFather.
- `OPENAI_API_KEY`: OpenAI API key.

Optional variables:

- `OPENAI_TRANSCRIBE_MODEL`: defaults to `whisper-1`.
- `OPENAI_REFINE_MODEL`: text model used to refine raw Whisper output. Defaults to `GPT5.4-mini`.
- `ALLOWED_TELEGRAM_USER_IDS`: comma-separated Telegram user IDs allowed to use the bot.
- `MAX_VIDEO_MB`: maximum Telegram video size accepted by the bot. Defaults to `100`.
- `MAX_OPENAI_AUDIO_MB`: maximum generated audio chunk size. Defaults to `24`, below OpenAI's 25 MB upload limit.
- `MAX_CONCURRENT_JOBS`: simultaneous transcription jobs. Defaults to `1`.

## Run Locally

Install `ffmpeg` first:

```bash
brew install ffmpeg
```

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m telegram_transcript.bot
```

## Run With Docker

Docker includes `ffmpeg`, so only `.env` is needed:

```bash
docker compose up --build
```

No ports are exposed because the bot uses long polling.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
