# Telegram Video Transcription Bot

A Python Telegram bot that receives a video, extracts speech audio with `ffmpeg`, sends the audio to OpenAI transcription, and replies with the transcript.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages and video documents.
- Works in private chats, groups, and supergroups.
- Ignores text, commands, non-video media, and oversized videos without replying.
- Converts video audio to slowed mono 16 kHz MP3 with `ffmpeg`.
- Splits audio into safe chunks when it would exceed OpenAI's per-file upload limit.
- Refines raw speech-to-text output into natural Baghdadi Iraqi Arabic before replying.
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

- `OPENAI_TRANSCRIBE_MODEL`: defaults to `gpt-4o-transcribe`.
- `OPENAI_REFINE_MODEL`: text model used to refine raw speech-to-text output. Defaults to `gpt-5.4-mini`.
- `REFINE`: set to `true` to run refinement after transcription, or `false` to return raw ASR output. Defaults to `true`.
- `ALLOWED_TELEGRAM_USER_IDS`: comma-separated Telegram user IDs allowed to use the bot.
- `MAX_VIDEO_MB`: maximum Telegram video size accepted by the bot. Defaults to `100`.
- `MAX_OPENAI_AUDIO_MB`: maximum generated audio chunk size. Defaults to `24`, below OpenAI's 25 MB upload limit.
- `AUDIO_TEMPO`: tempo for the generated MP3. Defaults to `1.0` for normal speed. Use values below `1` to slow fast speakers while preserving pitch.
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

## Group Chats

Add the bot to a group or supergroup to transcribe videos posted there. The bot replies to the original video message so the sender and thread stay clear.

If the bot should process ordinary group video messages without being mentioned or replied to, disable privacy mode for the bot in BotFather.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
