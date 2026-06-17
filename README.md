# Telegram Video Transcription Bot

A Python Telegram bot that receives a video, voice note, or MP3, gets speech audio (extracting it from video with `ffmpeg` when needed), sends the audio to OpenAI transcription, and replies with the transcript.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages and video documents.
- Accepts Telegram voice notes and `.mp3` files, transcribing them directly and skipping the `ffmpeg` step when the audio is already within OpenAI's upload limit.
- Works in private chats, groups, and supergroups.
- Ignores text, unsupported media, invalid commands, and oversized uploads without replying.
- Converts video audio to slowed mono 16 kHz MP3 with `ffmpeg`, and falls back to the same compression for audio uploads that are too large to send directly.
- Splits audio into safe chunks when it would exceed OpenAI's per-file upload limit.
- Converts raw speech-to-text output into a structured Arabic transcript, Persian translation, key-word, and line-break editing aid before replying.
- Sends short transcripts as Telegram messages and long transcripts as `transcript.txt`.
- Optional Telegram user allowlist to control usage.
- Dockerized runtime with `ffmpeg` included.
- Uses Telethon so the bot can download Telegram files up to 2 GiB.

## Environment

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

- `TELEGRAM_BOT_TOKEN`: token from BotFather.
- `TELEGRAM_API_ID`: Telegram API ID from <https://my.telegram.org/apps>.
- `TELEGRAM_API_HASH`: Telegram API hash from <https://my.telegram.org/apps>.
- `OPENAI_API_KEY`: OpenAI API key.

Optional variables:

- `OPENAI_TRANSCRIBE_MODEL`: defaults to `gpt-4o-transcribe`.
- `OPENAI_REFINE_MODEL`: text model used to refine raw speech-to-text output. Defaults to `gpt-5.4-mini`.
- `REFINE`: set to `true` to run refinement after transcription, or `false` to return raw ASR output. Defaults to `true`.
- `ALLOWED_TELEGRAM_USER_IDS`: comma-separated Telegram user IDs allowed to use the bot.
- `ALLOWED_TELEGRAM_TOPIC_ID`: Telegram forum topic ID to process in groups/supergroups. When set, group messages outside that topic are ignored. Private chats keep working.
- `MAX_VIDEO_MB`: maximum Telegram video size accepted by the bot. Defaults to `2048`, Telegram's 2 GiB file limit for this Telethon-based downloader.
- `MAX_OPENAI_AUDIO_MB`: maximum generated audio chunk size. Defaults to `24`, below OpenAI's 25 MB upload limit.
- `AUDIO_TEMPO`: tempo for the generated MP3. Defaults to `1.0` for normal speed. Use values below `1` to slow fast speakers while preserving pitch.
- `MAX_CONCURRENT_JOBS`: simultaneous transcription jobs. Defaults to `1`.

## Telegram File Size Notes

This bot uses Telethon instead of Telegram's hosted Bot API file endpoint, so it is not limited by the hosted Bot API `getFile` 20 MB download cap. Set `MAX_VIDEO_MB` to the largest file size you want to accept, up to `2048`.

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

Add the bot to a group or supergroup to transcribe videos, voice notes, and MP3 files posted there. The bot replies to the original message so the sender and thread stay clear.

For groups with topics, set `ALLOWED_TELEGRAM_TOPIC_ID` to make the bot process only one topic. The bot reads Telegram's `message_thread_id`, so send a test message in the target topic and temporarily log that value if you need to discover it.

Use `/tempo 1.2` in a group or supergroup to change the runtime audio tempo for future videos. Valid values are from `0.5` to `2.0`; the value resets to `AUDIO_TEMPO` after restart.

Use `/noise` before sending a video to apply default one-shot ffmpeg noise reduction to the next video in the same chat or topic. Use `/noise extra` for stronger filtering with high-pass, FFT denoise, and loudness normalization. The setting is consumed by the next accepted video and is not permanent.

If the bot should process ordinary group video messages without being mentioned or replied to, disable privacy mode for the bot in BotFather.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
