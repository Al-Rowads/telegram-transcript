# Telegram Video Transcription Bot

A Python Telegram bot that receives a video, extracts speech audio with `ffmpeg`, sends the audio to OpenAI transcription, and replies with the transcript.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages and video documents.
- Works in private chats, groups, and supergroups.
- Ignores text, non-video media, invalid commands, and oversized videos without replying.
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
- `ALLOWED_TELEGRAM_TOPIC_ID`: Telegram forum topic ID to process in groups/supergroups. When set, group messages outside that topic are ignored. Private chats keep working.
- `MAX_VIDEO_MB`: maximum Telegram video size accepted by the bot. Defaults to `100`. When using Telegram's hosted Bot API, downloads are still capped at 20 MB by Telegram before this bot can read the file.
- `MAX_OPENAI_AUDIO_MB`: maximum generated audio chunk size. Defaults to `24`, below OpenAI's 25 MB upload limit.
- `TELEGRAM_API_BASE_URL`: optional custom Bot API base URL, for example `http://telegram-bot-api:8081/bot`.
- `TELEGRAM_API_BASE_FILE_URL`: optional custom Bot API file URL, for example `http://telegram-bot-api:8081/file/bot`.
- `TELEGRAM_LOCAL_MODE`: set to `true` when using a local Telegram Bot API server so the bot can process downloads larger than the hosted 20 MB limit.
- `AUDIO_TEMPO`: tempo for the generated MP3. Defaults to `1.0` for normal speed. Use values below `1` to slow fast speakers while preserving pitch.
- `MAX_CONCURRENT_JOBS`: simultaneous transcription jobs. Defaults to `1`.

## Telegram File Size Notes

Telegram's hosted Bot API `getFile` endpoint only lets bots download files up to 20 MB. A 22.2 MB MP4 may work when sent as a video because Telegram can compress or transform it before the bot receives it, while sending it as a file preserves the original bytes and can exceed the download limit.

To process larger video documents, run a local Telegram Bot API server, set `TELEGRAM_API_BASE_URL`, `TELEGRAM_API_BASE_FILE_URL`, and `TELEGRAM_LOCAL_MODE=true`, then set `MAX_VIDEO_MB` to the largest size you want this bot to accept. Telegram documents this local-server option at <https://core.telegram.org/bots/api#using-a-local-bot-api-server>.

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

For groups with topics, set `ALLOWED_TELEGRAM_TOPIC_ID` to make the bot process only one topic. The bot reads Telegram's `message_thread_id`, so send a test message in the target topic and temporarily log that value if you need to discover it.

Use `/tempo 1.2` in a group or supergroup to change the runtime audio tempo for future videos. Valid values are from `0.5` to `2.0`; the value resets to `AUDIO_TEMPO` after restart.

If the bot should process ordinary group video messages without being mentioned or replied to, disable privacy mode for the bot in BotFather.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
