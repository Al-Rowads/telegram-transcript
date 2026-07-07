# Telegram Video Transcription Bot

A Python Telegram bot that receives video or audio media, downloads it with Telethon, extracts speech audio with `ffmpeg`, sends the audio to a selectable transcription provider, and replies with the transcript.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages, video documents, audio messages, audio documents, and voice notes.
- Works in private chats, groups, and supergroups.
- Ignores text, unsupported media, invalid commands, and oversized media without replying.
- Downloads bot media through Telethon, allowing media up to Telegram's 2 GiB file limit.
- Converts media audio to slowed mono 16 kHz MP3 with `ffmpeg`.
- Splits converted audio longer than 1300 seconds into 1300-second chunks before sending each chunk to the selected provider.
- Uses Deepgram Nova-3 Arabic for speech-to-text by default.
- Supports runtime transcription model selection with `/model`: Deepgram Nova-3, OpenAI `gpt-4o-transcribe-diarize`, or Gemini `gemini-3.5-flash`.
- Optionally translates each SRT cue to exactly one Persian line and derives a line-by-line Arabic/Persian transcript from the translated SRT.
- Sends raw transcripts first, `transcript.srt` second when timestamps are available, and the bilingual transcript third when `REFINE=true`.
- Sends short transcripts as Telegram messages and long transcripts as `.txt` documents.
- Optional Telegram user allowlist to control usage.
- Dockerized runtime with `ffmpeg` included.

## Environment

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:

- `TELEGRAM_BOT_TOKEN`: token from BotFather.
- `TELEGRAM_API_ID`: Telegram API ID from https://my.telegram.org for Telethon downloads.
- `TELEGRAM_API_HASH`: Telegram API hash from https://my.telegram.org for Telethon downloads.
- `DEEPGRAM_API_KEY`: Deepgram API key for transcription.
- `OPENAI_API_KEY`: OpenAI API key when `REFINE=true` or when selecting OpenAI transcription with `/model openai`.

Optional variables:

- `DEEPGRAM_TRANSCRIBE_MODEL`: defaults to `nova-3`.
- `DEEPGRAM_LANGUAGE`: defaults to `ar`.
- `OPENAI_REFINE_MODEL`: text model used to add Persian translations to SRT subtitles. Defaults to `gpt-5.4`.
- `GEMINI_API_KEY`: Gemini API key when selecting Gemini transcription with `/model gemini`.
- `REFINE`: set to `true` to run OpenAI SRT translation after transcription, or `false` to return raw ASR output. Defaults to `false`.
- `ALLOWED_TELEGRAM_USER_IDS`: comma-separated Telegram user IDs allowed to use the bot.
- `MAX_VIDEO_MB`: maximum Telegram media size accepted by the bot. Defaults to `2048`, Telegram's 2 GiB media limit.
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

Use `/tempo 1.2` in a group or supergroup to change the runtime audio tempo for future videos. Valid values are from `0.5` to `2.0`; the value resets to `AUDIO_TEMPO` after restart.

Use `/model` to show the active transcription model and available providers. Use `/model deepgram`, `/model openai`, or `/model gemini` to switch transcription for future media. The selected model is runtime-only and resets to Deepgram after restart.

If the bot should process ordinary group video messages without being mentioned or replied to, disable privacy mode for the bot in BotFather.

## Subtitle Files

When the selected transcription provider returns or can produce timestamps, the bot sends `transcript.srt` after the plain transcript. Deepgram timestamps come from Deepgram utterances or words. OpenAI `gpt-4o-transcribe-diarize` returns diarized JSON rather than direct SRT, so the bot renders those speaker/timestamp segments to SRT locally. Gemini is prompted to return valid SRT directly. The plain transcript message is derived by removing SRT cue numbers, timestamps, and green font markup when present.

With `REFINE=true`, each rendered SRT cue is sent to OpenAI separately for Persian translation. The bot preserves cue numbers, timestamps, and source subtitle text locally, appends exactly one normalized Persian line per cue, and rejects empty or malformed translation responses.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
