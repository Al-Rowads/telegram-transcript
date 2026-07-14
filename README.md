# Telegram Video Transcription Bot

A Python Telegram bot that receives video or audio media, downloads it with Telethon, extracts speech audio with `ffmpeg`, transcribes it through a timestamp-validated provider chain, and replies with the transcript and SRT subtitles.

## Features

- Telegram long polling, so no public webhook URL is required.
- Accepts Telegram video messages, video documents, audio messages, audio documents, and voice notes.
- Works in private chats, groups, and supergroups.
- Ignores text, unsupported media, invalid commands, and oversized media without replying.
- Downloads bot media through Telethon, allowing media up to Telegram's 2 GiB file limit.
- Converts media audio to lossless mono 16 kHz FLAC with `ffmpeg`.
- Splits converted audio into at most 420-second, 13 MiB chunks near silence boundaries, including 1.5 seconds of overlap, and removes duplicate overlap cues.
- Uses OpenRouter `google/gemini-3.5-flash` for speech-to-text by default.
- Automatically retries only a failed chunk using the selected model first, then the remaining providers in Gemini, Deepgram, Whisper Large V3, OpenAI preference order.
- Supports runtime primary-model selection with `/model`: OpenRouter Gemini 3.5 Flash, direct Deepgram Nova-3, OpenRouter `openai/whisper-large-v3`, or direct OpenAI `whisper-1`.
- Supports runtime translation model selection with `/tmodel`: Gemini 3.5 Flash, GPT-5.5, or Claude Sonnet 4.6 through OpenRouter.
- Supports `/translation natural` and `/translation literal`; legacy `v2` and `normal` aliases remain accepted.
- Persists successful command selections across bot and container restarts.
- Optionally translates batches of up to 12 SRT cues with preceding and following context, while preserving one Persian line and the original timing for every cue.
- For Deepgram results, preserves word confidence and rechecks only low-confidence Iraqi Arabic cues with a constrained Gemini candidate and resolver.
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
- `OPENROUTER_API_KEY`: OpenRouter key for Gemini and Whisper Large V3 transcription and optional Persian refinement.
- `DEEPGRAM_API_KEY`: Deepgram API key for Nova-3 transcription and automatic fallback.
- `OPENAI_API_KEY`: direct OpenAI API key for the timestamp-capable `whisper-1` fallback.

Optional variables:

- `DEEPGRAM_TRANSCRIBE_MODEL`: defaults to `nova-3`.
- `DEEPGRAM_LANGUAGE`: defaults to Iraqi Arabic (`ar-IQ`).
- `DEEPGRAM_KEYTERMS`: optional comma-separated Nova-3 keyterms for important names and terminology.
- `OPENROUTER_REFINE_MODEL`: OpenRouter text model used to add Persian translations to SRT subtitles. Defaults to `openai/gpt-5.5`.
- `REFINE`: set to `true` to run OpenAI SRT translation after transcription, or `false` to return raw ASR output. Defaults to `false`.
- `ALLOWED_TELEGRAM_USER_IDS`: comma-separated Telegram user IDs allowed to use the bot.
- `MAX_VIDEO_MB`: maximum Telegram media size accepted by the bot. Defaults to `2048`, Telegram's 2 GiB media limit.
- `AUDIO_TEMPO`: tempo for the generated FLAC. Defaults to `1.0` for normal speed. Use values below `1` to slow fast speakers while preserving pitch.
- `MAX_CONCURRENT_JOBS`: simultaneous transcription jobs. Defaults to `1`.
- `RUNTIME_STATE_PATH`: JSON file used for durable bot-wide command settings. Defaults to `data/runtime-settings.json`.

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

Use `/tempo 1.2` in a group or supergroup to change the audio tempo for future videos. Valid values are from `0.5` to `2.0`.

Use `/model` to show the primary transcription model and available providers. Use `/model gemini`, `/model deepgram`, `/model whisper`, or `/model openai` to select the first provider tried for each chunk. Failed chunks automatically try the remaining providers in Gemini, Deepgram, Whisper Large V3, OpenAI preference order; this temporary fallback does not change the saved primary model.

Use `/tmodel` to show the active translation model and whether translation is enabled. Use `/tmodel gemini`, `/tmodel gpt`, or `/tmodel claude` to switch translation for future media. Selecting a model does not enable translation when `REFINE=false`.

Use `/translation` to show the active translation style. Use `/translation natural` for conversational Persian informed by five preceding translated cues and five following Arabic cues, or `/translation literal` for a closer standard-Persian rendering. `/translation v2` maps to `natural` and `/translation normal` maps to `literal` for compatibility.

Successful `/tempo`, `/model`, `/tmodel`, and `/translation` changes are bot-wide and written atomically to `RUNTIME_STATE_PATH`. Docker Compose mounts `/app/data` as a named volume so selections survive container recreation.

If the bot should process ordinary group video messages without being mentioned or replied to, disable privacy mode for the bot in BotFather.

## Subtitle Files

The bot sends `transcript.srt` after the plain transcript. Gemini is prompted to return valid SRT directly, Deepgram timestamps come from utterances or words, and both OpenRouter Whisper Large V3 and direct OpenAI `whisper-1` return structured segment timestamps that are rendered locally. A provider result with spoken text but no valid timestamps is rejected and the same chunk is tried with the next provider. The plain transcript is derived from the validated cues.

With `REFINE=true`, rendered SRT cues are translated through OpenRouter in batches of up to 12. The bot preserves cue numbers, timestamps, and source subtitle text locally, appends exactly one validated Persian line per cue, retries one malformed batch, then recursively splits it. If translation ultimately fails, the validated raw transcript and SRT are still delivered with a warning.

## Quality Evaluation

The production transcription default remains Gemini until a representative private Iraqi Arabic evaluation passes the quality gate. Follow [`quality/README.md`](quality/README.md) to compare current and candidate transcripts. The included evaluator reports WER, CER, relative WER improvement, and whether the candidate clears the required 10% relative WER improvement.

## Tests

```bash
pytest
```

The `ffmpeg` integration test skips automatically when `ffmpeg` is not installed locally.
