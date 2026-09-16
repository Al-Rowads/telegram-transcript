# Iraqi Arabic Telegram Transcription Bot

A Telegram bot for faithful Iraqi Arabic transcription and Persian translation of video, audio, and voice messages.

## Pipeline

1. Telethon downloads the Telegram attachment to a per-job temporary directory.
2. `ffmpeg` extracts mono 16 kHz lossless FLAC. Optional tempo changes preserve pitch, and all output timestamps are mapped back to the original media timeline.
3. Audio is split near silence boundaries into provider-safe chunks with overlap ownership boundaries.
4. Deepgram Nova-3 transcribes each chunk by default. Only the explicitly selected provider is called; failure reports an error without switching to another provider.
5. Iraqi/Baghdadi text refinement and Gemini low-confidence audio correction are both off by default. `/refiner on` and `/correction on` enable these additional paid passes independently.
6. Persian translation uses Gemini 2.5 Flash-Lite by default. Natural translation batches up to 48 cues with at most two neighboring cues per side; literal translation omits neighboring context.
7. The bot delivers Arabic text, bilingual Arabic/Persian SRT, and Persian-only text. Optional refinement becomes the Arabic source when successful; failures retain the valid Arabic source with an explicit warning.

Supported transcription providers are Deepgram Nova-3 with `ar-IQ`, OpenRouter Gemini 3.5 Flash, OpenRouter Whisper Large V3, and direct OpenAI `whisper-1`. The selected provider needs its credentials; there is no automatic provider fallback. The OpenAI Python SDK remains the transport for OpenRouter and direct Whisper.

### Cost controls

Translation sends compact cue IDs and text, keeping timestamps local. Requests are limited to 12 KiB of serialized input, including at most 1 KiB of neighboring context. Output allowances scale with source length and never exceed 8,192 tokens for translation/refinement. Oversized cues are translated in fragments and reassembled under their original IDs and timestamps; spoken content is never truncated to fit a budget.

Translation and refinement allow at most three provider calls per original batch, including retries and smaller-batch recovery; SDK retries are disabled for these calls. Invalid or truncated output is rejected. Flash-Lite thinking is disabled, Qwen Instruct uses no thinking, and manually selected Gemini 3.5 uses its minimum thinking effort.

`lite` selects `google/gemini-2.5-flash-lite`, `qwen` selects `qwen/qwen3-30b-a3b-instruct-2507`, and `gemini` retains `google/gemini-3.5-flash`. These choices apply to both `/tmodel` and `/refiner`. GPT and Claude options have been removed.

Logs record provider-reported input/output/reasoning tokens and USD cost, including responses rejected by validation. Missing usage is logged as unknown (`None`), never estimated as zero. Actual quality and savings must be evaluated on representative, consented clips; cheaper models and reduced context can affect dialect and translation quality.

## Correctness and privacy properties

- Provider text is never interpolated into a system instruction. Cleaning uses a static system policy plus untrusted, structured user data.
- Raw provider cues remain available as fallback artifacts. Successfully refined Iraqi cues become the source for delivered Arabic, SRT rendering, and Persian translation.
- Structured responses must preserve cue IDs. Cleaning rejects number changes; translation reports numeric/script discrepancies as advisory warnings. Malformed output retains the valid Arabic source with a warning.
- Subtitle cues are normalized for readability, chunk overlap is assigned to one owner, and tempo-adjusted timestamps are rescaled to the original video.
- Jobs capture an immutable settings snapshot when queued, so later commands cannot change in-flight work.
- Preferences are isolated per private user or group. Only group administrators can mutate group settings.
- The SQLite state database stores preferences only. The bot does not persist per-media jobs, Telegram message or file identifiers, filenames, sender names, transcripts, or translations.
- Downloaded media, extracted audio, chunks, delivery buffers, and queued object references are released on success, failure, cancellation, rejection, and shutdown.
- Logs may retain random job IDs and anonymous sizes, timings, counts, and provider/model names. They exclude Telegram identifiers, filenames, sender data, transcript content, and provider response payloads.
- `/forget` removes the caller's private preferences, or the group preferences when invoked by an administrator.

Telegram and the configured AI providers still receive data as necessary to process a request. Use `/privacy` for the in-bot summary.

## Configuration

Copy `.env.example` to `.env` and configure it.

Required:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_API_ID` and `TELEGRAM_API_HASH` for Telethon downloads
- `DEEPGRAM_API_KEY` for the default Nova-3 transcription path
- `OPENROUTER_API_KEY` for Persian translation and optional Gemini/refinement passes

Optional:

- `DEEPGRAM_LANGUAGE` defaults to `ar-IQ`; `DEEPGRAM_KEYTERMS` can contain Iraqi names and domain terms. Existing language overrides are preserved.
- `OPENAI_API_KEY` enables direct OpenAI transcription.
- `OPENROUTER_TRANSCRIPTION_REFINEMENT_MODEL` and `OPENROUTER_REFINE_MODEL` default to `google/gemini-2.5-flash-lite`. Custom model overrides remain supported; the operator is responsible for their structured-output support and cost.
- `REFINE` controls the initial Persian translation default and defaults to `true`.
- `MAX_VIDEO_MB` defaults to Telegram's 2048 MiB media limit.
- `AUDIO_TEMPO` accepts `0.5`–`2.0` and defaults to `1.0`.
- `MAX_CONCURRENT_JOBS` defaults to `1`. The application maintains a bounded worker queue and keeps command handling responsive.
- `TELEGRAM_REQUEST_TIMEOUT_SECONDS` defaults to `30` for outbound Bot API connections, responses, writes, and connection-pool waits.
- `TELEGRAM_MEDIA_WRITE_TIMEOUT_SECONDS` defaults to `120` for outbound document uploads.
- `SCOPED_STATE_PATH` defaults to `data/bot-state.sqlite3` and stores scoped preferences only.
- `ALLOWED_TELEGRAM_USER_IDS` optionally restricts access.

`RUNTIME_STATE_PATH` remains accepted for older global preferences. `VIDEO_REGISTRY_PATH` identifies a legacy media registry; startup purges that database and its SQLite sidecars automatically before accepting work. It must not overlap `RUNTIME_STATE_PATH`.

On the first upgrade, JSON preferences (v1–v3) and the SQLite preference database migrate once: existing transcription selections become Deepgram, and correction/refinement start disabled. Later explicit provider selections and pass toggles survive restarts. Known GPT-5.5, GPT-5.4 mini, and Claude Sonnet 4.6 text selections, including environment overrides, map to Flash-Lite. Existing Gemini text selections, translation on/off, style, tempo, and custom models remain intact. JSON preferences are saved as v4; SQLite records migration version 1 using `PRAGMA user_version`.

## Commands

- `/model [gemini|deepgram|whisper|openai]`
- `/refiner [on|off|lite|qwen|gemini]` (selecting a model enables refinement; `off` remembers it)
- `/correction [on|off]` (extra Gemini audio calls for Deepgram only)
- `/translate`
- `/tmodel [lite|qwen|gemini]`
- `/translation [natural|literal]`
- `/tempo <0.5-2.0>` in groups
- `/privacy`
- `/forget`

Private-chat settings belong to that user. Group settings belong to the group and require an administrator for changes. Disable the bot's BotFather privacy mode if it should receive ordinary group media.

## Run

Python 3.12 and `ffmpeg`/`ffprobe` are recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m telegram_transcript.bot
```

Production uses the tested Python 3.12 Linux versions in `requirements.lock`; CI extends it with `requirements-dev.lock`. Install the appropriate file first when reproducing an environment exactly.

Or run the container:

```bash
docker compose up --build
```

The Compose data volume retains scoped preferences only. Temporary media, derived files, and in-memory output buffers are released for every terminal job outcome.

## Legacy privacy migration

Older releases may have a `data/videos.sqlite3` registry containing filenames, sender names, or transcripts. The bot no longer reads or writes that registry and deletes the configured registry plus SQLite sidecars automatically during startup. The standalone migration command remains available for offline cleanup:

```bash
python -m telegram_transcript.privacy_migrate data/videos.sqlite3 --confirm-delete
```

The command permanently drops and compacts a legacy content table while preserving unrelated tables. Startup fails before processing media if the configured legacy registry cannot be purged.

## Quality gate

Use the private-corpus workflow in [`quality/README.md`](quality/README.md). The evaluator reports corpus-weighted WER/CER, per-clip regressions, critical Iraqi term recall, and number preservation. A candidate passes only with at least 10% relative WER improvement and no regression in critical terms or numbers. Persian adequacy and naturalness still require blind human review.

## Tests

```bash
pytest
```

The ffmpeg integration test skips when ffmpeg is unavailable.
