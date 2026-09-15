# Iraqi Arabic Telegram Transcription Bot

A Telegram bot for faithful Iraqi Arabic transcription and Persian translation of video, audio, and voice messages.

## Pipeline

1. Telethon downloads the Telegram attachment to a per-job temporary directory.
2. `ffmpeg` extracts mono 16 kHz lossless FLAC. Optional tempo changes preserve pitch, and all output timestamps are mapped back to the original media timeline.
3. Audio is split near silence boundaries into provider-safe chunks with overlap ownership boundaries.
4. A timestamp-capable provider transcribes each chunk. The selected provider is tried first; only configured alternatives are used for failed chunks.
5. A structured-output pass refines the validated provider cues into Iraqi/Baghdadi Arabic without changing timestamps or numbers. Successful refinement becomes the Arabic delivery source; the raw provider transcript is retained only for fallback.
6. The refined cues are translated to Persian. Translation must preserve cue identity and numeric values.
7. The bot delivers the refined Arabic transcript, a bilingual refined-Arabic/Persian SRT, and a Persian-only transcript. If Iraqi refinement fails, it falls back atomically to the raw Arabic transcript and subtitles with a warning.

Supported transcription providers are OpenRouter Gemini, Deepgram Nova-3 with `ar-IQ`, OpenRouter Whisper Large V3, and direct OpenAI `whisper-1`. Provider credentials are optional individually; at least the selected provider and OpenRouter cleaning/translation path must be configured.

## Correctness and privacy properties

- Provider text is never interpolated into a system instruction. Cleaning uses a static system policy plus untrusted, structured user data.
- Raw provider cues remain available as fallback artifacts. Successfully refined Iraqi cues become the source for delivered Arabic, SRT rendering, and Persian translation.
- Structured responses must preserve cue IDs and numbers. Malformed cleaning output falls back atomically to raw Arabic with a warning; malformed translation output falls back to refined Arabic subtitles.
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
- `OPENROUTER_API_KEY` for the default transcription model, Iraqi cleaning, and Persian translation

Optional:

- `DEEPGRAM_API_KEY` enables Nova-3. `DEEPGRAM_LANGUAGE` defaults to `ar-IQ`; `DEEPGRAM_KEYTERMS` can contain Iraqi names and domain terms.
- `OPENAI_API_KEY` enables direct OpenAI transcription.
- `OPENROUTER_TRANSCRIPTION_REFINEMENT_MODEL` and `OPENROUTER_REFINE_MODEL` default to `openai/gpt-5.4-mini` for Iraqi Arabic cleaning and Persian translation, respectively. `/refiner gpt` and `/tmodel gpt` select this model.
- `REFINE` controls the initial Persian translation default and defaults to `true`.
- `MAX_VIDEO_MB` defaults to Telegram's 2048 MiB media limit.
- `AUDIO_TEMPO` accepts `0.5`–`2.0` and defaults to `1.0`.
- `MAX_CONCURRENT_JOBS` defaults to `1`. The application maintains a bounded worker queue and keeps command handling responsive.
- `TELEGRAM_REQUEST_TIMEOUT_SECONDS` defaults to `30` for outbound Bot API connections, responses, writes, and connection-pool waits.
- `TELEGRAM_MEDIA_WRITE_TIMEOUT_SECONDS` defaults to `120` for outbound document uploads.
- `SCOPED_STATE_PATH` defaults to `data/bot-state.sqlite3` and stores scoped preferences only.
- `ALLOWED_TELEGRAM_USER_IDS` optionally restricts access.

`RUNTIME_STATE_PATH` remains accepted for older global preferences. `VIDEO_REGISTRY_PATH` identifies a legacy media registry; startup purges that database and its SQLite sidecars automatically before accepting work. It must not overlap `RUNTIME_STATE_PATH`.

Saved GPT-5.5 translation and refinement selections are mapped to GPT-5.4 mini when loaded from either preference store and written back on the next settings save. Other preferences and environment model overrides remain supported.

## Commands

- `/model [gemini|deepgram|whisper|openai]`
- `/refiner [gpt|gemini]`
- `/translate`
- `/tmodel [gemini|gpt|claude]`
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
