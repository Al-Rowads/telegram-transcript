# Private Quality Evaluation

Place at least 20 representative Iraqi/Baghdadi clips and their human-reviewed references in this directory. Media and manifests in this directory are ignored by Git because they may contain private Telegram content.

Create `manifest.json` as a JSON array. Each object must contain these string fields:

- `id`: stable clip identifier.
- `reference_arabic`: manually corrected Arabic transcript.
- `current_transcript`: output from the current Gemini production path.
- `candidate_transcript`: output from the proposed Deepgram `ar-IQ` pipeline.

Evaluate transcription accuracy with:

```bash
python -m telegram_transcript.quality quality/manifest.json
```

The candidate passes the automatic model-change gate only when relative WER improves by at least 10%. Also record latency, provider cost, omissions, and a blind 1–5 Persian adequacy/naturalness review outside this manifest; do not change the production default if negation, numbers, or identities regress.
