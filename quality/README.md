# Private Iraqi Arabic quality evaluation

Use at least 20 consented, representative Iraqi/Baghdadi clips covering fast speech, noise, code-switching, names, negation, dates, prices, and telephone numbers. Media and manifests under this directory are ignored by Git.

Create `manifest.json` as a JSON array:

```json
[
  {
    "id": "clip-001",
    "reference_arabic": "human-reviewed faithful transcript",
    "current_transcript": "production pipeline output",
    "candidate_transcript": "candidate pipeline output",
    "critical_terms": ["Iraqi term", "person name"]
  }
]
```

Run:

```bash
python -m telegram_transcript.quality quality/manifest.json
```

The report uses corpus-weighted WER/CER rather than averaging clips equally. The automatic gate requires at least 10% relative WER improvement and no regression in critical-term recall or exact numeric-token preservation.

Before changing the production default, also compare end-to-end latency and provider cost, review the worst per-clip regressions, and perform a blind 1–5 human review of the Persian output for adequacy and naturalness. Automatic character metrics cannot establish translation quality.
