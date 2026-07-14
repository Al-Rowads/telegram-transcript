from __future__ import annotations

import pytest

from telegram_transcript.quality import QualityManifestError, evaluate_manifest, normalize_arabic_for_evaluation


def test_normalize_arabic_for_evaluation_removes_diacritics_and_punctuation() -> None:
    assert normalize_arabic_for_evaluation("هَلّـا، شلونك؟") == "هلا شلونك"


def test_evaluate_manifest_applies_relative_wer_gate() -> None:
    result = evaluate_manifest(
        [
            {
                "id": "clip-1",
                "reference_arabic": "هلا شلونك اليوم",
                "current_transcript": "هلا شخبارك اليوم",
                "candidate_transcript": "هلا شلونك اليوم",
            }
        ]
    )

    assert result["current_wer"] == pytest.approx(1 / 3)
    assert result["candidate_wer"] == 0
    assert result["passes_ten_percent_wer_gate"] is True


def test_evaluate_manifest_rejects_missing_reference_fields() -> None:
    with pytest.raises(QualityManifestError, match="missing required"):
        evaluate_manifest([{"id": "clip-1"}])
