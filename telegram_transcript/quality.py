from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


class QualityManifestError(RuntimeError):
    """Raised when a transcription quality manifest is invalid."""


def normalize_arabic_for_evaluation(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("ـ", "")
    normalized = ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = re.sub(r"[\W_]+", " ", normalized)
    return " ".join(normalized.split())


def edit_distance(reference: Sequence[str], candidate: Sequence[str]) -> int:
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_item in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1] + (reference_item != candidate_item),
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, candidate: str) -> float:
    reference_words = normalize_arabic_for_evaluation(reference).split()
    candidate_words = normalize_arabic_for_evaluation(candidate).split()
    if not reference_words:
        raise ValueError("Reference transcript must contain at least one word.")
    return edit_distance(reference_words, candidate_words) / len(reference_words)


def character_error_rate(reference: str, candidate: str) -> float:
    reference_characters = list(normalize_arabic_for_evaluation(reference).replace(" ", ""))
    candidate_characters = list(normalize_arabic_for_evaluation(candidate).replace(" ", ""))
    if not reference_characters:
        raise ValueError("Reference transcript must contain at least one character.")
    return edit_distance(reference_characters, candidate_characters) / len(reference_characters)


def evaluate_manifest(payload: Any) -> dict[str, object]:
    if not isinstance(payload, list) or not payload:
        raise QualityManifestError("Quality manifest must be a non-empty JSON array.")

    current_word_errors = []
    candidate_word_errors = []
    current_character_errors = []
    candidate_character_errors = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise QualityManifestError(f"Quality manifest item {index} must be an object.")
        required = {"id", "reference_arabic", "current_transcript", "candidate_transcript"}
        if not required.issubset(item) or not all(isinstance(item[key], str) for key in required):
            raise QualityManifestError(f"Quality manifest item {index} is missing required string fields.")
        current_word_errors.append(word_error_rate(item["reference_arabic"], item["current_transcript"]))
        candidate_word_errors.append(word_error_rate(item["reference_arabic"], item["candidate_transcript"]))
        current_character_errors.append(character_error_rate(item["reference_arabic"], item["current_transcript"]))
        candidate_character_errors.append(character_error_rate(item["reference_arabic"], item["candidate_transcript"]))

    current_wer = sum(current_word_errors) / len(current_word_errors)
    candidate_wer = sum(candidate_word_errors) / len(candidate_word_errors)
    relative_wer_improvement = (current_wer - candidate_wer) / current_wer if current_wer else 0.0
    return {
        "clips": len(payload),
        "current_wer": current_wer,
        "candidate_wer": candidate_wer,
        "current_cer": sum(current_character_errors) / len(current_character_errors),
        "candidate_cer": sum(candidate_character_errors) / len(candidate_character_errors),
        "relative_wer_improvement": relative_wer_improvement,
        "passes_ten_percent_wer_gate": relative_wer_improvement >= 0.10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current and candidate Iraqi Arabic transcripts.")
    parser.add_argument("manifest", type=Path, help="Private JSON quality manifest.")
    args = parser.parse_args()
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = evaluate_manifest(payload)
    except (OSError, json.JSONDecodeError, QualityManifestError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
