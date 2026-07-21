from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
NUMBER_RE = re.compile(r"[0-9\u0660-\u0669\u06f0-\u06f9]+(?:[.,،٫][0-9\u0660-\u0669\u06f0-\u06f9]+)*")


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


def error_counts(reference: str, candidate: str) -> tuple[int, int, int, int]:
    reference_normalized = normalize_arabic_for_evaluation(reference)
    candidate_normalized = normalize_arabic_for_evaluation(candidate)
    reference_words = reference_normalized.split()
    reference_characters = list(reference_normalized.replace(" ", ""))
    if not reference_words or not reference_characters:
        raise ValueError("Reference transcript must contain Arabic text.")
    return (
        edit_distance(reference_words, candidate_normalized.split()),
        len(reference_words),
        edit_distance(reference_characters, list(candidate_normalized.replace(" ", ""))),
        len(reference_characters),
    )


def contains_term(text: str, term: str) -> bool:
    normalized_text = f" {normalize_arabic_for_evaluation(text)} "
    normalized_term = normalize_arabic_for_evaluation(term)
    return bool(normalized_term) and f" {normalized_term} " in normalized_text


def number_tokens(text: str) -> tuple[str, ...]:
    digit_map = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹،٫", "01234567890123456789.,")
    return tuple(match.group(0).translate(digit_map) for match in NUMBER_RE.finditer(text))


def evaluate_manifest(payload: Any) -> dict[str, object]:
    if not isinstance(payload, list) or not payload:
        raise QualityManifestError("Quality manifest must be a non-empty JSON array.")

    current_word_errors = 0
    candidate_word_errors = 0
    reference_words = 0
    current_character_errors = 0
    candidate_character_errors = 0
    reference_characters = 0
    current_critical_hits = 0
    candidate_critical_hits = 0
    critical_terms_total = 0
    current_number_matches = 0
    candidate_number_matches = 0
    numbered_clips = 0
    clip_results: list[dict[str, object]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise QualityManifestError(f"Quality manifest item {index} must be an object.")
        required = {"id", "reference_arabic", "current_transcript", "candidate_transcript"}
        if not required.issubset(item) or not all(isinstance(item[key], str) for key in required):
            raise QualityManifestError(f"Quality manifest item {index} is missing required string fields.")
        terms = item.get("critical_terms", [])
        if not isinstance(terms, list) or not all(isinstance(term, str) and term.strip() for term in terms):
            raise QualityManifestError(
                f"Quality manifest item {index} critical_terms must be an array of non-empty strings."
            )
        current_counts = error_counts(item["reference_arabic"], item["current_transcript"])
        candidate_counts = error_counts(item["reference_arabic"], item["candidate_transcript"])
        current_word_errors += current_counts[0]
        candidate_word_errors += candidate_counts[0]
        reference_words += current_counts[1]
        current_character_errors += current_counts[2]
        candidate_character_errors += candidate_counts[2]
        reference_characters += current_counts[3]

        current_term_hits = sum(contains_term(item["current_transcript"], term) for term in terms)
        candidate_term_hits = sum(contains_term(item["candidate_transcript"], term) for term in terms)
        current_critical_hits += current_term_hits
        candidate_critical_hits += candidate_term_hits
        critical_terms_total += len(terms)

        reference_numbers = number_tokens(item["reference_arabic"])
        if reference_numbers:
            numbered_clips += 1
            current_number_matches += number_tokens(item["current_transcript"]) == reference_numbers
            candidate_number_matches += number_tokens(item["candidate_transcript"]) == reference_numbers

        clip_results.append(
            {
                "id": item["id"],
                "current_wer": current_counts[0] / current_counts[1],
                "candidate_wer": candidate_counts[0] / candidate_counts[1],
                "current_cer": current_counts[2] / current_counts[3],
                "candidate_cer": candidate_counts[2] / candidate_counts[3],
                "current_critical_terms": current_term_hits,
                "candidate_critical_terms": candidate_term_hits,
                "critical_terms": len(terms),
            }
        )

    current_wer = current_word_errors / reference_words
    candidate_wer = candidate_word_errors / reference_words
    relative_wer_improvement = (current_wer - candidate_wer) / current_wer if current_wer else 0.0
    current_critical_recall = (
        current_critical_hits / critical_terms_total if critical_terms_total else 1.0
    )
    candidate_critical_recall = (
        candidate_critical_hits / critical_terms_total if critical_terms_total else 1.0
    )
    current_number_preservation = current_number_matches / numbered_clips if numbered_clips else 1.0
    candidate_number_preservation = candidate_number_matches / numbered_clips if numbered_clips else 1.0
    passes_gate = (
        relative_wer_improvement >= 0.10
        and candidate_critical_recall >= current_critical_recall
        and candidate_number_preservation >= current_number_preservation
    )
    return {
        "clips": len(payload),
        "current_wer": current_wer,
        "candidate_wer": candidate_wer,
        "current_cer": current_character_errors / reference_characters,
        "candidate_cer": candidate_character_errors / reference_characters,
        "relative_wer_improvement": relative_wer_improvement,
        "current_critical_term_recall": current_critical_recall,
        "candidate_critical_term_recall": candidate_critical_recall,
        "current_number_preservation": current_number_preservation,
        "candidate_number_preservation": candidate_number_preservation,
        "passes_ten_percent_wer_gate": passes_gate,
        "clip_results": clip_results,
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
