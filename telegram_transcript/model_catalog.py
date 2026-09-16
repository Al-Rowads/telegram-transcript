"""Pinned model choices and migrations for the cost-conscious defaults."""

GEMINI_FLASH_LITE_MODEL = "google/gemini-2.5-flash-lite"
QWEN_TRANSLATION_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
GEMINI_FLASH_MODEL = "google/gemini-3.5-flash"
RETIRED_TEXT_MODELS = frozenset({
    "openai/gpt-5.5",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-4.6",
})


def normalize_legacy_text_model(model: str) -> str:
    # Pin migration targets independently of future changes to the defaults.
    return GEMINI_FLASH_LITE_MODEL if model in RETIRED_TEXT_MODELS else model
