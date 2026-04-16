from __future__ import annotations

from typing import List

try:
    from fugashi import Tagger
except Exception:  # pragma: no cover
    Tagger = None


_TAGGER = Tagger() if Tagger else None


def _fallback_ja_tokens(text: str) -> List[str]:
    text = text.strip()
    return [ch for ch in text if ch.strip()]


def tokenize_for_ui(text: str, language: str = "ja") -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []

    if language.startswith("ja"):
        if _TAGGER is None:
            return _fallback_ja_tokens(cleaned)
        tokens = [tok.surface for tok in _TAGGER(cleaned)]
        return [t for t in tokens if t.strip()]

    return cleaned.split()
