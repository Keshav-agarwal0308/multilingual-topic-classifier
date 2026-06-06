"""
preprocess.py — Text preprocessing utilities for the Multilingual Topic Classifier
Cleans and normalizes input text before passing it to the classifier pipeline.
"""

import re
import unicodedata


def remove_urls(text: str) -> str:
    """Remove http/https URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from text."""
    return re.sub(r"<[^>]+>", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_emojis(text: str) -> str:
    """Remove emoji characters from text."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to NFC form (handles accents, ligatures, etc.)."""
    return unicodedata.normalize("NFC", text)


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special/non-alphanumeric characters.
    Set keep_punctuation=False to strip ALL punctuation too.
    """
    if keep_punctuation:
        # Keep letters, digits, whitespace, and common punctuation
        return re.sub(r"[^\w\s.,!?;:'\"()\-]", "", text)
    else:
        return re.sub(r"[^\w\s]", "", text)


def truncate_text(text: str, max_chars: int = 512) -> str:
    """
    Truncate text to max_chars characters.
    XLM-RoBERTa has a 512-token limit; truncating long inputs avoids silent errors.
    """
    return text[:max_chars] if len(text) > max_chars else text


def preprocess(
    text: str,
    remove_urls_flag: bool = True,
    remove_html_flag: bool = True,
    remove_emoji_flag: bool = False,
    normalize_unicode_flag: bool = True,
    remove_special_chars_flag: bool = False,
    keep_punctuation: bool = True,
    max_chars: int = 512,
) -> str:
    """
    Full preprocessing pipeline. Apply steps in a sensible order.

    Args:
        text                    : Raw input string.
        remove_urls_flag        : Strip URLs (default True).
        remove_html_flag        : Strip HTML tags (default True).
        remove_emoji_flag       : Strip emojis (default False — emojis can signal topic).
        normalize_unicode_flag  : NFC normalize (default True).
        remove_special_chars_flag: Strip non-alphanumeric chars (default False).
        keep_punctuation        : When removing special chars, keep punctuation (default True).
        max_chars               : Truncate to this many characters (default 512).

    Returns:
        Cleaned string ready for the classifier.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    if remove_html_flag:
        text = remove_html_tags(text)
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_emoji_flag:
        text = remove_emojis(text)
    if normalize_unicode_flag:
        text = normalize_unicode(text)
    if remove_special_chars_flag:
        text = remove_special_characters(text, keep_punctuation=keep_punctuation)

    text = remove_extra_whitespace(text)
    text = truncate_text(text, max_chars=max_chars)

    return text


# ── Quick demo ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Check out https://example.com for the latest <b>health</b> news!! 🏥🏥",
        "  Le parlement   a voté\n\nnew loi   sur l'environnement.  ",
        "<p>Scientists discovered a new exoplanet 🌌 — visit www.nasa.gov</p>",
        "ਕ੍ਰਿਕੇਟ ਟੀਮ ਨੇ ਵਿਸ਼ਵ ਕੱਪ ਜਿੱਤਿਆ। 🏆🏆🏆",
    ]

    print("=" * 60)
    print("preprocess.py — Multilingual Topic Classifier")
    print("=" * 60)

    for raw in samples:
        cleaned = preprocess(raw)
        print(f"\nRAW     : {raw}")
        print(f"CLEANED : {cleaned}")

    print("\n" + "=" * 60)
    print("All samples processed successfully.")
