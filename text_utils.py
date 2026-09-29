from __future__ import annotations

import re

# Matches ``` optionally followed by a language tag on the same line
_CODE_FENCE = re.compile(r"^```[^\n]*$", re.MULTILINE)

# Markdown headings: ### text, ## text, # text at line start
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Bold: **text** or __text__ — guarded the same way as the italic patterns
# below (?<!\w) / (?!\w) so that exponentiation chains like `2**3**4` are not
# misread as `2` + bold("3") + `4`: real Markdown bold markers are never
# glued directly to a preceding/following word character, but exponent `**`
# operators always are.
_BOLD = re.compile(
    r"(?<!\w)\*\*(?!\*|\s)(.+?)(?<!\s)\*\*(?!\w)|(?<!\w)__(?!_|\s)(.+?)(?<!\s)__(?!\w)",
    re.DOTALL,
)

# Italic: *text* or _text_  — but only when NOT a bullet list marker and NOT
# flanked by word characters. The word-boundary guard (?<!\w) / (?!\w) is what
# keeps identifiers like `my_var_name` / `path_to_file_v2` and expressions like
# `2*3*4` intact: their delimiters sit directly against letters/digits, so they
# never qualify as emphasis markers, unlike "*word*" or "_word_" in prose.
_ITALIC_STAR = re.compile(r"(?<!\w)(?<!\*)\*(?!\*|\s)(.+?)(?<!\s)\*(?!\*)(?!\w)")
_ITALIC_UNDER = re.compile(r"(?<!\w)(?<!_)_(?!_|\s)(.+?)(?<!\s)_(?!_)(?!\w)")

# Inline code: `text`
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

# Bullet "-" → "•"  (lines starting with "- " or "* " but not inside a word)
_DASH_BULLET = re.compile(r"^- ", re.MULTILINE)
_STAR_BULLET = re.compile(r"^\* ", re.MULTILINE)

# Three or more consecutive blank lines → one blank line
_EXCESS_BLANKS = re.compile(r"\n{3,}")


def sanitize_ai_text(text: str) -> str:
    """Strip Markdown artefacts from an LLM response before sending to Telegram plain text."""

    # 1. Remove code-fence lines, keep content between them
    text = _CODE_FENCE.sub("", text)

    # 2. Strip heading markers (### → plain line)
    text = _HEADING.sub("", text)

    # 3. Strip bold markers
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2), text)

    # 4. Strip italic markers (skip bullet lines)
    text = _ITALIC_STAR.sub(r"\1", text)
    text = _ITALIC_UNDER.sub(r"\1", text)

    # 5. Strip inline code backticks
    text = _INLINE_CODE.sub(r"\1", text)

    # 6. Normalise bullet dashes/stars → •
    text = _DASH_BULLET.sub("• ", text)
    text = _STAR_BULLET.sub("• ", text)

    # 7. Collapse excess blank lines
    text = _EXCESS_BLANKS.sub("\n\n", text)

    return text.strip()


def split_into_chunks(text: str, limit: int) -> list[str]:
    """Split `text` into chunks of at most `limit` characters each.

    Joining the returned chunks in order reproduces `text` exactly (no
    characters are dropped, added, or reordered). Prefers to break on a
    newline or space near the limit so words aren't split mid-token when
    avoidable; falls back to a hard cut when no such boundary exists.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + limit, n)
        if end == n:
            split_at = end
        else:
            split_at = text.rfind("\n", start, end)
            if split_at <= start:
                split_at = text.rfind(" ", start, end)
            if split_at <= start:
                split_at = end
            else:
                split_at += 1  # keep the boundary character in this chunk
        chunks.append(text[start:split_at])
        start = split_at
    return chunks
