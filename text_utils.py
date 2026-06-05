from __future__ import annotations

import re

# Matches ``` optionally followed by a language tag on the same line
_CODE_FENCE = re.compile(r"^```[^\n]*$", re.MULTILINE)

# Markdown headings: ### text, ## text, # text at line start
_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)

# Bold: **text** or __text__
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.DOTALL)

# Italic: *text* or _text_  — but only when NOT a bullet list marker
# A bullet list line looks like "* item" or "- item" at the start of the line;
# we want to keep those intact, so we only strip *…* when it wraps actual content.
_ITALIC_STAR = re.compile(r"(?<!\*)\*(?!\*|\s)(.+?)(?<!\s)\*(?!\*)")
_ITALIC_UNDER = re.compile(r"(?<!_)_(?!_|\s)(.+?)(?<!\s)_(?!_)")

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
