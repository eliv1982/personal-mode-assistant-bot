from __future__ import annotations

import pytest

from text_utils import sanitize_ai_text, split_into_chunks


class TestSanitizeAiTextPreservesCode:
    def test_snake_case_variable_survives(self):
        assert sanitize_ai_text("my_var_name") == "my_var_name"

    def test_snake_case_in_sentence_survives(self):
        text = "Переименуй my_var_name в new_var_name."
        assert sanitize_ai_text(text) == text

    def test_filename_with_underscores_survives(self):
        assert sanitize_ai_text("path_to_file_v2") == "path_to_file_v2"

    def test_filename_with_extension_survives(self):
        text = "Открой path_to_file_v2.py и посмотри на импорт."
        assert sanitize_ai_text(text) == text

    def test_multiplication_syntax_survives(self):
        assert sanitize_ai_text("2*3*4") == "2*3*4"

    def test_multiplication_in_sentence_survives(self):
        text = "Результат: 2*3*4 = 24."
        assert sanitize_ai_text(text) == text

    def test_single_exponent_survives(self):
        assert sanitize_ai_text("2**3") == "2**3"

    def test_chained_exponent_survives(self):
        # Regression: the `**` bold rule used to greedily match the first
        # `**...**` span it found, corrupting `2**3**4` into `234`.
        assert sanitize_ai_text("2**3**4") == "2**3**4"

    def test_variable_exponent_survives(self):
        assert sanitize_ai_text("value**2") == "value**2"

    def test_chained_exponent_in_sentence_survives(self):
        text = "Результат: 2**3**4 = 4096."
        assert sanitize_ai_text(text) == text


class TestSanitizeAiTextCleansMarkdown:
    def test_bold_markers_stripped(self):
        assert sanitize_ai_text("**important**") == "important"

    def test_bold_word_in_prose_stripped(self):
        assert sanitize_ai_text("Это **важно** для дела.") == "Это важно для дела."

    def test_heading_markers_stripped(self):
        assert sanitize_ai_text("## Заголовок") == "Заголовок"

    def test_code_fence_lines_removed_content_kept(self):
        text = "```python\nprint(1)\n```"
        result = sanitize_ai_text(text)
        assert "```" not in result
        assert "print(1)" in result

    def test_dash_bullet_normalised(self):
        assert sanitize_ai_text("- item one\n- item two") == "• item one\n• item two"

    def test_inline_code_backticks_stripped(self):
        assert sanitize_ai_text("Используй `print()` для вывода.") == "Используй print() для вывода."

    def test_italic_word_in_prose_stripped(self):
        assert sanitize_ai_text("Это *очень* важно.") == "Это очень важно."

    def test_excess_blank_lines_collapsed(self):
        assert sanitize_ai_text("a\n\n\n\nb") == "a\n\nb"


class TestSplitIntoChunks:
    def test_short_text_returns_single_chunk(self):
        assert split_into_chunks("hello", 4096) == ["hello"]

    def test_empty_text_returns_no_chunks(self):
        assert split_into_chunks("", 4096) == []

    def test_long_text_split_into_multiple_ordered_chunks(self):
        text = "word " * 2000  # well over 4096 chars
        chunks = split_into_chunks(text, 4096)
        assert len(chunks) > 1
        assert all(len(c) <= 4096 for c in chunks)
        assert "".join(chunks) == text

    def test_chunk_boundaries_never_exceed_limit(self):
        text = "x" * 10000
        chunks = split_into_chunks(text, 100)
        assert all(len(c) <= 100 for c in chunks)
        assert "".join(chunks) == text

    def test_preserves_complete_text_and_ordering(self):
        text = "".join(f"line {i}\n" for i in range(500))
        chunks = split_into_chunks(text, 250)
        assert "".join(chunks) == text
        # Ordering: reconstructed text must match line-for-line.
        assert "".join(chunks).splitlines() == text.splitlines()

    def test_prefers_newline_boundary_over_hard_cut(self):
        text = ("a" * 50) + "\n" + ("b" * 50)
        chunks = split_into_chunks(text, 55)
        assert chunks[0] == ("a" * 50) + "\n"
        assert "".join(chunks) == text

    def test_invalid_limit_raises(self):
        with pytest.raises(ValueError):
            split_into_chunks("hello", 0)


class TestSplitIntoChunksBoundary:
    """Regression cases pinned to Telegram's exact 4096-character message limit."""

    def test_exactly_4096_chars_returns_one_chunk(self):
        text = "x" * 4096
        chunks = split_into_chunks(text, 4096)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_4097_chars_returns_multiple_chunks(self):
        text = "x" * 4097
        chunks = split_into_chunks(text, 4096)
        assert len(chunks) > 1
        assert all(len(c) <= 4096 for c in chunks)
        assert "".join(chunks) == text
